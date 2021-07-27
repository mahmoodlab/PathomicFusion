import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from resnet_custom import *
import pdb
import math
from pixelcnn import MaskCNN

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



def initialize_weights(module):
	"""
	args:
		module: any pytorch module with trainable parameters
	"""

	for m in module.modules():
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
			if m.bias is not None:
				m.bias.data.zero_()

		# if isinstance(m, nn.Linear):
		# 	nn.init.xavier_normal_(m.weight)
		# 	m.bias.data.zero_()

		if isinstance(m, nn.Linear):
			nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm2d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)


class CPC_model(nn.Module):
	def __init__(self, input_size = 1024, hidden_size = 128, k = 3, ln = False):
		"""
		args:
			input_size: input size to autoregresser (encoding size)
			hidden_size: number of hidden units in MaskedCNN
			num_layers: number of hidden layers in MaskedCNN
			k: prediction length
		"""
		super(CPC_model, self).__init__()
		
		### Settings
		self.seq_len = 49 # 7 x 7 grid of overlapping 64 x 64 patches extracted from each 256 x 256 image
		self.k = k 
		self.input_size = input_size
		self.hidden_size=hidden_size


		### Networks
		if ln:
			self.encoder = resnet50_ln(pretrained=False)
		else:
			self.encoder = resnet50(pretrained=False)
		self.reg = MaskCNN(n_channel=self.input_size, h=self.hidden_size)
		network_pred = [nn.Linear(input_size, input_size) for i in range(self.k)] #use an indepdent linear layer to predict each future row	
		self.network_pred= nn.ModuleList(network_pred)
		
		# initialize linear network and context network
		initialize_weights(self.network_pred)
		initialize_weights(self.reg)


		### Activation functions
		self.softmax  = nn.Softmax(dim=1)
		self.lsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, x):
		# input = [bs * 7 * 7, 3, 64, 64]
		
		# compute batch_size
		bs = x.size(0) // (self.seq_len)

		rows = int(math.sqrt(self.seq_len))
		cols = int(math.sqrt(self.seq_len))

		# compute latent representation for each patch
		z = self.encoder(x)
		# z.shape: [bs * 7 * 7, 1024]
		
		# reshape z into feature grid: [bs, 7, 7, 1024]
		z = z.contiguous().view(bs, rows, cols, self.input_size)
		
		device = z.device

		#randomly draw a row to predict what is k rows below it, using information in current row and above
		if self.training:
			pred_id = torch.randint(rows - self.k, size=(1,)).long() #low is 0, high is 3 (predicts row 4, 5, 6)
		
		else:
			pred_id = torch.tensor([3]).long()
		
		# feature predictions for the next k rows  e.g.  pred[i] is [bs * cols, 1024] for i in k
		pred = [torch.empty(bs * cols, self.input_size).float().to(device) for i in range(self.k)]

		# ground truth encodings for the next k rows e.g. encode_samples[i] is [bs * cols, 1024] for i in k
		encode_samples = [torch.empty(bs * cols, self.input_size).float().to(device) for i in range(self.k)]

		for i in np.arange(self.k):
			# add ground truth encodings
			start_row = pred_id.item()+i+1
			encode_samples[i] = z[:,start_row, :, :].contiguous().view(bs * cols, self.input_size)

		# reshape feature grid to channel first (required by Pytorch convolution convention)
		z = z.permute(0, 3, 1, 2) 
		# z.shape: from [bs, 7, 7, 1024] --> [bs, 1024, 7, 7]
		
		# apply aggregation to compute context 
		output = self.reg(z)
		# reg is fully convolutional --> output size is [bs, 1024, 7, 7]

		output = output.permute(0, 2, 3, 1) # reshape back to feature grid
		# output.shape: [bs, row, col, 1024]
		
		# context for each patch in the row
		c_t = output[:,pred_id + 1,:, :]
		# c_t.shape: [bs, 1, 7, 1024]

		# reshape for linear classification:
		c_t = c_t.contiguous().view(bs * cols, self.input_size)
		# c_t.shape: [bs * cols, 1024]

		# linear prediction: Wk*c_t
		for i in np.arange(0, self.k):
			if type(self.network_pred) == nn.DataParallel:
				pred[i] = self.network_pred.module[i](c_t)
			
			else:
				pred[i] = self.network_pred[i](c_t)  #e.g. size [bs * cols, 1024]

		nce = 0 # average over prediction length, cols, and batch 
		accuracy = np.zeros((self.k,))
		
		for i in np.arange(0, self.k):
			"""
			goal: can network correctly match predicted features with ground truth features among negative targets 
			i.e. match z_i+k,j with W_k * c_i,j
			postivie target: patch with the correct groundtruth encoding
			negative targets: patches with wrong groundtruth encodings (sampled from other patches in the same image, or other images in the minibatch)

			1) dot product for each k to obtain raw prediction logits 
			total = (a_ij) = [bs * col, bs * col], where a_ij is the logit of ith patch prediction matching jth patch encoding
			
			2) apply softmax along each row to get probability that ith patch prediction matches jth patch encoding 
			we want ith patch prediction to correctly match ith patch encoding, therefore target has 1s along diagnol, and 0s off diagnol

			3) we take the argmax along softmaxed rows to get the patch prediction for the ith patch, this value should be i

			4) compute nce loss as the cross-entropy of classifying the positive sample correctly (sum of logsoftmax along diagnol)

			5) normalize loss by batchsize and k and number of patches in a row
			
			"""
			total = torch.mm(pred[i], torch.transpose(encode_samples[i],0,1)) # e.g. size [bs * col, bs * col]

			accuracy[i] = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=1), torch.arange(0, bs * cols).to(device))).item() 
			accuracy[i] /= 1. * (bs * cols) 
			
			nce += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor
		
		nce /= -1. * bs * cols * self.k
		# accuracy = 1.*correct.item() / (bs * cols * self.k)
		
		return nce, np.array(accuracy)


# crop data into 64 by 64 with 32 overlap 
def cropdata(data, num_channels=3, kernel_size = 64, stride = 32):
	if len(data.shape) == 3:
		data = data.unsqueeze(0)

	data = data.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
	data = data.permute(0,2,3,1,4,5)
	data = data.contiguous().view(-1, num_channels, kernel_size, kernel_size)
	return data

if __name__ == '__main__':
	torch.set_printoptions(threshold=1e6)
	x = torch.rand(2, 3, 256, 256)
	x = cropdata(x)
	print(x.shape)
	model = CPC_model(1024, 256)
	nce, accuracy = model(x)


