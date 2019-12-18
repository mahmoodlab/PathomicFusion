### data_loaders.py
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing

# Env
from networks import define_net
from utils import getCleanAllDataset
import torch
from torchvision import transforms
from options import parse_gpuids

### Initializes parser and data
"""
all_st
python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 0 --use_vgg_features 0 --roi_dir all_st # for training Surv Path, Surv Graph, and testing Surv Graph
python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 1 --use_vgg_features 0 --roi_dir all_st # for training Grad Path, Grad Graph, and testing Surv_graph
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 0 --roi_dir all_st # for training Surv Omic, Surv Graphomic
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 1 --use_vgg_features 0 --roi_dir all_st # for training Grad Omic, Grad Graphomic

all_st_patches_512 (no VGG)
python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 0 --use_vgg_features 0 --roi_dir all_st_patches_512 # for testing Surv Path
python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 1 --use_vgg_features 0 --roi_dir all_st_patches_512 # for testing Grad Path

all_st_patches_512 (use VGG)
python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 0 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name surv_15 --gpu_ids 0 # for Surv Pathgraph
python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 1 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name grad_15 --act_type LSM --label_dim 3 --gpu_ids 1 # for Grad Pathgraph
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name surv_15 --gpu_ids 2 # for Surv Pathomic, Pathgraphomic
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 1 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name grad_15 --act_type LSM --label_dim 3 --gpu_ids 3 # for Grad Pathomic, Pathgraphomic


python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 1 --make_all_train 1

python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 0 --roi_dir all_st --use_rnaseq 1
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 1 --use_vgg_features 0 --roi_dir all_st --use_rnaseq 1
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name surv_15 --use_rnaseq 1 --gpu_ids 2
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 1 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name grad_15 --use_rnaseq 1 --act_type LSM --label_dim 3 --gpu_ids 3


python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 0 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name surv_15_rnaseq --gpu_ids 0
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name surv_15_rnaseq --use_rnaseq 1 --gpu_ids 0

python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 1 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name grad_15 --act_type LSM --label_dim 3 --gpu_ids 1
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 1 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name grad_15 --use_rnaseq 1 --act_type LSM --label_dim 3 --gpu_ids 1

python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 0 --roi_dir all_st --use_rnaseq 1
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name surv_15_rnaseq --gpu_ids 2

python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 1 --use_vgg_features 0 --roi_dir all_st --use_rnaseq 1
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 1 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name grad_15 --act_type LSM --label_dim 3 --gpu_ids 3




"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./data/TCGA_GBMLGG/', help="datasets")
    parser.add_argument('--roi_dir', type=str, default='all_st')
    parser.add_argument('--graph_feat_type', type=str, default='cpc', help="graph features to use")
    parser.add_argument('--ignore_missing_moltype', type=int, default=0, help="Ignore data points with missing molecular subtype")
    parser.add_argument('--ignore_missing_histype', type=int, default=0, help="Ignore data points with missign histology subtype")
    parser.add_argument('--make_all_train', type=int, default=0)
    parser.add_argument('--use_vgg_features', type=int, default=0)
    parser.add_argument('--use_rnaseq', type=int, default=0)


    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/TCGA_GBMLGG/', help='models are saved here')
    parser.add_argument('--exp_name', type=str, default='surv_15_rnaseq', help='name of the project. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--mode', type=str, default='path', help='mode')
    parser.add_argument('--model_name', type=str, default='path', help='mode')
    parser.add_argument('--task', type=str, default='surv', help='surv | grad')
    parser.add_argument('--act_type', type=str, default='Sigmoid', help='activation function')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--label_dim', type=int, default=1, help='size of output')
    parser.add_argument('--batch_size', type=int, default=32, help="Number of batches to train/test for. Default: 256")
    parser.add_argument('--path_dim', type=int, default=32)
    parser.add_argument('--init_type', type=str, default='none', help='network initialization [normal | xavier | kaiming | orthogonal | max]. Max seems to work well')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='0 - 0.25. Increasing dropout_rate helps overfitting. Some people have gone as high as 0.5. You can try adding more regularization')

    opt = parser.parse_known_args()[0]
    opt = parse_gpuids(opt)
    return opt

opt = parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
metadata, all_dataset = getCleanAllDataset(opt.dataroot, opt.ignore_missing_moltype, opt.ignore_missing_histype, opt.use_rnaseq)

### Creates a mapping from TCGA ID -> Image ROI
img_fnames = os.listdir(os.path.join(opt.dataroot, opt.roi_dir))
pat2img = {}
for pat, img_fname in zip([img_fname[:12] for img_fname in img_fnames], img_fnames):
    if pat not in pat2img.keys(): pat2img[pat] = []
    pat2img[pat].append(img_fname)

### Dictionary file containing split information
data_dict = {}
data_dict['data_pd'] = all_dataset
#data_dict['pat2img'] = pat2img
#data_dict['img_fnames'] = img_fnames
cv_splits = {}

### Extracting K-Fold Splits
pnas_splits = pd.read_csv(opt.dataroot+'pnas_splits.csv')
pnas_splits.columns = ['TCGA ID']+[str(k) for k in range(1, 16)]
pnas_splits.index = pnas_splits['TCGA ID']
pnas_splits = pnas_splits.drop(['TCGA ID'], axis=1)

### get path_feats
def get_vgg_features(model, device, img_path):
    if model is None:
        return img_path
    else:
        x_path = Image.open(img_path).convert('RGB')
        normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        x_path = torch.unsqueeze(normalize(x_path), dim=0)
        features, hazard = model(x_path=x_path.to(device))
        return features.cpu().detach().numpy()

### method for constructing aligned
def getAlignedMultimodalData(opt, model, device, all_dataset, pat_split, pat2img):
    x_patname, x_path, x_grph, x_omic, e, t, g = [], [], [], [], [], [], []

    for pat_name in pat_split:
        if pat_name not in all_dataset.index: continue

        for img_fname in pat2img[pat_name]:
            grph_fname = img_fname.rstrip('.png')+'.pt'
            assert grph_fname in os.listdir(os.path.join(opt.dataroot, '%s_%s' % (opt.roi_dir, opt.graph_feat_type)))
            assert all_dataset[all_dataset['TCGA ID'] == pat_name].shape[0] == 1

            x_patname.append(pat_name)
            x_path.append(get_vgg_features(model, device, os.path.join(opt.dataroot, opt.roi_dir, img_fname)))
            x_grph.append(os.path.join(opt.dataroot, '%s_%s' % (opt.roi_dir, opt.graph_feat_type), grph_fname))
            x_omic.append(np.array(all_dataset[all_dataset['TCGA ID'] == pat_name].drop(metadata, axis=1)))
            e.append(int(all_dataset[all_dataset['TCGA ID']==pat_name]['censored']))
            t.append(int(all_dataset[all_dataset['TCGA ID']==pat_name]['Survival months']))
            g.append(int(all_dataset[all_dataset['TCGA ID']==pat_name]['Grade']))

    return x_patname, x_path, x_grph, x_omic, e, t, g

print(all_dataset.shape)

for k in pnas_splits.columns:
    print('Creating Split %s' % k)
    pat_train = pnas_splits.index[pnas_splits[k] == 'Train'] if opt.make_all_train == 0 else pnas_splits.index
    pat_test = pnas_splits.index[pnas_splits[k] == 'Test']
    cv_splits[int(k)] = {}

    model = None
    if opt.use_vgg_features:
        load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%s.pt' % (opt.model_name, k))
        model_ckpt = torch.load(load_path, map_location=device)
        model_state_dict = model_ckpt['model_state_dict']
        if hasattr(model_state_dict, '_metadata'): del model_state_dict._metadata
        model = define_net(opt, None)
        if isinstance(model, torch.nn.DataParallel): model = model.module
        print('Loading the model from %s' % load_path)
        model.load_state_dict(model_state_dict)
        model.eval()

    train_x_patname, train_x_path, train_x_grph, train_x_omic, train_e, train_t, train_g = getAlignedMultimodalData(opt, model, device, all_dataset, pat_train, pat2img)
    test_x_patname, test_x_path, test_x_grph, test_x_omic, test_e, test_t, test_g = getAlignedMultimodalData(opt, model, device, all_dataset, pat_test, pat2img)

    train_x_omic, train_e, train_t = np.array(train_x_omic).squeeze(axis=1), np.array(train_e, dtype=np.float64), np.array(train_t, dtype=np.float64)
    test_x_omic, test_e, test_t = np.array(test_x_omic).squeeze(axis=1), np.array(test_e, dtype=np.float64), np.array(test_t, dtype=np.float64)
        
    scaler = preprocessing.StandardScaler().fit(train_x_omic)
    train_x_omic = scaler.transform(train_x_omic)
    test_x_omic = scaler.transform(test_x_omic)

    train_data = {'x_patname': train_x_patname,
                  'x_path':np.array(train_x_path),
                  'x_grph':train_x_grph,
                  'x_omic':train_x_omic,
                  'e':np.array(train_e, dtype=np.float64), 
                  't':np.array(train_t, dtype=np.float64),
                  'g':np.array(train_g, dtype=np.float64)}

    test_data = {'x_patname': test_x_patname,
                 'x_path':np.array(test_x_path),
                 'x_grph':test_x_grph,
                 'x_omic':test_x_omic,
                 'e':np.array(test_e, dtype=np.float64),
                 't':np.array(test_t, dtype=np.float64),
                 'g':np.array(test_g, dtype=np.float64)}

    dataset = {'train':train_data, 'test':test_data}
    cv_splits[int(k)] = dataset

    if opt.make_all_train: break
    
data_dict['cv_splits'] = cv_splits

pickle.dump(data_dict, open('%s/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (opt.dataroot, opt.roi_dir, opt.ignore_missing_moltype, opt.ignore_missing_histype, opt.use_vgg_features, '_rnaseq' if opt.use_rnaseq else ''), 'wb'))