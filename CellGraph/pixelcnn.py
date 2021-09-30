import torch.nn as nn
from layers_custom import maskConv0, MaskConvBlock
import torch

class MaskCNN(nn.Module):
    def __init__(self, n_channel=1024, h=128):
        """PixelCNN Model"""
        super(MaskCNN, self).__init__()

        self.MaskConv0 = maskConv0(n_channel, h, k_size=7, stride=1, pad=3)
        # large 7 x 7 masked filter with image downshift to ensure that each output neuron's receptive field only sees what is above it in the image 


        MaskConv = []
        
        # stack of 10 gated residual masked conv blocks
        for i in range(10):
            MaskConv.append(MaskConvBlock(h, k_size=3, stride=1, pad=1))
        self.MaskConv = nn.Sequential(*MaskConv)

        # 1x1 conv to upsample to required feature (channel) length

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(h, n_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(n_channel),
            nn.ReLU()
            )


    def forward(self, x):
        """
        Args:
            x: [batch_size, channel, height, width]
        Return:
            out [batch_size, channel, height, width]
        """
        # fully convolutional, feature map dimension maintained constant throughout
        x = self.MaskConv0(x)

        x = self.MaskConv(x)

        x = self.out(x)

        return x

if __name__ == '__main__':
    from torchsummary import summary
    model = PixelCNN(1024, 128)
    summary(model, (1024, 7,7))
    x = torch.rand(2, 1024, 7, 7)
    x = model(x)
    print(x.shape)

