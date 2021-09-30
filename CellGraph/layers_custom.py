import torch
import torch.nn as nn
import pdb

def down_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when downshifting, the last row is removed 
    x = x[:, :, :xs[2] - 1, :]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((0, 0, 1, 0)) if pad is None else pad
    return pad(x)

class MaskedConv2d(nn.Conv2d):
    def __init__(self, c_in, c_out, k_size, stride, pad, use_down_shift=False):
        super(MaskedConv2d, self).__init__(
            c_in, c_out, k_size, stride, pad, bias=False)

        ch_out, ch_in, height, width = self.weight.size()

                # Mask
        #         -------------------------------------
        #        |  1       1       1       1       1 |
        #        |  1       1       1       1       1 |
        #        |  1       1       1       1       1 |   H // 2
        #        |  0       0       0       0       0 |   H // 2 + 1
        #        |  0       0       0       0       0 |
        #         -------------------------------------
        #  index    0       1     W//2    W//2+1
        
        mask = torch.ones(ch_out, ch_in, height, width)
        mask[:, :, height // 2 + 1:] = 0
        self.register_buffer('mask', mask)
        self.use_down_shift = use_down_shift

    def forward(self, x):
        self.weight.data *= self.mask
        if self.use_down_shift:
            x = down_shift(x)
        return super(MaskedConv2d, self).forward(x)

def maskConv0(c_in=3, c_out=256, k_size=7, stride=1, pad=3):
    """2D Masked Convolution first layer"""
    return nn.Sequential(
        MaskedConv2d(c_in, c_out * 2, k_size, stride, pad, use_down_shift=True),
        nn.BatchNorm2d(c_out * 2),
        Gate()
        )

class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()

    def forward(self, x):
        # gated activation 
        xf, xg = torch.chunk(x, 2, dim=1)
        f = torch.tanh(xf)
        g = torch.sigmoid(xg)
        return f * g

class MaskConvBlock(nn.Module):
    def __init__(self, h=128, k_size=3, stride=1, pad=1):
        """1x1 Conv + 2D Masked Convolution (type B) + 1x1 Conv"""
        super(MaskConvBlock, self).__init__()

        self.net = nn.Sequential(
            MaskedConv2d(h, 2 * h, k_size, stride, pad),
            nn.BatchNorm2d(2 * h),
            Gate()
        )

    def forward(self, x):
        """Residual connection"""
        return self.net(x) + x

if __name__ == '__main__':
    def conv(x, kernel):
        return nn.functional.conv2d(x, kernel, padding=1)
    x = torch.ones((1, 1, 5, 5)) * 0.1
    x[:,:,1,0] = 1000
    
    print("blindspot experiment")
    normal_kernel = torch.ones(1, 1, 3, 3)
    mask_kernel = torch.zeros(1, 1, 3, 3)
    mask_kernel[:,:,0,:] = 1
    mask_b = mask_kernel.clone()
    mask_b[:,:,1,1] = 1
    # mask_kernel[:,:,1,1] = 1

    print("unmasked kernel:", "\n",normal_kernel.squeeze(), "\n")
    print("masked kernel:", "\n", mask_kernel.squeeze(), "\n")

    print("normal conv")
    print("orig image", "\n", x.squeeze(), "\n")

    y = conv(x, normal_kernel)
    print(y[:,0, :,:], "\n")

    y = conv(y, normal_kernel)
    print(y[:,0, :,:], "\n")

    print("with mask")
    print("orig image", "\n", x.squeeze(), "\n")

    y = conv(x, mask_kernel)
    print(y[:,0, :,:], "\n")
    
    y = conv(y, mask_b)
    print(y[:,0, :,:], "\n")

    y = conv(y, mask_b)
    print(y[:,0, :,:],"\n")

    print("with down_shift")
    print("orig image", x.squeeze(), "\n")
    c_kernel = mask_kernel
    c_kernel[:,:,1,:] = 1

    print("custom kernel:", "\n", c_kernel.squeeze(), "\n")
    y = conv(down_shift(x), c_kernel)
    print(y[:,0, :,:],"\n")
    y = conv(y, c_kernel)
    print(y[:,0, :,:],"\n")
    y = conv(y, c_kernel)
    print(y[:,0, :,:],"\n")
    y = conv(y, c_kernel)
    print(y[:,0, :,:],"\n")





