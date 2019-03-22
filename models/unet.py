import torch
import torchvision
import torch.nn as nn

import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch,bn=False):
        super(double_conv, self).__init__()
        # self.bn=bn
        # if bn:
        self.conv= nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # else:
        #     self.conv = nn.Sequential(
        #         nn.Conv2d(in_ch, out_ch, 3, padding=1),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(out_ch, out_ch, 3, padding=1),
        #         nn.ReLU(inplace=True),
        #     )

    def forward(self, x):
        x=self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch,bn=False):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch,bn)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch,bn=False):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch,bn),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False,bn=False):
        super(up, self).__init__()
        self.bilinear=bilinear
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if self.bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch,in_ch//2,1),)

        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch,bn)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    '''
    layer_nums mean num of layers of half of the Unet
    and the features change with ratio of 2
    '''
    def __init__(self,n_channels,layer_nums,features_root=64,output_channel=1,bn=False):
        super(UNet,self).__init__()
        self.inc = inconv(n_channels, 64,bn)
        self.down1 = down(64, 128,bn)
        self.down2 = down(128, 256,bn)
        self.down3 = down(256, 512,bn)
        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.outc = outconv(64, output_channel)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return torch.sigmoid(x)

'''
class SA_UNet(nn.Module):
'''
def _test():
    rand=torch.ones([4,12,256,256]).cuda()
    t=UNet(12,0,3).cuda()

    r=t(rand)
    print(r.grad_fn)
    print(r.requires_grad)

if __name__=='__main__':
    _test()
