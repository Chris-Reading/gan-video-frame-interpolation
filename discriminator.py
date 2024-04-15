import torch.nn as nn

class VimeoDiscriminator2(nn.Module):

    def __init__(self):
        super(VimeoDiscriminator2, self).__init__()
        self.nfg = 32  # the size of feature map
        self.c = 3
        
        self.conv_blocks = nn.Sequential(
            # input is c * 64 * 64
            nn.Conv2d(self.c, self.nfg, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.15),
            # state: nfg * 32 * 32
            nn.Conv2d(self.nfg, self.nfg * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.nfg * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(self.nfg * 2, self.nfg * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.nfg * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(self.nfg * 4, self.nfg * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.nfg * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(self.nfg * 8, self.nfg * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.nfg * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(self.nfg * 16, self.nfg * 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.nfg * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(self.nfg * 32, self.nfg * 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.nfg * 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(self.nfg * 64, 1, kernel_size=4, stride=1, padding=0, bias=False)
            )
    
    def forward(self, data):
        return self.conv_blocks(data)
    
class Discriminator2DVimeo(nn.Module):
    def __init__(self, colour_channels, features_d):
        super(Discriminator2DVimeo, self).__init__()
        # Input of dimensions: N x colour_channels x 512 x 512
        self.disc = nn.Sequential(
            nn.Conv2d(
                colour_channels, features_d, kernel_size=4, stride=2, padding=1
            ), #4
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1), #8
            self._block(features_d*2, features_d*4, 4, 2, 1), #16
            self._block(features_d*4, features_d*8, 4, 2, 1), #32
            self._block(features_d*8, features_d*16, 4, 2, 1), #64
            self._block(features_d*16, features_d*32, 4, 2, 1), #128
            self._block(features_d*32, features_d*64, 4, 2, 1), #256

            nn.Conv2d(features_d*64, 1, kernel_size=4, stride=2, padding=0), #64
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.disc(x)
    
class Discriminator2DMSU(nn.Module):
    def __init__(self, colour_channels, features_d):
        super(Discriminator2DMSU, self).__init__()
        # Input of dimensions: N x colour_channels x 2048 x 2048
        self.disc = nn.Sequential(
            nn.Conv2d(
                colour_channels, features_d, kernel_size=4, stride=2, padding=1
            ), #4
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1), #8
            self._block(features_d*2, features_d*4, 4, 2, 1), #16
            self._block(features_d*4, features_d*8, 4, 2, 1), #32
            self._block(features_d*8, features_d*16, 4, 2, 1), #64
            self._block(features_d*16, features_d*32, 4, 2, 1), #128
            self._block(features_d*32, features_d*64, 4, 2, 1), #256
            self._block(features_d*64, features_d*128, 4, 2, 1), #512
            self._block(features_d*128, features_d*256, 4, 2, 1), #1024

            nn.Conv2d(features_d*256, 1, kernel_size=4, stride=2, padding=0), #2048
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.disc(x)
