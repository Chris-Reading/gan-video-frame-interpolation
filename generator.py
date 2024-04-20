import torch.nn as nn
import torch

class VimeoGenerator2(nn.Module):
    def __init__(self):
        super(VimeoGenerator2, self).__init__()
        self.feature_groups = 32  # the size of the feature map
        self.channels = 3 # target channels (rgb)
        filter_size = 4 # params
        stride_size = 2 #params
        
        self.down_blocks = nn.Sequential( # stacks neural network layers
            nn.Conv2d(self.channels * 2, self.feature_groups * 2, kernel_size=3, stride=1, padding=1, bias=False),  # increases number of channels from 6 to 64
            nn.BatchNorm2d(self.feature_groups * 2), # normalizes activations using stdev and mean
            nn.LeakyReLU(0.02, inplace=True), # allows small negative inputs through
            nn.Conv2d(self.feature_groups * 2, self.feature_groups * 2, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_groups * 2),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(self.feature_groups * 2, self.feature_groups * 4, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_groups * 4),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(self.feature_groups * 4, self.feature_groups * 8, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_groups * 8),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(self.feature_groups * 8, self.feature_groups * 16, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_groups * 16),
            nn.LeakyReLU(0.02, inplace=True),
            )
        
        self.up_block = nn.Sequential(
            nn.ConvTranspose2d(self.feature_groups * 16, self.feature_groups * 8, kernel_size=filter_size, stride=stride_size, padding=1, bias=False), # decreases channels by 2
            nn.BatchNorm2d(self.feature_groups * 8),
            nn.LeakyReLU(0.02, inplace=True),
            nn.ConvTranspose2d(self.feature_groups * 8, self.feature_groups * 4, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_groups * 4),
            nn.LeakyReLU(0.02, inplace=True),
            nn.ConvTranspose2d(self.feature_groups * 4, self.feature_groups * 2, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_groups * 2),
            nn.LeakyReLU(0.02, inplace=True),
            nn.ConvTranspose2d(self.feature_groups * 2, self.feature_groups, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_groups),
            nn.LeakyReLU(0.02, inplace=True),
            nn.ConvTranspose2d(self.feature_groups, self.channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh() # forces values between the range -1 and 1
            )
    
    def forward(self, tensor0, tensor2):
        inTensor = torch.cat((tensor0, tensor2), 1)  # combines 2 tensor frames along the channel dimension
        outTensor = self.up_block(self.down_blocks(inTensor)) # applies the generators layers to the tensor
        return outTensor

class Generator2DVimeo(nn.Module):
    def __init__(self, z_dim, colour_channels, features_g):
        super(Generator2DVimeo, self).__init__()
        # Input of dimensions: N x z_dim x 1 x 1
        self.gen = nn.Sequential(
            self._block(z_dim, features_g*128, 4, 1, 0), # Dimensions: N x features_g*128 x 4 x 4
            self._block(features_g*128, features_g*64, 4, 2, 1), # Dimensions: N x features_g*64 x 8 x 8
            self._block(features_g*64, features_g*32, 4, 2, 1), # Dimensions: N x features_g*32 x 16 x 16
            self._block(features_g*32, features_g*16, 4, 2, 1), # Dimensions: N x features_g*16 x 32 x 32
            self._block(features_g*16, features_g*8, 4, 2, 1), # Dimensions: N x features_g*8 x 64 x 64
            self._block(features_g*8, features_g*4, 4, 2, 1), # Dimensions: N x features_g*4 x 128 x 128
            self._block(features_g*4, features_g*2, 4, 2, 1), # Dimensions: N x features_g*2 x 256 x 256
            nn.ConvTranspose2d(
                features_g*2, colour_channels, kernel_size=4, stride=2, padding=1
            ), # Dimensions: N x colour_channels x 512 x 512
            nn.Tanh()
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def _iblock(self, in_channels, out_channels, kernel_size, stride, padding):
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
        return self.gen(x)
    
class Generator2DMSU(nn.Module):
    def __init__(self, z_dim, colour_channels, features_g):
        super(Generator2DMSU, self).__init__()
        # Input of dimensions: N x z_dim x 1 x 1
        self.gen = nn.Sequential(
            self._block(z_dim, features_g*512, 4, 1, 0), # Dimensions: N x features_g*512 x 4 x 4
            self._block(features_g*512, features_g*256, 4, 2, 1), # Dimensions: N x features_g*256 x 8 x 8
            self._block(features_g*256, features_g*128, 4, 2, 1), # Dimensions: N x features_g*128 x 16 x 16
            self._block(features_g*128, features_g*64, 4, 2, 1), # Dimensions: N x features_g*64 x 32 x 32
            self._block(features_g*64, features_g*32, 4, 2, 1), # Dimensions: N x features_g*32 x 64 x 64
            self._block(features_g*32, features_g*16, 4, 2, 1), # Dimensions: N x features_g*16 x 128 x 128
            self._block(features_g*16, features_g*8, 4, 2, 1), # Dimensions: N x features_g*8 x 256 x 256
            self._block(features_g*8, features_g*4, 4, 2, 1), # Dimensions: N x features_g*4 x 512 x 512
            self._block(features_g*4, features_g*2, 4, 2, 1), # Dimensions: N x features_g*2 x 1024 x 1024
            nn.ConvTranspose2d(
                features_g*2, colour_channels, kernel_size=4, stride=2, padding=1
            ), # Dimensions: N x colour_channels x 2048 x 2048
            nn.Tanh()
        )
        self.extract = nn.Sequential(
            nn.Conv2d(
                colour_channels, features_g, kernel_size=4, stride=(3,2), padding=1
            ), #4
            nn.LeakyReLU(0.2),
            self._iblock(features_g, features_g*2, 4, 2, 1), #8
            self._iblock(features_g*2, features_g*4, 4, 2, 1), #16
            self._iblock(features_g*4, features_g*8, 4, 2, 1), #32
            self._iblock(features_g*8, features_g*16, 4, 2, 1), #64
            self._iblock(features_g*16, features_g*32, 4, 2, 1), #128
            self._iblock(features_g*32, features_g*64, 4, 2, 1), #256
            self._iblock(features_g*64, features_g*128, 4, 2, 1), #512
            self._iblock(features_g*128, features_g*256, 4, 2, 1), #1024

            nn.Conv2d(features_g*256, 100, kernel_size=4, stride=2, padding=0), #2048
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def _iblock(self, in_channels, out_channels, kernel_size, stride, padding):
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
        extract = self.extract(x)
        return self.gen(extract)
