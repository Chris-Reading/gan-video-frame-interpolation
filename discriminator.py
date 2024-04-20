import torch.nn as nn

class VimeoDiscriminator2(nn.Module):

    def __init__(self):
        super(VimeoDiscriminator2, self).__init__()
        self.feature_groups = 32  # the size of feature map
        self.channels = 3 # target channels (rgb)
        
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(self.channels, self.feature_groups, kernel_size=4, stride=2, padding=1, bias=False), # increases number of channels from 3 to 32
            nn.LeakyReLU(0.2, inplace=True), # allows small negative inputs through
            nn.Dropout2d(0.15), # randomly drops out a proportion of inputs to prevent overfitting
            nn.Conv2d(self.feature_groups, self.feature_groups * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_groups * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(self.feature_groups * 2, self.feature_groups * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_groups * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(self.feature_groups * 4, self.feature_groups * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_groups * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(self.feature_groups * 8, self.feature_groups * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_groups * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(self.feature_groups * 16, self.feature_groups * 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_groups * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(self.feature_groups * 32, self.feature_groups * 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_groups * 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(self.feature_groups * 64, 1, kernel_size=4, stride=1, padding=0, bias=False) # decreases channels to 1 (for a singular probability)
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
            nn.Sigmoid() # changes output to a singular classifier between 0 and 1 (a probability in this case)
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
            nn.BatchNorm2d(out_channels), # normalizes activations using stdev and mean
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
