import torch.nn as nn

class Discriminator2D(nn.Module):
    def __init__(self, colour_channels, features_d):
        super(Discriminator2D, self).__init__()
        # Input of dimensions: N x colour_channels x 448 x 256
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