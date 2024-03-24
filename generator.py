import torch.nn as nn

class Generator2D(nn.Module):
    def __init__(self, z_dim, colour_channels, features_g):
        super(Generator2D, self).__init__()
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
    
    def forward(self, x):
        return self.gen(x)