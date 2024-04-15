import torch                                # PyTorch library - Used for tensors and neural networks in the deep learning model.
from torch.utils.data import DataLoader     # DataLoader class of torch.utils.data module - Used for loading dataset into batches.
from torchvision import transforms          # Transforms module of torchvision library - Used for transformation functions applied to images.
import torch.nn as nn                       # Neural network module of of torch library - Used for defining weights of each layer of the models.
import numpy as np                          # Numpy library - Used for creating and modifying matrices of various dimensions.
import random                               # random library - Used for setting the seed in order to have reproducable code.
import torch.optim as optim                 # optim module of torch library - Used for defining the optimizers which train the neural networks.
import time                                 # time library - Used for calculating the duration of various sections of the script.
import math                                 # math library - Used for calculating values for visualization e.g. Number of batches based on dataset and batch sizes.
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import matplotlib.animation as animation
import configparser
from dataset import VimeoDataset

config = configparser.ConfigParser()

config.read('params.ini')

generator_layers = int(config['generator']['generator_layers'])
generator_core_layer = config['generator']['generator_core_layer']
generator_norm_layer = config['generator']['generator_norm_layer']
generator_activation_layer = config['generator']['generator_activation_layer']
genLSTM = bool(config['generator']['genLSTM'])
genLSTMBias = bool(config['generator']['genLSTMBias'])
genLSTMBi = bool(config['generator']['genLSTMBi'])
genLSTMDrop = float(config['generator']['genLSTMDrop'])
genBias = bool(config['generator']['genBias'])
genNormEps = float(config['generator']['genNormEps'])
genNormAffine = bool(config['generator']['genNormAffine'])
genDropout = bool(config['generator']['genDropout'])
genActivationInplace = bool(config['generator']['genActivationInplace'])
genActivationNegativeSlope = float(config['generator']['genActivationNegativeSlope'])
genActivationAlpha = float(config['generator']['genActivationAlpha'])
genKernelSize = int(config['generator']['genKernelSize'])
genStrideSize = int(config['generator']['genStrideSize'])
genPaddingSize = int(config['generator']['genPaddingSize'])


discriminator_layers = int(config['discriminator']['discriminator_layers'])
discriminator_core_layer = config['discriminator']['discriminator_core_layer']
discriminator_norm_layer = config['discriminator']['discriminator_norm_layer']
discriminator_activation_layer = config['discriminator']['discriminator_activation_layer']
disLSTM = bool(config['discriminator']['disLSTM'])
disLSTMBias = bool(config['discriminator']['disLSTMBias'])
disLSTMBi = bool(config['discriminator']['disLSTMBi'])
disLSTMDrop = float(config['discriminator']['disLSTMDrop'])
disBias = bool(config['discriminator']['disBias'])
disNormEps = float(config['discriminator']['disNormEps'])
disNormAffine = bool(config['discriminator']['disNormAffine'])
disDropout = bool(config['discriminator']['disDropout'])
disActivationInplace = bool(config['discriminator']['disActivationInplace'])
disActivationNegativeSlope = float(config['discriminator']['disActivationNegativeSlope'])
disActivationAlpha = float(config['discriminator']['disActivationAlpha'])
disKernelSize = int(config['discriminator']['disKernelSize'])
disStrideSize = int(config['discriminator']['disStrideSize'])
disPaddingSize = int(config['discriminator']['disPaddingSize'])


seed = int(config['training']['seed'])
deterministic_algorithms = bool(config['training']['deterministic_algorithms'])
batch_size = int(config['training']['batch_size'])
latent_dim = int(config['training']['latent_dim'])
epochs = int(config['training']['epochs'])
threads = int(config['training']['threads'])
loss_function = config['training']['loss_function']
optimizer = config['training']['optimizer']
lr = float(config['training']['lr'])
beta1 = float(config['training']['beta1'])
beta2 = float(config['training']['beta2'])
save_generated_frames = bool(config['training']['save_generated_frames'])
batches_per_frame_save = int(config['training']['batches_per_frame_save'])
model_save_type = config['training']['model_save_type']
model_save_count = int(config['training']['model_save_count'])
plot_loss = bool(config['training']['plot_loss'])
plot_preds = bool(config['training']['plot_preds'])
plot_frames = bool(config['training']['plot_frames'])


#print(generator_layers, generator_core_layer, generator_norm_layer, generator_activation_layer, genLSTM, genLSTMBias)
#print(genLSTMBi, genLSTMDrop, genBias, genNormEps, genNormAffine, genDropout, genActivationInplace )
#print(genActivationNegativeSlope, genActivationAlpha, genKernelSize, genStrideSize, genPaddingSize)

#print(discriminator_layers, discriminator_core_layer, discriminator_norm_layer, discriminator_activation_layer, disLSTM, disLSTMBias)
#print(disLSTMBi, disLSTMDrop, disBias, disNormEps, disNormAffine, disDropout, disActivationInplace )
#print(disActivationNegativeSlope, disActivationAlpha, disKernelSize, disStrideSize, disPaddingSize)

#print(seed, deterministic_algorithms, batch_size, latent_dim, epochs, threads, loss_function, optimizer, lr, beta1, beta2)
#print(save_generated_frames, batches_per_frame_save, model_save_type, model_save_count, plot_loss, plot_preds, plot_frames)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Input of dimensions: N x colour_channels x 512 x 512
        self.colour_channels = 3
        self.feature_groups = 32
        layers = []

        if discriminator_core_layer == "Convolution Layer":
            layers.append(nn.Conv2d(self.colour_channels, self.feature_groups, kernel_size=disKernelSize, stride=disStrideSize, padding=disPaddingSize, bias=disBias))
        elif discriminator_core_layer == "Dense Layer":
            layers.append(nn.Linear(self.colour_channels, self.feature_groups))

        if discriminator_activation_layer == "ReLU":
            layers.append(nn.ReLU(inplace=disActivationInplace))
        elif discriminator_activation_layer == "LeakyReLU":
            layers.append(nn.LeakyReLU(negative_slope=disActivationNegativeSlope, inplace=disActivationInplace))
        elif discriminator_activation_layer == "ELU":
            layers.append(nn.ELU(alpha=disActivationAlpha, inplace=disActivationInplace))
        elif discriminator_activation_layer == "Softplus":
            layers.append(nn.Softplus())

        for i in range(discriminator_layers):
            if discriminator_core_layer == "Convolution Layer":
                layers.append(nn.Conv2d(self.feature_groups*(2**i), self.feature_groups*(2**(i+1)),kernel_size=disKernelSize, stride=disStrideSize, padding=disPaddingSize, bias=disBias))
            elif discriminator_core_layer == "Dense Layer":
                layers.append(nn.Linear(self.feature_groups*(2**i), self.feature_groups*(2**(i+1))))
            
            if discriminator_norm_layer == "Batch":
                layers.append(nn.BatchNorm2d(self.feature_groups*(2**(i+1)), eps=disNormEps, affine=disNormAffine))
            elif discriminator_norm_layer == "Instance":
                layers.append(nn.InstanceNorm2d(self.feature_groups*(2**(i+1)), eps=disNormEps, affine=disNormAffine))

            if discriminator_activation_layer == "ReLU":
                layers.append(nn.ReLU(inplace=disActivationInplace))
            elif discriminator_activation_layer == "LeakyReLU":
                layers.append(nn.LeakyReLU(negative_slope=disActivationNegativeSlope))
            elif discriminator_activation_layer == "ELU":
                layers.append(nn.ELU(alpha=disActivationAlpha))
            elif discriminator_activation_layer == "Softplus":
                layers.append(nn.Softplus())
            
        if discriminator_core_layer == "Convolution Layer":
            layers.append(nn.Conv2d(self.feature_groups*(2**discriminator_layers), 1,kernel_size=disKernelSize, stride=disStrideSize, padding=0, bias=disBias))
        elif discriminator_core_layer == "Dense Layer":
            layers.append(nn.Linear(self.feature_groups*(2**discriminator_layers), 1))

        layers.append(nn.Sigmoid())
        self.disc = nn.Sequential(*layers)

    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.feature_groups = 32
        self.channels = 3

        layers_up = []
        layers_down = []

        if generator_core_layer == "Convolutional Layer":
            layers_down.append(nn.Conv2d(self.channels*2, self.feature_groups*2, kernel_size=genKernelSize, stride=genStrideSize, padding=genPaddingSize, bias=genBias))
        elif generator_core_layer == "Dense Layer":
            layers_down.append(nn.Linear(self.channels*2, self.feature_groups*2))

        if discriminator_norm_layer == "Batch":
            layers_down.append(nn.BatchNorm2d(self.feature_groups*2, eps=disNormEps, affine=disNormAffine))
        elif discriminator_norm_layer == "Instance":
            layers_down.append(nn.InstanceNorm2d(self.feature_groups*2, eps=disNormEps, affine=disNormAffine))
            
        if generator_activation_layer == "ReLU":
            layers_down.append(nn.ReLU(inplace=genActivationInplace))
        elif discriminator_activation_layer == "LeakyReLU":
            layers_down.append(nn.LeakyReLU(negative_slope=genActivationNegativeSlope, inplace=genActivationInplace))
        elif discriminator_activation_layer == "ELU":
            layers_down.append(nn.ELU(alpha=genActivationAlpha, inplace=genActivationInplace))
        elif discriminator_activation_layer == "Softplus":
            layers_down.append(nn.Softplus())

        for i in range(generator_layers):
            j = 10-1-i
            if generator_core_layer == "Convolution Layer":
                layers_down.append(nn.Conv2d(self.feature_groups*(2**(i)), self.feature_groups*(2**(i+1)), kernel_size=genKernelSize, stride=genStrideSize, padding=genPaddingSize, bias=genBias))
                layers_up.append(nn.ConvTranspose2d(self.feature_groups*(2**(j)), self.feature_groups*(2**(j-1)), kernel_size=genKernelSize, stride=genStrideSize, padding=genPaddingSize, bias=genBias))
            elif generator_core_layer == "Dense Layer":
                layers_down.append(nn.Linear(self.feature_groups*(2**(i)), self.feature_groups*(2**(i+1))))
                layers_up.append(nn.Linear(self.feature_groups*(2**(j)), self.feature_groups*(2**(j-1))))

            if discriminator_norm_layer == "Batch":
                layers_down.append(nn.BatchNorm2d(self.feature_groups*(2**(i+1)), eps=disNormEps, affine=disNormAffine))
                layers_up.append(nn.BatchNorm2d(self.feature_groups*(2**(j-1)), eps=disNormEps, affine=disNormAffine))
            elif discriminator_norm_layer == "Instance":
                layers_down.append(nn.InstanceNorm2d(self.feature_groups*(2**(i+1)), eps=disNormEps, affine=disNormAffine))
                layers_up.append(nn.InstanceNorm2d(self.feature_groups*(2**(j-1)), eps=disNormEps, affine=disNormAffine))

            if generator_activation_layer == "ReLU":
                layers_down.append(nn.ReLU(inplace=genActivationInplace))
                layers_up.append(nn.ReLU(inplace=genActivationInplace))
            elif discriminator_activation_layer == "LeakyReLU":
                layers_down.append(nn.LeakyReLU(negative_slope=genActivationNegativeSlope, inplace=genActivationInplace))
                layers_up.append(nn.LeakyReLU(negative_slope=genActivationNegativeSlope, inplace=genActivationInplace))
            elif discriminator_activation_layer == "ELU":
                layers_down.append(nn.ELU(alpha=genActivationAlpha, inplace=genActivationInplace))
                layers_up.append(nn.ELU(alpha=genActivationAlpha, inplace=genActivationInplace))
            elif discriminator_activation_layer == "Softplus":
                layers_down.append(nn.Softplus())        
                layers_up.append(nn.Softplus())  

        if generator_core_layer == "Convolutional Layer":
            layers_up.append(nn.Conv2d(self.feature_groups, self.channels, kernel_size=genKernelSize, stride=genStrideSize, padding=genPaddingSize, bias=genBias))
        elif generator_core_layer == "Dense Layer":
            layers_up.append(nn.Linear(self.feature_groups, self.channels))

        layers_up.append(nn.Tanh())  

        self.down_sample_blocks = nn.Sequential(*layers_down)
        self.up_sample_blocks = nn.Sequential(*layers_up)

    def forward(self, tensor0, tensor2):

        out = torch.cat((tensor0, tensor2), 1)
        out_down = self.down_sample_blocks(out)
        out_up = self.up_sample_blocks(out_down)

        return out_up
    
def start():
    training_batches = int()
    training_remainder = int()  
    # Attempts to use a GPU if available, otherwise uses the users CPU.
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"Error during GPU initialization: {e}")
        device = torch.device("cpu")
    print(f"Selected device: {device} ({torch.cuda.get_device_name(device.index)}), number of GPU's: {torch.cuda.device_count()}")

    train_dataset = VimeoDataset(root_dir='src/vimeo_septuplet', split='train')
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=threads)
    N, in_channels, height, width = 7, 3, 512, 512

    training_batches = math.ceil(len(train_dataset) / batch_size)
    training_remainder = len(train_dataset) % batch_size

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    if loss_function == "BCE":
        criteron = nn.BCELoss
    elif loss_function == "BCEWithLogitsLoss":
        criteron = nn.BCEWithLogitsLoss
    elif loss_function == "MSE":
        criteron = nn.MSELoss

    if optimizer == "Adam":
        optimizer_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=lr/epochs)
        optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=lr/epochs)
    elif optimizer == "RMSProp":
        optimizer_generator = optim.RMSprop(generator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=lr/epochs)
        optimizer_discriminator = optim.RMSprop(discriminator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=lr/epochs)
    elif optimizer == "RMSProp":
        optimizer_generator = optim.SGD(generator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=lr/epochs)
        optimizer_discriminator = optim.SGD(discriminator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=lr/epochs)
    elif optimizer == "RMSProp":
        optimizer_generator = optim.Adagrad(generator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=lr/epochs)
        optimizer_discriminator = optim.Adagrad(discriminator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=lr/epochs)
    elif optimizer == "RMSProp":
        optimizer_generator = optim.AdamW(generator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=lr/epochs)
        optimizer_discriminator = optim.AdamW(discriminator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=lr/epochs)

    return train_data, training_batches, training_remainder, device, generator, discriminator, criteron, optimizer_generator, optimizer_discriminator