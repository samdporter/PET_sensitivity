import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.res(x)

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, num_res_blocks=9):
        super(Generator, self).__init__()
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(num_res_blocks)])
        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(64, output_channels, 7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.output(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

def min_max_scaling(tensor, min_val, max_val):
    """Scale the input tensor to [-1, 1]."""
    return (tensor - min_val) / (max_val - min_val) * 2 - 1

def get_generator_loss(D_X, D_Y, real_X, fake_X, real_Y, fake_Y, criterion, g_criterion):
    """Calculate the generator loss."""
    d_fake_X = D_X(fake_X)
    g_loss_X = criterion(d_fake_X, torch.ones_like(d_fake_X))

    d_fake_Y = D_Y(fake_Y)
    g_loss_Y = criterion(d_fake_Y, torch.ones_like(d_fake_Y))

    cycle_consistency_loss_X = g_criterion(fake_X, real_X)
    cycle_consistency_loss_Y = g_criterion(fake_Y, real_Y)

    g_loss = g_loss_X + g_loss_Y + cycle_consistency_loss_X + cycle_consistency_loss_Y
    return g_loss

def get_discriminator_loss(D_X, D_Y, real_X, fake_X, real_Y, fake_Y, criterion):
    """Calculate the discriminator loss."""
    d_real_X = D_X(real_X)
    d_real_Y = D_Y(real_Y)

    d_fake_X = D_X(fake_X.detach())
    d_fake_Y = D_Y(fake_Y.detach())

    d_loss_real_X = criterion(d_real_X, torch.ones_like(d_real_X))
    d_loss_real_Y = criterion(d_real_Y, torch.ones_like(d_real_Y))

    d_loss_fake_X = criterion(d_fake_X, torch.zeros_like(d_fake_X))
    d_loss_fake_Y = criterion(d_fake_Y, torch.zeros_like(d_fake_Y))

    d_loss_X = (d_loss_real_X + d_loss_fake_X) * 0.5
    d_loss_Y = (d_loss_real_Y + d_loss_fake_Y) * 0.5

    d_loss = d_loss_X + d_loss_Y
    return d_loss