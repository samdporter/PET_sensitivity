import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, in_channels=2, n_class=1):
        super().__init__()

        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        # Calculate padding dimensions
        h, w = x.size(2), x.size(3)
        h_pad = ((h - 1) // 16 + 1) * 16
        w_pad = ((w - 1) // 16 + 1) * 16
        pad_h = (h_pad - h, 0)
        pad_w = (w_pad - w, 0)

        # Apply padding
        x = F.pad(x, pad_w + pad_h)

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x[:, :, :conv3.size(2), :conv3.size(3)], conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x[:, :, :conv2.size(2), :conv2.size(3)], conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x[:, :, :conv1.size(2), :conv1.size(3)], conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        # Crop back to the original size
        out = out[:, :, :h, :w]
        return out
