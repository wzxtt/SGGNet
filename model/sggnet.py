import torch
import torch.nn as nn
import torch.nn.functional as F

from .dscg import DSCG
from .grfb import GRFB
from .swin_snake_dual import SwinSnakeDual


class ConvBnReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SGGNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_channels=52):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.encoder1 = SwinSnakeDual(base_channels, base_channels * 2)
        self.grfb1 = GRFB(base_channels * 2, base_channels * 2)

        self.encoder2 = SwinSnakeDual(base_channels * 2, base_channels * 4)
        self.grfb2 = GRFB(base_channels * 4, base_channels * 4)

        self.encoder3 = SwinSnakeDual(base_channels * 4, base_channels * 8)
        self.grfb3 = GRFB(base_channels * 8, base_channels * 8)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder3_conv = DSCG(base_channels * 8 + base_channels * 4, base_channels * 4)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder2_conv = DSCG(base_channels * 4 + base_channels * 2, base_channels * 2)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder1_conv = DSCG(base_channels * 2 + base_channels, base_channels)

        self.final_conv = nn.Conv2d(base_channels, num_classes, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape

        initial_feat = self.initial(x)

        skip1 = self.grfb1(self.encoder1(initial_feat))
        skip2 = self.grfb2(self.encoder2(skip1))
        bottleneck = self.grfb3(self.encoder3(skip2))

        d3 = self.upsample3(bottleneck)
        d3 = torch.cat([d3, skip2], dim=1)
        d3 = self.decoder3_conv(d3)

        d2 = self.upsample2(d3)
        d2 = torch.cat([d2, skip1], dim=1)
        d2 = self.decoder2_conv(d2)

        d1 = self.upsample1(d2)
        d1 = torch.cat([d1, initial_feat], dim=1)
        d1 = self.decoder1_conv(d1)

        out = self.final_conv(d1)
        out = torch.sigmoid(out)
        return out
