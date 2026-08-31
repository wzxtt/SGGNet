import torch
import torch.nn as nn


class GRFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 4

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 5, padding=2, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.branch_d2 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.branch_d4 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(mid * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b1 = self.branch3(x)
        b2 = self.branch5(x)
        b3 = self.branch_d2(x)
        b4 = self.branch_d4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.fusion(out)
        return out
