import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicSnakeDeformableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, axis='x'):
        super().__init__()
        self.axis = axis
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_half = kernel_size // 2

        self.offset_conv = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        B, C, H, W = x.shape
        K = self.kernel_size

        offsets = self.offset_conv(x)  # (B, 2, H, W)

        if self.axis == 'x':
            offsets[:, 1, :, :] = 0.0
        else:
            offsets[:, 0, :, :] = 0.0

        padded = F.pad(x, [self.k_half] * 4, mode='replicate')

        result = torch.zeros(B, self.out_channels, H, W, device=x.device, dtype=x.dtype)

        for i in range(K):
            for j in range(K):
                dy = (i - self.k_half) + offsets[:, 1]  # (B, H, W)
                dx = (j - self.k_half) + offsets[:, 0]

                grid_y = torch.arange(H, device=x.device, dtype=x.dtype).view(1, H, 1) + dy
                grid_x = torch.arange(W, device=x.device, dtype=x.dtype).view(1, 1, W) + dx

                grid_y = grid_y.unsqueeze(-1)
                grid_x = grid_x.unsqueeze(-1)

                grid = torch.cat([grid_x, grid_y], dim=-1)  # (B, H, W, 2)

                grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
                grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
                grid = grid.clamp(-1, 1)

                shifted = F.grid_sample(padded, grid, mode='bilinear',
                                        padding_mode='border', align_corners=True)

                w_k = self.weight[:, :, i, j].view(self.out_channels, self.in_channels)
                result += F.conv2d(shifted, w_k.unsqueeze(-1).unsqueeze(-1), bias=None)

        result = result + self.bias.view(1, -1, 1, 1)
        return result


class DSCG(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2

        self.branch_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            nn.GELU(),
        )
        self.branch_deform_x = DynamicSnakeDeformableConv(in_channels, mid, kernel_size=3, axis='x')
        self.gelu_x = nn.GELU()
        self.branch_deform_y = DynamicSnakeDeformableConv(in_channels, mid, kernel_size=3, axis='y')
        self.gelu_y = nn.GELU()
        self.fusion = nn.Sequential(
            nn.Conv2d(mid * 3, out_channels, 1, bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        b1 = self.branch_conv(x)
        b2 = self.gelu_x(self.branch_deform_x(x))
        b3 = self.gelu_y(self.branch_deform_y(x))
        return self.fusion(torch.cat([b1, b2, b3], dim=1))
