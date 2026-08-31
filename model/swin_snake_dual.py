import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.logit_scale = nn.Parameter(
            math.log(10) * torch.ones((num_heads, 1, 1)), requires_grad=True)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)

        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01))
        attn = (q_norm @ k_norm.transpose(-2, -1)) * logit_scale.exp()

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(B_ // num_windows, num_windows, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerV2Block(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=2.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

        if self.shift_size > 0:
            H = W = window_size
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -window_size),
                        slice(-window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -window_size),
                        slice(-window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class DSCGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        from .dscg import DSCG
        self.dscg = DSCG(in_channels, out_channels)

    def forward(self, x):
        return self.dscg(x)


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class SwinSnakeDual(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, window_size=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.window_size = window_size

        self.dscg = DSCGBlock(in_channels, out_channels)
        self.pool = nn.AvgPool2d(2, stride=2)

        self.swin_block1 = SwinTransformerV2Block(
            dim=in_channels, num_heads=num_heads, window_size=window_size, shift_size=0)
        self.swin_block2 = SwinTransformerV2Block(
            dim=in_channels, num_heads=num_heads, window_size=window_size,
            shift_size=window_size // 2)
        self.norm = nn.LayerNorm(in_channels)
        self.patch_merge = PatchMerging(in_channels)

        self.fusion = nn.Conv2d(out_channels * 2, out_channels, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        branch1 = self.dscg(x)
        branch1 = self.pool(branch1)

        pad_h = (-H) % 14
        pad_w = (-W) % 14
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        else:
            x_pad = x

        H_p, W_p = H + pad_h, W + pad_w

        x_t = x_pad.permute(0, 2, 3, 1).contiguous().view(B, H_p * W_p, C)
        x_t = self.swin_block1(x_t, H_p, W_p)
        x_t = self.swin_block2(x_t, H_p, W_p)
        x_t = self.norm(x_t)
        x_t = self.patch_merge(x_t, H_p, W_p)

        branch2 = x_t.permute(0, 2, 1).contiguous().view(B, self.out_channels, H_p // 2, W_p // 2)

        if H_p != H or W_p != W:
            out_H = H // 2
            out_W = W // 2
            branch2 = branch2[:, :, :out_H, :out_W]

        out = torch.cat([branch1, branch2], dim=1)
        out = self.fusion(out)
        return out
