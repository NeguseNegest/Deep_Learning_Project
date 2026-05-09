import torch
import torch.nn as nn

import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """
    Time embedding as in "Attention Is All You Need"
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        q = self.q(h).reshape(B, C, H * W).permute(0, 2, 1)
        k = self.k(h).reshape(B, C, H * W)
        v = self.v(h).reshape(B, C, H * W)

        attn = torch.softmax(q @ k / math.sqrt(C), dim=-1)
        out = (v @ attn.permute(0, 2, 1)).reshape(B, C, H, W)

        return x + self.proj(out)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()

        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)
        self.act = nn.SiLU()

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(self.act(self.norm1(x)))

        scale, shift = self.emb_proj(emb).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.conv2(self.act(h))

        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, img_ch=3, base_ch=128):
        super().__init__()

        emb_dim = base_ch * 4

        self.time_emb = SinusoidalTimeEmbedding(emb_dim)

        self.emb_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        self.conv_in = nn.Conv2d(img_ch, base_ch, 3, padding=1)

        # Encoder
        self.down1 = ResBlock(base_ch, 128, emb_dim)
        self.ds0 = Downsample(128)

        self.down2 = ResBlock(128, 256, emb_dim)
        self.ds1 = Downsample(256)

        self.down3 = ResBlock(256, 512, emb_dim)
        self.attn16 = SelfAttention(512)
        self.ds2 = Downsample(512)

        self.down4 = ResBlock(512, 512, emb_dim)
        self.attn8 = SelfAttention(512)

        # Bottleneck
        self.mid1 = ResBlock(512, 512, emb_dim)
        self.mid_attn = SelfAttention(512)
        self.mid2 = ResBlock(512, 512, emb_dim)

        # Decoder
        self.up4 = ResBlock(1024, 512, emb_dim)
        self.attn8_up = SelfAttention(512)
        self.us2 = Upsample(512)

        self.up3 = ResBlock(1024, 512, emb_dim)
        self.attn16_up = SelfAttention(512)
        self.us1 = Upsample(512)

        self.up2 = ResBlock(768, 256, emb_dim)
        self.us0 = Upsample(256)

        self.up1 = ResBlock(384, 128, emb_dim)

        self.conv_out = nn.Conv2d(128, img_ch, 3, padding=1)

    def forward(self, x, t):
        emb = self.emb_mlp(self.time_emb(t))

        x = self.conv_in(x)

        # Encoder
        d1 = self.down1(x, emb)
        d1_down = self.ds0(d1)

        d2 = self.down2(d1_down, emb)
        d2_down = self.ds1(d2)

        d3 = self.down3(d2_down, emb)
        d3 = self.attn16(d3)
        d3_down = self.ds2(d3)

        d4 = self.down4(d3_down, emb)
        d4 = self.attn8(d4)

        # Bottleneck
        mid = self.mid1(d4, emb)
        mid = self.mid_attn(mid)
        mid = self.mid2(mid, emb)

        # Decoder
        u4 = self.up4(torch.cat([mid, d4], dim=1), emb)
        u4 = self.attn8_up(u4)
        u4_up = self.us2(u4)

        u3 = self.up3(torch.cat([u4_up, d3], dim=1), emb)
        u3 = self.attn16_up(u3)
        u3_up = self.us1(u3)

        u2 = self.up2(torch.cat([u3_up, d2], dim=1), emb)
        u2_up = self.us0(u2)

        u1 = self.up1(torch.cat([u2_up, d1], dim=1), emb)

        return self.conv_out(u1)