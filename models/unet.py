import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def choose_num_groups(num_channels: int, max_groups: int = 32) -> int:
    """
    GroupNorm kräver att num_channels är delbart med num_groups.
    Denna funktion väljer ett giltigt antal grupper.
    """
    num_groups = min(max_groups, num_channels)

    while num_channels % num_groups != 0:
        num_groups -= 1

    return num_groups


class SinusoidalPositionalEncoding(nn.Module):
    """
    Timestep embedding enligt exakt formel:

        e(t)_{2i}   = sin(t / 10000^{2i / d})
        e(t)_{2i+1} = cos(t / 10000^{2i / d})

    Input:
        timestep: [B] eller [B, 1]

    Output:
        embeddings: [B, embedd_dim]
    """

    def __init__(self, embedd_dim: int):
        super().__init__()

        if embedd_dim % 2 != 0:
            raise ValueError("embedd_dim måste vara jämnt, till exempel 64 eller 128.")

        self.embedd_dim = embedd_dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.dim() != 1:
            timestep = timestep.view(-1)

        device = timestep.device
        batch_size = timestep.shape[0]

        d = self.embedd_dim
        half_dim = d // 2

        i = torch.arange(
            half_dim,
            device=device,
            dtype=torch.float32,
        )

        denominator = 10000.0 ** (2.0 * i / d)

        taljare = timestep.float()[:, None] / denominator[None, :]

        embeddings = torch.zeros(
            batch_size,
            d,
            device=device,
            dtype=torch.float32,
        )

        embeddings[:, 0::2] = torch.sin(taljare)
        embeddings[:, 1::2] = torch.cos(taljare)

        return embeddings


class ResBlock(nn.Module):
    """
    Residual block med timestep-information.

    Input:
        x:     [B, in_channels, H, W]
        t_emb: [B, time_dim]

    Output:
        [B, out_channels, H, W]
    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()

        self.norm1 = nn.GroupNorm(
            num_groups=choose_num_groups(in_channels),
            num_channels=in_channels,
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.time_proj = nn.Linear(time_dim, out_channels)

        self.norm2 = nn.GroupNorm(
            num_groups=choose_num_groups(out_channels),
            num_channels=out_channels,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        time_addition = self.time_proj(t_emb)
        h = h + time_addition[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


class Downsample(nn.Module):
    """
    32x32 -> 16x16
    16x16 -> 8x8
    8x8   -> 4x4
    """

    def __init__(self, channels: int):
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    4x4   -> 8x8
    8x8   -> 16x16
    16x16 -> 32x32
    """

    def __init__(self, channels: int):
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="nearest",
        )

        return self.conv(x)


class AttentionBlock(nn.Module):
    """
    Self-attention i bottleneck.

    För CIFAR-10 32x32 hamnar bottleneck på 4x4.
    Då är attention ganska billig men kan hjälpa modellen förstå global struktur.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError("channels måste vara delbart med num_heads.")

        self.norm = nn.GroupNorm(
            num_groups=choose_num_groups(channels),
            num_channels=channels,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        residual = x

        x = self.norm(x)

        x = x.view(B, C, H * W)
        x = x.transpose(1, 2)

        x, _ = self.attention(
            x,
            x,
            x,
            need_weights=False,
        )

        x = x.transpose(1, 2)
        x = x.view(B, C, H, W)

        return x + residual


class CNN(nn.Module):
    """
    Detta är egentligen en liten U-Net, men klassen heter CNN
    för att passa din kompis kodstruktur.

    Input:
        x: noisy image x_t
           shape [B, 3, 32, 32]

        t: timestep
           shape [B]

    Output:
        predicted noise epsilon_theta(x_t, t)
        shape [B, 3, 32, 32]
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden: int = 64,
        time_dim: int = 128,
        use_attention: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden = hidden
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEncoding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(
            in_channels,
            hidden,
            kernel_size=3,
            padding=1,
        )

        # Encoder:
        # 32x32 -> 16x16 -> 8x8 -> 4x4

        self.down_block1 = ResBlock(
            in_channels=hidden,
            out_channels=hidden,
            time_dim=time_dim,
        )
        self.downsample1 = Downsample(hidden)

        self.down_block2 = ResBlock(
            in_channels=hidden,
            out_channels=hidden * 2,
            time_dim=time_dim,
        )
        self.downsample2 = Downsample(hidden * 2)

        self.down_block3 = ResBlock(
            in_channels=hidden * 2,
            out_channels=hidden * 4,
            time_dim=time_dim,
        )
        self.downsample3 = Downsample(hidden * 4)

        # Bottleneck: 4x4

        self.mid_block1 = ResBlock(
            in_channels=hidden * 4,
            out_channels=hidden * 4,
            time_dim=time_dim,
        )

        if use_attention:
            self.mid_attention = AttentionBlock(
                channels=hidden * 4,
                num_heads=4,
            )
        else:
            self.mid_attention = nn.Identity()

        self.mid_block2 = ResBlock(
            in_channels=hidden * 4,
            out_channels=hidden * 4,
            time_dim=time_dim,
        )

        # Decoder:
        # 4x4 -> 8x8 -> 16x16 -> 32x32

        self.upsample3 = Upsample(hidden * 4)
        self.up_block3 = ResBlock(
            in_channels=hidden * 4 + hidden * 4,
            out_channels=hidden * 2,
            time_dim=time_dim,
        )

        self.upsample2 = Upsample(hidden * 2)
        self.up_block2 = ResBlock(
            in_channels=hidden * 2 + hidden * 2,
            out_channels=hidden,
            time_dim=time_dim,
        )

        self.upsample1 = Upsample(hidden)
        self.up_block1 = ResBlock(
            in_channels=hidden + hidden,
            out_channels=hidden,
            time_dim=time_dim,
        )

        self.final_norm = nn.GroupNorm(
            num_groups=choose_num_groups(hidden),
            num_channels=hidden,
        )

        self.final_conv = nn.Conv2d(
            hidden,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        if t.dim() != 1:
            t = t.view(-1)

        t_emb = self.time_mlp(t)

        # Initial projection
        x = self.init_conv(x)

        # Encoder
        h1 = self.down_block1(x, t_emb)
        x = self.downsample1(h1)

        h2 = self.down_block2(x, t_emb)
        x = self.downsample2(h2)

        h3 = self.down_block3(x, t_emb)
        x = self.downsample3(h3)

        # Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_attention(x)
        x = self.mid_block2(x, t_emb)

        # Decoder
        x = self.upsample3(x)

        if x.shape[-2:] != h3.shape[-2:]:
            x = F.interpolate(x, size=h3.shape[-2:], mode="nearest")

        x = torch.cat([x, h3], dim=1)
        x = self.up_block3(x, t_emb)

        x = self.upsample2(x)

        if x.shape[-2:] != h2.shape[-2:]:
            x = F.interpolate(x, size=h2.shape[-2:], mode="nearest")

        x = torch.cat([x, h2], dim=1)
        x = self.up_block2(x, t_emb)

        x = self.upsample1(x)

        if x.shape[-2:] != h1.shape[-2:]:
            x = F.interpolate(x, size=h1.shape[-2:], mode="nearest")

        x = torch.cat([x, h1], dim=1)
        x = self.up_block1(x, t_emb)

        # Final prediction of noise epsilon
        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)

        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CNN(
        in_channels=3,
        out_channels=3,
        hidden=64,
        time_dim=128,
        use_attention=True,
    ).to(device)

    x = torch.randn(4, 3, 32, 32).to(device)
    t = torch.randint(0, 1000, (4,), device=device)

    y = model(x, t)

    print("Input shape: ", x.shape)
    print("Output shape:", y.shape)
    print("Parameter count:", count_parameters(model))

    assert y.shape == x.shape

    print("CNN/U-Net test passed.")