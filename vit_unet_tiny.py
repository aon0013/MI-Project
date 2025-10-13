import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --Helpers--

class OverlapPatchEmbed(nn.Module):
    """Overlapped patch embedding (Conv stem)"""
    def __init__(self, in_ch, embed_dim, patch_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x = self.norm(x_flat)
        return x, (H, W)


class PositionalEmbedding(nn.Module):
    """Sine-cosine positional embedding (no params, robust to size)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, B, H, W, device):
        ys = torch.arange(H, device=device).float()
        xs = torch.arange(W, device=device).float()
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=device).float() * -(math.log(10000.0) / self.dim))
        pe = torch.zeros(H, W, self.dim, device=device)
        pe[:, :, 0::2] = torch.sin(grid_y[..., None] * div_term)
        pe[:, :, 1::2] = torch.cos(grid_y[..., None] * div_term)
        return pe.view(1, H * W, self.dim).repeat(B, 1, 1)


class TransformerBlock(nn.Module):
    """Vanilla MHSA + FFN with LayerNorm."""
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MixFFNFuse(nn.Module):
    """Simple fusion head (MLP) to mix multi scale features """
    def __init__(self, in_dims, out_dim):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, out_dim, 1, bias=False) for c in in_dims])

    def forward(self, feats):
        target_h, target_w = feats[0].shape[-2:]
        outs = []
        for i, f in enumerate(feats):
            if f.shape[-2:] != (target_h, target_w):
                f = F.interpolate(f, size=(target_h, target_w), mode="bilinear", align_corners=False)
            outs.append(self.proj[i](f))
        return torch.stack(outs, dim=0).sum(0)


# - Model -

class ViTUNetTiny(nn.Module):
    """Tiny hybrid: overlapped patch embeds (4-stage), tiny Transformer per stage, FPN like fusion."""
    def __init__(self, in_ch: int, K: int, kernels: int = 8, factor: int = 2):
        super().__init__()
        dims = [32, 64, 128, 256]
        heads = [1, 2, 4, 8]
        depths = [1, 1, 2, 2]

        self.pe1 = OverlapPatchEmbed(in_ch, dims[0], patch_size=7, stride=4, padding=3)
        self.pe2 = OverlapPatchEmbed(dims[0], dims[1], patch_size=3, stride=4, padding=1)
        self.pe3 = OverlapPatchEmbed(dims[1], dims[2], patch_size=3, stride=2, padding=1)
        self.pe4 = OverlapPatchEmbed(dims[2], dims[3], patch_size=3, stride=2, padding=1)

        self.pos1 = PositionalEmbedding(dims[0])
        self.pos2 = PositionalEmbedding(dims[1])
        self.pos3 = PositionalEmbedding(dims[2])
        self.pos4 = PositionalEmbedding(dims[3])

        def make_stage(dim, h, d):
            return nn.Sequential(*[TransformerBlock(dim, num_heads=h, mlp_ratio=3.0) for _ in range(d)])

        self.enc1 = make_stage(dims[0], heads[0], depths[0])
        self.enc2 = make_stage(dims[1], heads[1], depths[1])
        self.enc3 = make_stage(dims[2], heads[2], depths[2])
        self.enc4 = make_stage(dims[3], heads[3], depths[3])

        self.fuse = MixFFNFuse(in_dims=dims, out_dim=128)
        self.refine = nn.Sequential(
            nn.Conv2d(128, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.head = nn.Conv2d(64, K, kernel_size=1, bias=True)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def tokens_to_map(x, hw):
        B, N, C = x.shape
        H, W = hw
        return x.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, x):
        B, _, H, W = x.shape
        s1, hw1 = self.pe1(x)
        s1 = s1 + self.pos1(B, *hw1, x.device)
        s1 = self.enc1(s1)
        f1 = self.tokens_to_map(s1, hw1)

        s2, hw2 = self.pe2(f1)
        s2 = s2 + self.pos2(B, *hw2, x.device)
        s2 = self.enc2(s2)
        f2 = self.tokens_to_map(s2, hw2)

        s3, hw3 = self.pe3(f2)
        s3 = s3 + self.pos3(B, *hw3, x.device)
        s3 = self.enc3(s3)
        f3 = self.tokens_to_map(s3, hw3)

        s4, hw4 = self.pe4(f3)
        s4 = s4 + self.pos4(B, *hw4, x.device)
        s4 = self.enc4(s4)
        f4 = self.tokens_to_map(s4, hw4)

        fused = self.fuse([f1, f2, f3, f4])
        out = self.refine(fused)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        logits = self.head(out)
        return logits
