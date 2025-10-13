import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Utility modules

class DropPath(nn.Module):
    """Stochastic depth per sample."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = x.new_empty(shape).bernoulli_(keep)
        return x * rand / keep


class LayerScale(nn.Module):
    """Per channel learnable rescaling for residual branches"""
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))
    def forward(self, x):
        return x * self.gamma



# Embedding and positional encoding

class OverlapPatchEmbed(nn.Module):
    """Overlapped patch embedding (Conv stem) > tokens"""
    def __init__(self, in_ch, embed_dim, patch_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x = self.norm(x_flat)
        return x, (H, W)


class PositionalEmbedding(nn.Module):
    """Sine-cosine 2D positional embedding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, B, H, W, device):
        ys = torch.arange(H, device=device).float()
        xs = torch.arange(W, device=device).float()
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        div = torch.exp(torch.arange(0, self.dim, 2, device=device).float() *
                        -(math.log(10000.0) / self.dim))
        pe = torch.zeros(H, W, self.dim, device=device)
        pe[..., 0::2] = torch.sin(gy[..., None] * div)
        pe[..., 1::2] = torch.cos(gy[..., None] * div)
        return pe.view(1, H * W, self.dim).repeat(B, 1, 1)



# Transformer block and MLP
class MLPWithDWConv(nn.Module):
    """Feed forward MLP with depthwise conv for local token mixing."""
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x, hw):
        B, N, C = x.shape
        H, W = hw
        h = self.fc1(x).transpose(1, 2).reshape(B, -1, H, W)
        h = self.dwconv(h).flatten(2).transpose(1, 2)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return self.drop(h)


class TransformerBlock(nn.Module):
    """Pre norm MHSA + (DWConv)MLP + LayerScale + DropPath."""
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0,
                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.ls1 = LayerScale(dim)
        self.drop1 = DropPath(drop_path)
        hidden = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPWithDWConv(dim, hidden, drop=drop)
        self.ls2 = LayerScale(dim)
        self.drop2 = DropPath(drop_path)
    def forward(self, x, hw):
        x = x + self.drop1(self.ls1(self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                                              need_weights=False)[0]))
        x = x + self.drop2(self.ls2(self.mlp(self.norm2(x), hw)))
        return x



# Fusion and Squeeze-Excitation

class SE(nn.Module):
    """Channel attention via squeeze excitation"""
    def __init__(self, c, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c // r, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(c // r, c, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class MixFFNFuse(nn.Module):
    """Project + upsample multi scale features, fuse with SE block."""
    def __init__(self, in_dims, out_dim, mid=192):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, mid, 1, bias=False) for c in in_dims])
        self.fuse = nn.Sequential(
            nn.Conv2d(mid * len(in_dims), mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            SE(mid),
            nn.Conv2d(mid, out_dim, 1, bias=False)
        )
    def forward(self, feats):
        target_h, target_w = feats[0].shape[-2:]
        ups = []
        for i, f in enumerate(feats):
            if f.shape[-2:] != (target_h, target_w):
                f = F.interpolate(f, size=(target_h, target_w), mode="bilinear", align_corners=False)
            ups.append(self.proj[i](f))
        x = torch.cat(ups, dim=1)
        return self.fuse(x)



# Refined Model

class ViTUNetTiny(nn.Module):
    """
    v2: Tiny hybrid Transformer UNet.
    Adds DropPath + LayerScale, depthwise MLP, SE fusion, separable refine convs.
    API stays compatible: ViTUNetTiny(in_ch, K, kernels=8, factor=2)
    """
    def __init__(self, in_ch, K, kernels=8, factor=2):
        super().__init__()
        dims = [32, 64, 128, 256]
        heads = [1, 2, 4, 8]
        depths = [1, 1, 2, 2]
        dpr = torch.linspace(0, 0.07, sum(depths)).tolist()
        idx = 0

        # Patch embeddings
        self.pe1 = OverlapPatchEmbed(in_ch, dims[0], 7, 4, 3)
        self.pe2 = OverlapPatchEmbed(dims[0], dims[1], 3, 4, 1)
        self.pe3 = OverlapPatchEmbed(dims[1], dims[2], 3, 2, 1)
        self.pe4 = OverlapPatchEmbed(dims[2], dims[3], 3, 2, 1)

        # Positional embeddings
        self.pos1 = PositionalEmbedding(dims[0])
        self.pos2 = PositionalEmbedding(dims[1])
        self.pos3 = PositionalEmbedding(dims[2])
        self.pos4 = PositionalEmbedding(dims[3])

        # Transformer encoder stages
        def make_stage(dim, h, d):
            nonlocal idx
            blocks = []
            for _ in range(d):
                blocks.append(TransformerBlock(dim, num_heads=h, mlp_ratio=3.0,
                                               drop_path=dpr[idx]))
                idx += 1
            return nn.ModuleList(blocks)

        self.enc1 = make_stage(dims[0], heads[0], depths[0])
        self.enc2 = make_stage(dims[1], heads[1], depths[1])
        self.enc3 = make_stage(dims[2], heads[2], depths[2])
        self.enc4 = make_stage(dims[3], heads[3], depths[3])

        # Fusion and head
        self.fuse = MixFFNFuse(dims, 128, mid=192)
        self.refine = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False),
            nn.Conv2d(128, 96, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 96, 3, padding=1, groups=96, bias=False),
            nn.Conv2d(96, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.head = nn.Conv2d(64, K, 1, bias=True)

        self.apply(self._init)

   
    # Helpers

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

    def _encode_stage(self, tokens, pos, blocks, hw):
        x = tokens + pos
        for blk in blocks:
            x = blk(x, hw)
        return x

 
    # Forward

    def forward(self, x):
        B, _, H, W = x.shape

        s1, hw1 = self.pe1(x)
        s1 = self._encode_stage(s1, self.pos1(B, *hw1, x.device), self.enc1, hw1)
        f1 = self.tokens_to_map(s1, hw1)

        s2, hw2 = self.pe2(f1)
        s2 = self._encode_stage(s2, self.pos2(B, *hw2, x.device), self.enc2, hw2)
        f2 = self.tokens_to_map(s2, hw2)

        s3, hw3 = self.pe3(f2)
        s3 = self._encode_stage(s3, self.pos3(B, *hw3, x.device), self.enc3, hw3)
        f3 = self.tokens_to_map(s3, hw3)

        s4, hw4 = self.pe4(f3)
        s4 = self._encode_stage(s4, self.pos4(B, *hw4, x.device), self.enc4, hw4)
        f4 = self.tokens_to_map(s4, hw4)

        fused = self.fuse([f1, f2, f3, f4])
        out = self.refine(fused)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        logits = self.head(out)
        return logits
