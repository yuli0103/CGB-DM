from torch import nn
import torch
from module.arch_transformer import Transformer


class image_adapter(nn.Module):
    def __init__(self, in_dim=768, out_dim=512, num_heads=8, num_tokens=16, n_layers=2):
        super().__init__()
        scale = in_dim ** -0.5
        self.num_tokens = num_tokens
        self.style_emb = nn.Parameter(torch.randn(1, num_tokens, in_dim) * scale)
        self.transformer_blocks = Transformer(
            width=in_dim,
            layers=n_layers,
            heads=num_heads,
        )
        self.ln1 = nn.LayerNorm(in_dim)
        self.ln2 = nn.LayerNorm(in_dim)
        self.proj = nn.Parameter(torch.randn(in_dim, out_dim) * scale)

    def forward(self, x):
        style_emb = self.style_emb.repeat(x.shape[0], 1, 1)
        x = torch.cat([style_emb, x], dim=1)
        # x = torch.cat([x, style_emb], dim=1)
        x = self.ln1(x)

        x = x.permute(1, 0, 2)
        x = self.transformer_blocks(x)
        x = x.permute(1, 0, 2)

        x = self.ln2(x[:, :self.num_tokens, :])
        x = x @ self.proj
        return x

class ScaleEncoder(nn.Module):
    def __init__(self, in_dim=512, out_dim=1, num_heads=8, num_tokens=1, n_layers=2):
        super().__init__()
        scale = in_dim ** -0.5
        self.num_tokens = num_tokens
        self.scale_emb = nn.Parameter(torch.randn(1, num_tokens, in_dim) * scale)
        self.transformer_blocks = Transformer(
            width=in_dim,
            layers=n_layers,
            heads=num_heads,
        )
        self.ln1 = nn.LayerNorm(in_dim)
        self.ln2 = nn.LayerNorm(in_dim)

        self.out = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, out_dim),
            nn.Softplus(),
        )

    def forward(self, x):
        # x [bs, 65, 1024]
        scale_emb = self.scale_emb.repeat(x.shape[0], 1, 1)
        x = torch.cat([scale_emb, x], dim=1)
        x = self.ln1(x)

        x = x.permute(1, 0, 2)
        x = self.transformer_blocks(x)
        x = x.permute(1, 0, 2)

        x = self.ln2(x[:, :self.num_tokens, :])
        x = self.out(x)
        return x