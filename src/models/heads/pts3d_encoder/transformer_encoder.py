# Copyright (c) 2026 Weirong Chen
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers.hunyuan_block import FourierEmbedder

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)





class CrossAttentionBlock(nn.Module):
    def __init__(self, token_dim, point_dim, hidden_dim=128, num_heads=8):
        super().__init__()
        self.norm_kv = nn.LayerNorm(point_dim)
        self.norm_q = nn.LayerNorm(token_dim)

        self.attn = nn.MultiheadAttention(embed_dim=token_dim, kdim=point_dim, vdim=point_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)
        self.mlp = MLP(token_dim, token_dim * 4)


    def forward(self, tokens, points):

        tokens = self.norm_q(tokens)
        points = self.norm_kv(points)

        tokens = tokens + self.attn(tokens, points, points)[0]
        tokens = tokens + self.mlp(self.norm(tokens))
        return tokens

class SelfAttentionBlock(nn.Module):
    def __init__(self, token_dim, hidden_dim=128, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads, batch_first=True)
        self.mlp = MLP(token_dim, token_dim * 4)

    def forward(self, tokens):
        tokens = self.norm1(tokens)
        tokens = tokens + self.attn(tokens, tokens, tokens)[0]
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=3, k=512, df=64, df_out=16, d_point=64, cross_depth=6, self_depth=2, hidden_dim=128, num_heads=8):
        super().__init__()

        self.pts3d_embed = FourierEmbedder(num_freqs=8)
        self.pts3d_proj = nn.Linear(input_dim * 16 + input_dim, d_point)

        self.self_blocks = nn.ModuleList([
            SelfAttentionBlock(df, hidden_dim, num_heads) for _ in range(self_depth)
        ])

        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(df, d_point, hidden_dim, num_heads) for _ in range(cross_depth)
        ])

        self.linear_out = nn.Linear(df, df_out)


        self.shape_tokens = nn.Parameter(torch.randn(k, df), requires_grad=True)
        # initialize the shape tokens
        nn.init.xavier_uniform_(self.shape_tokens)


    def forward(self, points):

        pts3d_embed = self.pts3d_embed(points)
        pts3d_embed = self.pts3d_proj(pts3d_embed)


        tokens = self.shape_tokens.unsqueeze(0).expand(pts3d_embed.size(0), -1, -1)

        for block in self.cross_blocks:
            tokens = block(tokens, pts3d_embed)
        for block in self.self_blocks:
            tokens = block(tokens)

        tokens = self.linear_out(tokens)
        return tokens
