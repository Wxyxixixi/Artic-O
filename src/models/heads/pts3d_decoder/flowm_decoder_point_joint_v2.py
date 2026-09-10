# Copyright (c) 2026 Weirong Chen

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..head_act import activate_head
from ...layers.mlp import Mlp
from einops import rearrange, repeat
import math
import time

from third_party.croco.models.blocks import DecoderBlock
from src.models.layers.rope import RotaryPositionEmbedding2D, RotaryPositionEmbedding4D, PositionGetter
from src.models.layers.block import Block
from src.models.layers import PatchEmbed

from src.models.layers.hunyuan_block import CrossAttentionDecoder, FourierEmbedder
from src.models.heads.pts3d_decoder.utils import pred_act, conf_act

def modulate(x, shift, scale):
    if len(x.shape) == len(shift.shape) + 1:
        # If x is [B, N, D] and shift/scale are [B, D], we need to unsqueeze them
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class PointEmbedded(nn.Module):
    pass



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
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

    def forward(self, query, key, value):
        # query, key, value: [B, N, d]
        out, _ = self.attn(query, key, value)
        return out


class FMBlock(nn.Module):
    def __init__(self, d_model, d_point, d_time, d_shape, num_heads=8, mlp_ratio=4, dropout=0.0, activation='gelu'):
        super().__init__()
        # self.shift_scale1 = nn.Sequential(
        #     nn.Linear(d_time, d_model),
        #     nn.SiLU(),
        #     nn.Linear(d_model, d_model)
        # )

        self.norm_point = nn.LayerNorm(d_point)
        self.proj_point = nn.Linear(d_point, d_model)

        self.cross_attn = CrossAttentionBlock(d_model, num_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_model * 4)

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(d_time, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )



    def forward(self, pos_encoded_time, pos_encoded_point, shape_tokens):
        """
        pos_encoded_time: [B, d_time]
        pos_encoded_point: [B, N, d_point]
        shape_tokens: [B, K, d_shape]
        """
        B, N, _ = pos_encoded_point.shape

        # Time-based modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(pos_encoded_time).chunk(6, dim=-1)

        if len(gate_msa.shape) != len(pos_encoded_point.shape):
            # If gate_msa is [B, d_model], we need to unsqueeze it to [B, 1, d_model]
            gate_msa = gate_msa.unsqueeze(1)
            gate_mlp = gate_mlp.unsqueeze(1)

        x = pos_encoded_point  # [B, N, d_point]

        x1 = modulate(self.norm1(x), shift_msa, scale_msa)  # [B, N, d_model]
        x = x + gate_msa * self.cross_attn(x1, shape_tokens, shape_tokens)

        x2 = modulate(self.norm2(x), shift_mlp, scale_mlp)  # [B, N, d_model]
        x = x + gate_mlp * self.mlp(x2)  # [B, N, d_model]

        return x


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class Attention(nn.Module):
    def __init__(
        self, query_dim, context_dim=None, num_heads=8, qkv_bias=False, use_sdpa=False
    ):
        super().__init__()
        # inner_dim = dim_head * num_heads
        dim_head = query_dim // num_heads
        inner_dim = query_dim
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = num_heads
        self.use_sdpa = use_sdpa

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, attn_bias=None):
        B, N1, C = x.shape
        h = self.heads
        q = self.to_q(x).reshape(B, N1, h, C // h).permute(0, 2, 1, 3)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        N2 = context.shape[1]
        k = k.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        v = v.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)

        if self.use_sdpa:
            attn_mask = attn_bias
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )  # [B,h,N1,d]
            out = out.transpose(1, 2).contiguous().view(B, N1, C)
            return self.to_out(out)

        else:
            sim = (q @ k.transpose(-2, -1)) * self.scale

            if attn_bias is not None:
                sim = sim + attn_bias
            attn = sim.softmax(dim=-1)

            x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
            return self.to_out(x)





class CrossAttnBlock(nn.Module):
    def __init__(
        self, hidden_size, context_dim, time_dim=None, num_heads=1, mlp_ratio=4.0, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)
        self.cross_attn = Attention(
            hidden_size,
            context_dim=context_dim,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

        if time_dim is not None:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(time_dim, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.time_dim = time_dim


    def forward(self, x, context, pos_encoded_time=None, mask=None):
        attn_bias = None
        if pos_encoded_time is not None and self.time_dim is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(pos_encoded_time).chunk(6, dim=-1)

            if len(gate_msa.shape) != len(x.shape):
                gate_msa = gate_msa.unsqueeze(1)
                gate_mlp = gate_mlp.unsqueeze(1)
        else:
            shift_msa, scale_msa, gate_msa = torch.zeros_like(x), torch.ones_like(x), torch.ones_like(x)
            shift_mlp, scale_mlp, gate_mlp = torch.zeros_like(x), torch.ones_like(x), torch.ones_like(x)

        if mask is not None:
            if mask.shape[1] == x.shape[1]:
                mask = mask[:, None, :, None].expand(
                    -1, self.cross_attn.heads, -1, context.shape[1]
                )
            else:
                mask = mask[:, None, None].expand(
                    -1, self.cross_attn.heads, x.shape[1], -1
                )

            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value

        x1 = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.cross_attn(
            x1, context=self.norm_context(context), attn_bias=attn_bias
        )
        x2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x2)
        return x


class CrossAttnContextBlock(nn.Module):
    def __init__(
        self, hidden_size, context_dim, time_dim=None, num_heads=1, mlp_ratio=4.0, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)
        self.cross_attn = Attention(
            hidden_size,
            context_dim=context_dim,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

        if time_dim is not None:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(time_dim, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )
        self.time_dim = time_dim


    def forward(self, x, context, pos_encoded_time=None, mask=None):
        attn_bias = None
        if pos_encoded_time is not None and self.time_dim is not None:
            shift_msa, scale_msa = self.adaLN_modulation(pos_encoded_time).chunk(2, dim=-1)
        else:
            shift_msa, scale_msa = torch.zeros_like(context), torch.ones_like(context)

        if mask is not None:
            if mask.shape[1] == x.shape[1]:
                mask = mask[:, None, :, None].expand(
                    -1, self.cross_attn.heads, -1, context.shape[1]
                )
            else:
                mask = mask[:, None, None].expand(
                    -1, self.cross_attn.heads, x.shape[1], -1
                )

            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value

        # x1 = modulate(self.norm1(x), shift_msa, scale_msa)
        x1 = self.norm1(x)

        context_mod = modulate(self.norm_context(context), shift_msa, scale_msa)

        x = x + self.cross_attn(
            x1, context=context_mod, attn_bias=attn_bias
        )
        # x2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

class AttnBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        attn_class=Attention,
        mlp_ratio=4.0,
        **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = attn_class(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, mask=None):
        attn_bias = mask
        if mask is not None:
            mask = (
                (mask[:, None] * mask[:, :, None])
                .unsqueeze(1)
                .expand(-1, self.attn.num_heads, -1, -1)
            )
            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value
        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x


class PointJointFMDecoderV2(nn.Module):
    """Flow matching decoder that jointly processes visual and 3D tokens to predict point cloud coordinates.

    Uses cross-attention between query points and encoded tokens, with time-step
    conditioning for the ODE-based generation process.
    """

    def __init__(self, has_conf=False, dim_in=1024, output_dim=6, patch_3d_size=256, conf_activation="expp1", rgb_activation='sigmoid', pts_activation='linear', num_latents=512,  query_dim=3, cross_depth=3, self_depth=3, dim_model=32, num_virtual_tracks=512, use_sdpa=False, use_num_view_cond=False, **kwargs):
        super().__init__()
        # self.patch_size = net.patch_embed.patch_size[0]
        self.patch_3d_size = patch_3d_size
        self.conf_activation = conf_activation
        self.pts_activation = pts_activation
        self.rgb_activation = rgb_activation

        self.has_conf = has_conf
        self.dim_in = dim_in

        self.use_sdpa = use_sdpa
        self.use_num_view_cond = use_num_view_cond

        self.fourier_embedder = FourierEmbedder(num_freqs=8, include_pi=True)

        self.mlp_token = nn.Linear(dim_in, dim_model)

        d_model = dim_model
        d_point = d_model
        d_time = d_model
        d_shape = d_model

        self.pts3d_embed = nn.Linear(query_dim * 16 + query_dim, d_model)
        # self.time_embed = nn.Linear(1, d_model)
        self.t_embedder = TimestepEmbedder(d_model)

        if self.use_num_view_cond:
            self.view_embedder = TimestepEmbedder(d_model)  # Assuming max 100 views, can be changed to something else

        # self.cross_blocks = nn.ModuleList([
        #     FMBlock(d_model, d_point, d_time, d_shape) for _ in range(cross_depth)
        # ])

        assert self_depth == cross_depth
        num_heads = 8
        mlp_ratio = 4.0

        self.self_depth = self_depth
        self.self_virtual_blocks = nn.ModuleList([
            AttnBlock(d_model, num_heads=num_heads, mlp_ratio=mlp_ratio, attn_class=Attention, use_sdpa=self.use_sdpa) for _ in range(self_depth)
        ])

        self.self_point2virtual_blocks = nn.ModuleList(
            [
                CrossAttnBlock(
                    d_model, d_model, time_dim=d_time, num_heads=num_heads, mlp_ratio=mlp_ratio, use_sdpa=self.use_sdpa
                )
                for _ in range(self_depth)
            ]
        )

        self.self_virtual2point_blocks = nn.ModuleList(
            [
                CrossAttnContextBlock(
                    d_model, d_model, num_heads=num_heads, time_dim=d_time, mlp_ratio=mlp_ratio,
                    use_sdpa=self.use_sdpa
                )
                for _ in range(self_depth)
            ]
        )

        self.virual_tracks = nn.Parameter(
            torch.randn(1, num_virtual_tracks, d_model)
        )
        self.num_virtual_tracks = num_virtual_tracks

        self.linear_out = nn.Linear(d_model, output_dim)


    def forward(self, decout, query_points=None, timestep=None, num_views=None, **kwargs):
        """
        query_points: [B, N, 3]
        timestep: [B, N]
        """
        start_time = time.time()

        tokens = decout[-1]
        B, S, D = tokens.shape

        tokens = self.mlp_token(tokens)

        timestep = rearrange(timestep, 'b n -> (b n)')
        pos_encoded_time = self.t_embedder(timestep)   # [B, 1, 32]
        pos_encoded_time = rearrange(pos_encoded_time, '(b n) d -> b n d', b=B)  # [B, 1, d_model]

        if self.use_num_view_cond and num_views is not None:
            # assume num_views is [B], with values in [0, 7]
            num_views = repeat(num_views, 'b -> (b n)', n=query_points.shape[1])  # [B*N]
            num_views = num_views / 8.0  # likely normalize into [0, 1]
            pos_encoded_view = self.view_embedder(num_views)  # [B, d_model]
            pos_encoded_view = rearrange(pos_encoded_view, '(b n) d -> b n d', b=B)  # [B, 1, d_model]
            pos_encoded_time = pos_encoded_time + pos_encoded_view


        pts3d_embed = self.fourier_embedder(query_points)
        pos_encoded_query = self.pts3d_embed(pts3d_embed)  # [B, N, d_model]
        x = pos_encoded_query

        # virtual_tokens = self.virual_tracks.repeat(B, 1, 1)
        virtual_tokens = tokens # [B, K, d_model]

        # pos_encoded_time_k = repeat(pos_encoded_time[:,0], 'b d -> b k d', k=virtual_tokens.shape[1])  # [B, N, d_model]

        for i in range(self.self_depth):
            # block = self.cross_blocks[i]
            # x = block(pos_encoded_time, x, tokens)

            virtual_tokens = self.self_virtual2point_blocks[i](virtual_tokens, x, pos_encoded_time=pos_encoded_time)

            # Self attention on virtual tracks
            virtual_tokens = self.self_virtual_blocks[i](virtual_tokens)

            # Cross attention from point features to virtual tracks
            x = self.self_point2virtual_blocks[i](x, virtual_tokens, pos_encoded_time=pos_encoded_time)

            # add



        velocity = self.linear_out(x)  # [B, N, 3]

        return velocity
