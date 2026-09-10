"""Segmentation heads for the LatentArc Stage-2 articulated path.

Heads defined here, all produced by :func:`build_seg_head`:

- ``"linear"`` — Phase-3 diagnostic probe: a single per-point
  ``nn.Linear`` over the FM-decoder's last-block features.
- ``"pat"`` — Phase-4 PARTICULATE-style head: M slot embeddings (M = 2
  fixed for the MVP, slot 0 = context / slot 1 = active), bidirectional
  cross-attention blocks between point tokens and slot queries, and a
  per-(point, slot) MLP decoder for the binary mask logits. Mirrors
  PARTICULATE's ``Block`` and ``point_mask_decoder`` (see
  ``third_party/particulate/particulate/models.py:60-130, 177-260``).
- ``"image_grounded_pat"`` — Phase-4b head with an explicit
  image-grounded conditioning path. A learnable ``(M_bank, D)`` part
  bank (M_bank = 128 by default) cross-attends one-way into the
  concatenation of {post-SE image patch tokens, post-projection scene
  tokens}, producing an image-grounded summary the K = 2 PAT slots
  read from before the standard slot↔point exchange. Pretrained
  encoder/scene streams are not perturbed — attention is asymmetric
  (bank reads from images/scene, never the reverse). See
  ``docs/project_cur.md`` (Phase 4b) for the rationale.

The PAT block here uses stock ``nn.MultiheadAttention`` + ``nn.LayerNorm``
rather than PARTICULATE's qk-norm RMS variant — close enough for a first
cut. The PARTICULATE-faithful version can plug in later if accuracy
warrants the extra complexity.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Linear probe (Phase 3)
# ---------------------------------------------------------------------------


class LinearSegProbe(nn.Module):
    """Per-point linear probe over decoder features.

    Input:  ``[B, N, in_dim]``
    Output: ``[B, N, num_classes]``
    """

    def __init__(self, in_dim: int, num_classes: int = 2):
        super().__init__()
        self.proj = nn.Linear(in_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)


# ---------------------------------------------------------------------------
# PAT seg head (Phase 4)
# ---------------------------------------------------------------------------


class PATBlock(nn.Module):
    """Bidirectional cross-attention block over (point tokens x, slot queries q).

    Mirrors PARTICULATE's ``Block`` (``particulate/models.py:60-130``):

    1. Query self-attention.
    2. Point-to-query cross-attention (q reads from x).
    3. Query-to-point cross-attention (x reads from q).
    4. Two parallel MLP residuals (one for x, one for q).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.attn_q_self = nn.MultiheadAttention(
            dim, n_heads, batch_first=True, dropout=dropout
        )

        self.norm_q2 = nn.LayerNorm(dim)
        self.attn_q_from_x = nn.MultiheadAttention(
            dim, n_heads, batch_first=True, dropout=dropout
        )

        self.norm_x = nn.LayerNorm(dim)
        self.attn_x_from_q = nn.MultiheadAttention(
            dim, n_heads, batch_first=True, dropout=dropout
        )

        self.norm_x_ff = nn.LayerNorm(dim)
        self.norm_q_ff = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp_x = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )
        self.mlp_q = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(
        self, x: torch.Tensor, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_n = self.norm_q1(q)
        q = q + self.attn_q_self(q_n, q_n, q_n, need_weights=False)[0]

        q_n = self.norm_q2(q)
        q = q + self.attn_q_from_x(q_n, x, x, need_weights=False)[0]

        x_n = self.norm_x(x)
        x = x + self.attn_x_from_q(x_n, q, q, need_weights=False)[0]

        x = x + self.mlp_x(self.norm_x_ff(x))
        q = q + self.mlp_q(self.norm_q_ff(q))
        return x, q


class PointMaskDecoder(nn.Module):
    """Per-(point, slot) logit via concat([x_i, q_m]) -> 2-layer MLP -> scalar.

    Output shape ``[B, N, M]``. Same shape as the linear-probe head, so the
    trainer's CE + diagnostics code is unchanged across Phase 3 / Phase 4.
    """

    def __init__(self, dim: int, hidden_ratio: int = 4):
        super().__init__()
        hidden = dim * hidden_ratio
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D], q: [B, M, D]
        B, N, D = x.shape
        M = q.shape[1]
        x_exp = x.unsqueeze(2).expand(-1, -1, M, -1)
        q_exp = q.unsqueeze(1).expand(-1, N, -1, -1)
        return self.mlp(torch.cat([x_exp, q_exp], dim=-1)).squeeze(-1)


class PATSegHead(nn.Module):
    """PARTICULATE-style PAT segmenter with M fixed slot embeddings.

    For the LatentArc MVP we use M = 2 (slot 0 = context, slot 1 = active
    part). No Hungarian matching — slot-to-GT-class assignment is fixed,
    so CE supervision aligns directly with the slot order.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 2,
        num_blocks: int = 2,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        # M = num_classes learnable slot embeddings.
        self.slot_embed = nn.Parameter(torch.randn(num_classes, in_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                PATBlock(in_dim, n_heads, mlp_ratio, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.point_mask_decoder = PointMaskDecoder(in_dim)

    def forward(
        self, features: torch.Tensor, return_slots: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = features
        B = x.shape[0]
        q = self.slot_embed.unsqueeze(0).expand(B, -1, -1).contiguous()
        for block in self.blocks:
            x, q = block(x, q)
        logits = self.point_mask_decoder(x, q)
        if return_slots:
            # Final slot embeddings, ``[B, num_classes, in_dim]`` — slot 0 is
            # context, slot 1 is the active part. Phase-5 articulation heads
            # read from slot 1.
            return logits, q
        return logits


# ---------------------------------------------------------------------------
# Image-grounded PAT seg head (Phase 4b)
# ---------------------------------------------------------------------------


class BankCrossAttnLayer(nn.Module):
    """One-way cross-attention layer: bank queries attend into image+scene K/V.

    Asymmetric by construction — the K/V tensor is never updated, so the
    pretrained encoder/scene-token streams keep their gradient paths clean.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, batch_first=True, dropout=dropout
        )
        self.norm_ff = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        q = q + self.attn(q_n, kv_n, kv_n, need_weights=False)[0]
        q = q + self.mlp(self.norm_ff(q))
        return q


class ImagePartBank(nn.Module):
    """Stage A of the image-grounded seg head.

    A learnable ``(num_slots, dim)`` bank cross-attends into the
    concatenation of {image patch tokens, scene tokens}. The bank acts
    as a fixed-capacity latent reservoir for image-grounded part
    information — capacity (``num_slots``) is decoupled from the final
    class count handed to the PAT slots.

    Image patch tokens are typically the post-aggregation
    frame+global intermediates ``[B, S, P, image_in_dim]`` (with
    ``image_in_dim = 2 * embed_dim``); scene tokens are the projected
    NOVA3R 3D tokens ``[B, K, scene_in_dim]``. Camera + register tokens
    can be dropped via ``patch_start_idx`` on the call side.
    """

    def __init__(
        self,
        dim: int,
        image_in_dim: int,
        scene_in_dim: int,
        num_slots: int = 128,
        num_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        kv_sources: tuple[str, ...] | list[str] = ("image", "scene"),
    ):
        super().__init__()
        kv_sources = tuple(kv_sources)
        for src in kv_sources:
            if src not in ("image", "scene"):
                raise ValueError(
                    f"kv_sources entries must be 'image' or 'scene', got {src!r}"
                )
        if not kv_sources:
            raise ValueError("kv_sources must contain at least one of 'image' / 'scene'")
        self.kv_sources = kv_sources

        self.bank = nn.Parameter(torch.randn(num_slots, dim) * 0.02)
        # Only instantiate the projection for an active source — keeps the
        # parameter count honest when comparing the K/V ablation cells.
        self.img_proj = (
            (nn.Linear(image_in_dim, dim) if image_in_dim != dim else nn.Identity())
            if "image" in kv_sources
            else None
        )
        self.scene_proj = (
            (nn.Linear(scene_in_dim, dim) if scene_in_dim != dim else nn.Identity())
            if "scene" in kv_sources
            else None
        )
        self.layers = nn.ModuleList(
            [
                BankCrossAttnLayer(dim, n_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        image_tokens: torch.Tensor,
        scene_tokens: torch.Tensor,
        patch_start_idx: int = 0,
    ) -> torch.Tensor:
        # image_tokens: [B, S, P, image_in_dim]
        # scene_tokens: [B, K, scene_in_dim]
        kv_parts: list[torch.Tensor] = []
        if "image" in self.kv_sources:
            if patch_start_idx and patch_start_idx > 0:
                image_tokens = image_tokens[:, :, patch_start_idx:, :]
            B, S, P_eff, _ = image_tokens.shape
            img_proj = self.img_proj(image_tokens)              # [B, S, P', D]
            kv_parts.append(img_proj.reshape(B, S * P_eff, -1)) # [B, S*P', D]
        if "scene" in self.kv_sources:
            kv_parts.append(self.scene_proj(scene_tokens))      # [B, K, D]
        kv = kv_parts[0] if len(kv_parts) == 1 else torch.cat(kv_parts, dim=1)
        B = kv.shape[0]
        bank = self.bank.unsqueeze(0).expand(B, -1, -1).contiguous()
        for layer in self.layers:
            bank = layer(bank, kv)
        return bank


class PATBlockWithBank(nn.Module):
    """PAT block extended with a slot↔bank cross-attention step.

    Per block:
      1. q-self-attn (slots talk to each other).
      2. q-from-bank cross-attn (slots read image-grounded summary).
      3. q-from-x cross-attn (slots read per-point features).
      4. x-from-q cross-attn (per-point features absorb slot state).
      5. parallel MLPs on x and q.

    Step 2 is the only addition over :class:`PATBlock`.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.attn_q_self = nn.MultiheadAttention(
            dim, n_heads, batch_first=True, dropout=dropout
        )

        self.norm_q_bank = nn.LayerNorm(dim)
        self.attn_q_from_bank = nn.MultiheadAttention(
            dim, n_heads, batch_first=True, dropout=dropout
        )

        self.norm_q2 = nn.LayerNorm(dim)
        self.attn_q_from_x = nn.MultiheadAttention(
            dim, n_heads, batch_first=True, dropout=dropout
        )

        self.norm_x = nn.LayerNorm(dim)
        self.attn_x_from_q = nn.MultiheadAttention(
            dim, n_heads, batch_first=True, dropout=dropout
        )

        self.norm_x_ff = nn.LayerNorm(dim)
        self.norm_q_ff = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp_x = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )
        self.mlp_q = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        bank: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_n = self.norm_q1(q)
        q = q + self.attn_q_self(q_n, q_n, q_n, need_weights=False)[0]

        q_n = self.norm_q_bank(q)
        q = q + self.attn_q_from_bank(q_n, bank, bank, need_weights=False)[0]

        q_n = self.norm_q2(q)
        q = q + self.attn_q_from_x(q_n, x, x, need_weights=False)[0]

        x_n = self.norm_x(x)
        x = x + self.attn_x_from_q(x_n, q, q, need_weights=False)[0]

        x = x + self.mlp_x(self.norm_x_ff(x))
        q = q + self.mlp_q(self.norm_q_ff(q))
        return x, q


class ImageGroundedPATSegHead(nn.Module):
    """Phase-4b head: bank-grounded PAT segmenter.

    Forward expects per-point features ``[B, N, in_dim]`` *and*
    ``image_tokens`` / ``scene_tokens`` that the trainer must pass
    explicitly (they are not in the per-point feature stream). The
    bank dim is locked to ``in_dim`` so slots, bank, and point tokens
    share a single hidden size — keeps the cross-attention clean.
    """

    def __init__(
        self,
        in_dim: int,
        image_in_dim: int,
        scene_in_dim: int,
        num_classes: int = 2,
        bank_size: int = 128,
        bank_layers: int = 2,
        pat_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        kv_sources: tuple[str, ...] | list[str] = ("image", "scene"),
    ):
        super().__init__()
        D = in_dim
        self.bank = ImagePartBank(
            dim=D,
            image_in_dim=image_in_dim,
            scene_in_dim=scene_in_dim,
            num_slots=bank_size,
            num_layers=bank_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            kv_sources=kv_sources,
        )
        self.slot_embed = nn.Parameter(torch.randn(num_classes, D) * 0.02)
        self.blocks = nn.ModuleList(
            [
                PATBlockWithBank(D, n_heads, mlp_ratio, dropout)
                for _ in range(pat_layers)
            ]
        )
        self.point_mask_decoder = PointMaskDecoder(D)

    def forward(
        self,
        features: torch.Tensor,
        image_tokens: torch.Tensor,
        scene_tokens: torch.Tensor,
        patch_start_idx: int = 0,
        return_slots: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        bank_tokens = self.bank(
            image_tokens, scene_tokens, patch_start_idx=patch_start_idx
        )
        x = features
        B = x.shape[0]
        q = self.slot_embed.unsqueeze(0).expand(B, -1, -1).contiguous()
        for block in self.blocks:
            x, q = block(x, q, bank_tokens)
        logits = self.point_mask_decoder(x, q)
        if return_slots:
            # Final slot embeddings, ``[B, num_classes, in_dim]`` — slot 0
            # is context, slot 1 is the active part. Phase-5 articulation
            # heads read from slot 1.
            return logits, q
        return logits


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_seg_head(in_dim: int, cfg: dict, **extra) -> nn.Module:
    """Construct a seg head from a config dict.

    Recognized config keys:
        - ``type``: ``"linear"`` (Phase 3), ``"pat"`` (Phase 4), or
          ``"image_grounded_pat"`` (Phase 4b).
        - ``num_classes``: int, defaults to 2 (slot-0 ctx, slot-1 active).
        - PAT-only: ``num_blocks`` (2), ``n_heads`` (4), ``mlp_ratio`` (4.0),
          ``dropout`` (0.0).
        - Image-grounded PAT: ``bank_size`` (128), ``bank_layers`` (2),
          ``pat_layers`` (2), shares ``n_heads`` / ``mlp_ratio`` / ``dropout``
          with PAT defaults. Requires ``image_in_dim`` and ``scene_in_dim``
          via ``extra`` (the wrapping model knows these from its config).
    """
    head_type = cfg.get("type", "linear")
    num_classes = int(cfg.get("num_classes", 2))
    if head_type == "linear":
        return LinearSegProbe(in_dim=in_dim, num_classes=num_classes)
    if head_type == "pat":
        return PATSegHead(
            in_dim=in_dim,
            num_classes=num_classes,
            num_blocks=int(cfg.get("num_blocks", 2)),
            n_heads=int(cfg.get("n_heads", 4)),
            mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    if head_type == "image_grounded_pat":
        if "image_in_dim" not in extra or "scene_in_dim" not in extra:
            raise ValueError(
                "image_grounded_pat requires 'image_in_dim' and 'scene_in_dim' "
                "to be passed via extra kwargs (the wrapping model owns them)."
            )
        kv_sources = cfg.get("kv_sources", ["image", "scene"])
        if isinstance(kv_sources, str):
            kv_sources = [kv_sources]
        return ImageGroundedPATSegHead(
            in_dim=in_dim,
            image_in_dim=int(extra["image_in_dim"]),
            scene_in_dim=int(extra["scene_in_dim"]),
            num_classes=num_classes,
            bank_size=int(cfg.get("bank_size", 128)),
            bank_layers=int(cfg.get("bank_layers", 2)),
            pat_layers=int(cfg.get("pat_layers", 2)),
            n_heads=int(cfg.get("n_heads", 4)),
            mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
            dropout=float(cfg.get("dropout", 0.0)),
            kv_sources=tuple(kv_sources),
        )
    raise ValueError(f"Unknown seg head type: {head_type!r}")
