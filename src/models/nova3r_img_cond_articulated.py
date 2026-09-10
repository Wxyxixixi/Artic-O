"""Articulated extension of :class:`Nova3rImgCond`.

LatentArc Stage 2 adds two articulation-specific outputs on top of NOVA3R's
image-to-latent path:

1. Per-point segmentation logits (active part vs context).
2. Per-active-part articulation parameters (motion class, axis, range).

(2) lands in Phase 5. This module wires up (1) for Phase 3 — the feature
diagnostic — using a forward pre-hook on the FM decoder's ``linear_out``
to capture the last-block per-point features without modifying the
upstream NOVA3R FM decoder code.

It also (optionally) adds a per-view state embedding: a small
``nn.Embedding(2, embed_dim)`` indexed by ``state_tag`` (0 = s0, 1 = s1)
whose lookup is plumbed through ``_encode`` → ``forward_vggt`` →
``AggregatorPts3D.forward`` as a ``view_offset`` tensor and added to
each view's patch tokens *after* the optional ViT detach. Going
through the call chain (rather than via a forward hook on
``patch_embed``) is what keeps the offset on the live side of
``patch_tokens.detach()`` so the embedding table actually receives
gradients. Zero-init means the module is a no-op on day 0 and does
not perturb the loaded NOVA3R pretrained weights — gradients learn
the offset only if it actually helps.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.heads.articulation import ArticulationHead
from src.models.heads.matcher import hungarian_match_movable
from src.models.heads.seg_pat import ImageGroundedPATSegHead, build_seg_head
from src.models.nova3r_img_cond import Nova3rImgCond


class Nova3rImgCondArticulated(Nova3rImgCond):
    """``Nova3rImgCond`` with a seg head + per-point feature capture +
    optional per-view state embedding.

    The per-point feature ``x`` (input to ``linear_out`` in
    :class:`PointJointFMDecoderV2`) is captured via
    ``register_forward_pre_hook`` so the underlying NOVA3R decoder file
    stays unmodified.

    When ``state_embedding=True`` a learnable
    ``nn.Embedding(2, aggregator.embed_dim)`` is created (zero-init) and
    added to each view's patch tokens — broadcast across all patch
    positions of that view — by passing a ``view_offset`` tensor through
    ``_encode`` / ``forward_vggt`` into the aggregator. The aggregator
    applies it after the ViT-feature detach so the offset's gradient
    path survives. The trainer must pass ``state_tag`` of shape
    ``[B, S]`` into ``forward`` / ``_encode``; if absent (or if
    ``state_embedding=False``) the offset is ``None`` and the
    aggregator skips the addition.

    Forward kwargs (in addition to the base class):
        run_seg (bool): When True, the seg head runs over the captured
            per-point features and the result is added to the prediction
            dict under ``seg_logits``.
        state_tag (Tensor, optional): per-view state ids of shape
            ``[B, S]`` with values in ``{0, 1}``. Looked up against the
            state-embedding table during ``_encode`` and passed to the
            aggregator as ``view_offset``; ignored on the decode-only
            path (``encoder_data`` provided), since the offset has
            already been baked into the cached tokens.
        encoder_data (dict, optional): When provided, skip the image
            encoder and decode directly from ``encoder_data['tokens']``.
            Lets the trainer encode once per step and reuse the latent
            across multiple decoder passes (e.g. FM-loss pass at random
            ``t`` + feature pass at ``t = 1``), which is the dominant
            cost in this pipeline. ``state_tag`` was already consumed
            during the upstream ``_encode`` so it is not re-applied here.
    """

    def __init__(
        self,
        *args,
        seg_head: dict | None = None,
        state_embedding: bool = False,
        articulation_head: bool | dict = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Buffer for the most recent per-point feature tensor captured by
        # the pre-hook. Reset at the start of every forward.
        self._captured_features: torch.Tensor | None = None

        feat_dim = int(self.pts3d_head.linear_out.in_features)
        seg_cfg = dict(seg_head) if seg_head is not None else {}
        seg_cfg.setdefault("type", "linear")
        seg_cfg.setdefault("num_classes", 2)
        # >2 classes => multi-part mode (slot 0 = static, 1..P0 = movable parts).
        self.num_seg_classes = int(seg_cfg["num_classes"])
        # Image-grounded PAT needs to know the dim of the image and scene
        # K/V tensors to set up its projections. The wrapping model owns
        # those dims via the aggregator config:
        #   image_in_dim = 2 * embed_dim — post-aggregation concat of frame
        #     and global intermediates (``output_list[-1]`` in the aggregator).
        #   scene_in_dim = token_dim — the post-``img_token_proj`` scene
        #     tokens in ``encode_data['tokens']`` (also what the FM decoder
        #     receives as conditioning, so the seg head sees the same scene
        #     summary the geometry path does).
        embed_dim = int(self.cfg.aggregator.params.embed_dim)
        token_dim = int(self.cfg.aggregator.params.token_dim)
        self.seg_head = build_seg_head(
            in_dim=feat_dim,
            cfg=seg_cfg,
            image_in_dim=2 * embed_dim,
            scene_in_dim=token_dim,
        )

        self.pts3d_head.linear_out.register_forward_pre_hook(
            self._capture_features_hook
        )

        # ----- optional state embedding -----------------------------------
        # The embedding offset is plumbed through ``_encode`` →
        # ``forward_vggt`` → aggregator and added to patch tokens AFTER
        # the ViT-feature detach (``detach_vit_token``). This is critical:
        # if we instead added it via a forward hook on ``patch_embed``,
        # the aggregator's ``patch_tokens.detach()`` line would sever the
        # offset's gradient path, and the embedding table would never be
        # trained. Passing the offset along the call chain keeps it on
        # the live side of the detach.
        self.state_embedding_table: nn.Embedding | None = None
        if state_embedding:
            embed_dim = int(self.cfg.aggregator.params.embed_dim)
            table = nn.Embedding(2, embed_dim)
            nn.init.zeros_(table.weight)
            self.state_embedding_table = table

        # ----- optional articulation head (Phase 5) -----------------------
        # Reads the slot-1 (active part) embedding emitted by the seg head
        # and predicts (revolute Plücker, revolute range, prismatic axis,
        # prismatic range). Both motion-type branches always run; the
        # trainer gates the loss by GT motion type. When the seg head is
        # the linear probe (no slot embeddings), the articulation head is
        # a no-op; the trainer should not enable both at once.
        self.articulation_head: ArticulationHead | None = None
        if articulation_head:
            art_cfg = articulation_head if isinstance(articulation_head, dict) else {}
            hidden_ratio = int(art_cfg.get("hidden_ratio", 4))
            motion_repr = str(art_cfg.get("motion_representation", "plucker"))
            self.articulation_head = ArticulationHead(
                dim=feat_dim,
                hidden_ratio=hidden_ratio,
                motion_representation=motion_repr,
            )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _capture_features_hook(self, module: nn.Module, inputs: tuple) -> None:
        # ``inputs`` is the positional tuple to ``linear_out.forward``.
        # ``Linear`` is called as ``linear_out(x)`` -> ``inputs == (x,)``.
        self._captured_features = inputs[0]

    # ------------------------------------------------------------------
    # Forward overrides — derive view_offset from state_tag and pass it
    # ------------------------------------------------------------------

    def _state_view_offset(
        self, state_tag: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Look up the per-view state embedding from ``state_tag``.

        Returns ``None`` when the state-embedding table is disabled or
        the trainer didn't pass ``state_tag``, so the aggregator falls
        back to the unmodified path (no offset).
        """
        if self.state_embedding_table is None or state_tag is None:
            return None
        device = self.state_embedding_table.weight.device
        return self.state_embedding_table(state_tag.to(device).long())  # [B, S, C]

    def _encode(self, images, state_tag: torch.Tensor | None = None, **kwargs):
        view_offset = self._state_view_offset(state_tag)
        return super()._encode(images, view_offset=view_offset, **kwargs)

    def forward(
        self,
        *,
        run_seg: bool = False,
        state_tag: torch.Tensor | None = None,
        encoder_data: dict | None = None,
        mp_part_ids: torch.Tensor | None = None,
        mp_num_movable: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        # Reset capture so a stale tensor cannot leak between steps if
        # the underlying forward fails silently before reaching linear_out.
        self._captured_features = None

        images = kwargs.get("images", None)
        if encoder_data is None:
            # Full path: encode here with ``state_tag`` plumbed through
            # ``_state_view_offset`` -> ``view_offset``, then carry the
            # full ``encoder_data`` (incl. image/scene tokens) into the
            # seg-head dispatch below.
            view_offset = self._state_view_offset(state_tag)
            encoder_data = super()._encode(images, view_offset=view_offset)
        # else: caller already encoded; ``state_tag`` was consumed there.

        tokens = encoder_data["tokens"]
        predictions = self._decode(
            tokens=tokens,
            images=images,
            token_mask=kwargs.get("token_mask", None),
            query_points=kwargs.get("query_points", None),
            timestep=kwargs.get("timestep", None),
        )

        feats = self._captured_features
        if run_seg:
            if feats is None:
                raise RuntimeError(
                    "Nova3rImgCondArticulated.forward(run_seg=True) was called "
                    "but the FM decoder's linear_out did not run — features "
                    "were never captured. Check that query_points/timestep "
                    "are passed and that pts3d_head is enabled."
                )
            predictions["point_features"] = feats
            seg_logits, slots = self._run_seg_head(
                feats, encoder_data, return_slots=True,
            )
            predictions["seg_logits"] = seg_logits
            predictions["slots"] = slots  # [B, num_classes, dim]
            if self.articulation_head is not None:
                if slots is not None and self.num_seg_classes > 2:
                    # Multi-part: all movable slots 1..P0 (slot 0 = static).
                    slots_movable = slots[:, 1:, :]                 # [B, P0, dim]
                    predictions["slots_movable"] = slots_movable
                    art_out = self.articulation_head.forward_slots(slots_movable)
                    if mp_part_ids is not None:
                        # Teacher-forced (GT-part) Hungarian matching + per-point
                        # closest-pt, computed INSIDE forward so DDP tracks the
                        # point_axis_decoder params. Matching is non-differentiable.
                        target_col, matched_slot = hungarian_match_movable(
                            seg_logits, mp_part_ids, mp_num_movable,
                        )
                        Bb, Nn = mp_part_ids.shape
                        Dd = feats.shape[-1]
                        part_idx = (mp_part_ids - 1).clamp(min=0)          # [B, N]
                        ms_safe = matched_slot.clamp(min=0)                # [B, P0]
                        pslot = torch.gather(ms_safe, 1, part_idx)         # [B, N]
                        slot_emb = torch.gather(
                            slots_movable, 1,
                            pslot.unsqueeze(-1).expand(Bb, Nn, Dd),
                        )                                                  # [B, N, D]
                        art_out["revolute_closest_pt"] = (
                            self.articulation_head.closest_pt(feats, slot_emb)
                        )
                        predictions["mp_target_col"] = target_col
                        predictions["mp_matched_slot"] = matched_slot
                    predictions["articulation"] = art_out
                elif slots is not None:
                    # Single active part: slot 1 (slot 0 is context).
                    slot_active = slots[:, 1, :]
                    predictions["articulation"] = self.articulation_head(
                        slot_active, point_features=feats,
                    )
                else:
                    # Linear seg has no slots. For the point-only-seg
                    # ablation we synthesise a pseudo-slot by pooling
                    # per-point features weighted by predicted active-prob,
                    # giving the articulation head an active-part embedding
                    # of the same dim as a real slot[1].
                    active_prob = torch.softmax(seg_logits, dim=-1)[..., 1:2]  # [B, N, 1]
                    slot_active = (active_prob * feats).sum(dim=1) / (
                        active_prob.sum(dim=1).clamp_min(1e-6)
                    )
                    predictions["articulation"] = self.articulation_head(
                        slot_active, point_features=feats,
                    )
        return predictions

    # ------------------------------------------------------------------
    # Seg-head dispatch
    # ------------------------------------------------------------------

    def _run_seg_head(
        self,
        feats: torch.Tensor,
        encoder_data: dict,
        return_slots: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Route to the right seg-head signature.

        ``ImageGroundedPATSegHead`` needs the post-SE image tokens and the
        scene tokens in addition to the per-point features; the linear /
        plain-PAT heads only consume the per-point features. Splitting
        the dispatch here keeps the trainer agnostic to the seg-head
        choice.

        When ``return_slots=True``, PAT-family heads also return their
        final slot embeddings (the linear probe doesn't have slots, so
        it returns ``None`` for the slots field).
        """
        if isinstance(self.seg_head, ImageGroundedPATSegHead):
            image_tokens = encoder_data.get("image_tokens")
            if image_tokens is None:
                raise RuntimeError(
                    "image_grounded_pat requires 'image_tokens' in encoder_data — "
                    "the encoder must surface aggregator output_list[-1]."
                )
            patch_start_idx = int(encoder_data.get("patch_start_idx", 0) or 0)
            out = self.seg_head(
                feats,
                image_tokens=image_tokens,
                scene_tokens=encoder_data["tokens"],
                patch_start_idx=patch_start_idx,
                return_slots=return_slots,
            )
            return out
        # Plain PAT: also accepts return_slots.
        try:
            return self.seg_head(feats, return_slots=return_slots)
        except TypeError:
            # Linear probe doesn't take return_slots (no slots). Return
            # a tuple with ``None`` so callers get a uniform shape.
            logits = self.seg_head(feats)
            if return_slots:
                return logits, None
            return logits
