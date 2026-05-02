#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""pi0.5 + VGGT 3D-Mix policy.

Architecture
------------
- A frozen VGGT-1B aggregator runs on the camera images and produces
  geometry-aware patch tokens F_VGGT in R^[B, S*P, 2*C_vggt].
- A linear projection W_proj maps F_VGGT -> F_geo in R^[B, N_patches, D_mllm].
  F_geo is computed once per forward pass and reused across all layers.
- At each fusion layer i in [fusion_layer_start, fusion_layer_end):
    s_global^(i)  = mean_j H_MLLM^(i)[:, j, :]                         (eq.7)
    g_j^(i)       = sigmoid( W_gate^(i) @ [s_broadcast; F_geo[:, j]] ) (eq.3)
    F_fused^(i)_j = g_j^(i) * (W_s^(i) F_geo[:, j])
                   + (1 - g_j^(i)) * (W_g^(i) F_geo[:, j])             (eq.4)
  The fused tokens F_fused^(i) are appended to the joint K/V at layer i so
  the action-expert (suffix) queries can attend over enriched context
  H_cond^(i) = [H_MLLM^(i); F_fused^(i)]                                 (eq.9)
  exactly as in eq.10 of the paper (pi-style layer-wise integration).

Notes
-----
- F_fused tokens are recomputed each layer; their layer outputs are discarded
  (only prefix/MLLM and suffix/action-expert states carry forward).
- We project F_fused through the layer's own input_layernorm + q/k/v/o so we
  reuse the existing per-layer attention parameters and don't introduce a new
  attention sub-block. With fusion_zero_init=True, the fusion projections are
  zero-initialized so the model starts behaving like vanilla pi0.5.
"""

from __future__ import annotations

import builtins
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)
from lerobot.utils.import_utils import _transformers_available, require_package

from ..pi05.modeling_pi05 import (
    ActionSelectKwargs,
    PI05Policy,
    PI05Pytorch,
    make_att_2d_masks,
    pad_vector,
    resize_with_pad_torch,
)
from ..pretrained import T
from .configuration_pi05_vggt_3d_mix import PI05VGGT3DMixConfig

if TYPE_CHECKING or _transformers_available:
    from transformers.models.gemma import modeling_gemma

    from ..pi_gemma import _gated_residual, layernorm_forward
else:
    modeling_gemma = None
    _gated_residual = None
    layernorm_forward = None


# ---------------------------------------------------------------------------
# VGGT encoder wrapper
# ---------------------------------------------------------------------------


class VGGTAggregatorEncoder(nn.Module):
    """Frozen VGGT aggregator that returns last-iteration patch tokens.

    The full VGGT model has heads for camera/depth/pointmap/track that we don't
    need; we only run the aggregator and skip everything else.
    """

    def __init__(self, pretrained_name: str, freeze: bool = True):
        super().__init__()
        try:
            from vggt.models.vggt import VGGT
        except ImportError as e:
            raise ImportError(
                "VGGT is required for the pi05_vggt_3d_mix policy. Install it with:\n"
                "    pip install git+https://github.com/facebookresearch/vggt.git\n"
                f"(original error: {e})"
            )

        full = VGGT.from_pretrained(pretrained_name)
        # Only keep the aggregator; drop heads to save memory.
        self.aggregator = full.aggregator
        self.patch_size = self.aggregator.patch_size
        # 2 * embed_dim because aggregator concatenates frame+global intermediates.
        self.feature_dim = 2 * self.aggregator.patch_embed.embed_dim if hasattr(
            self.aggregator.patch_embed, "embed_dim"
        ) else 2 * 1024

        if freeze:
            self.aggregator.eval()
            for p in self.aggregator.parameters():
                p.requires_grad = False
        self._freeze = freeze

    def train(self, mode: bool = True):  # noqa: D401 - keep frozen even in train()
        super().train(mode)
        if self._freeze:
            self.aggregator.eval()
        return self

    @torch.no_grad()
    def _aggregate(self, images: Tensor) -> tuple[Tensor, int]:
        # images: [B, S, 3, H, W] in [0, 1].
        return self.aggregator(images)

    def forward(self, images: Tensor) -> Tensor:
        """Encode multi-view images to flat patch tokens.

        Args:
            images: [B, S, 3, H, W], float32, in [0, 1].

        Returns:
            patch_tokens: [B, S * N_patches, 2 * embed_dim]
        """
        if self._freeze:
            with torch.no_grad():
                outputs, patch_start = self._aggregate(images)
        else:
            outputs, patch_start = self.aggregator(images)
        # Take the last iteration's intermediate; shape [B, S, P, 2C].
        last = outputs[-1]
        # Drop camera + register tokens.
        patches = last[:, :, patch_start:, :]
        b, s, p, c = patches.shape
        return patches.reshape(b, s * p, c)


# ---------------------------------------------------------------------------
# Gated fusion module (per layer)
# ---------------------------------------------------------------------------


class GatedFusion3DMix(nn.Module):
    """Semantic-conditioned gated fusion (3D-Mix, eqs. 2-4)."""

    def __init__(self, dim: int, zero_init: bool = True):
        super().__init__()
        self.dim = dim
        self.w_gate = nn.Linear(2 * dim, dim, bias=True)
        self.w_s = nn.Linear(dim, dim, bias=False)
        self.w_g = nn.Linear(dim, dim, bias=False)
        if zero_init:
            # Start with no contribution from F_fused: W_s = W_g = 0 -> F_fused = 0,
            # gate doesn't matter. Optimizer learns to grow these.
            nn.init.zeros_(self.w_s.weight)
            nn.init.zeros_(self.w_g.weight)
            nn.init.zeros_(self.w_gate.weight)
            nn.init.zeros_(self.w_gate.bias)

    def forward(self, semantic_hidden: Tensor, f_geo: Tensor) -> Tensor:
        """Compute F_fused for one layer.

        Args:
            semantic_hidden: [B, L, D] - layer-input MLLM hidden states.
            f_geo:           [B, N, D] - shared, projected VGGT tokens.

        Returns:
            f_fused: [B, N, D]
        """
        # Mean-pool semantic tokens to a global context, then broadcast to N positions.
        s_global = semantic_hidden.mean(dim=1, keepdim=True)  # [B, 1, D]
        s_broadcast = s_global.expand(-1, f_geo.shape[1], -1)  # [B, N, D]

        gate_input = torch.cat([s_broadcast, f_geo], dim=-1)  # [B, N, 2D]
        gate = torch.sigmoid(self.w_gate(gate_input))  # [B, N, D]

        s_proj = self.w_s(s_broadcast)
        g_proj = self.w_g(f_geo)
        f_fused = gate * s_proj + (1.0 - gate) * g_proj
        return f_fused


# ---------------------------------------------------------------------------
# Layer compute with 3D-Mix injection
# ---------------------------------------------------------------------------


def _compute_layer_3dmix(
    layer_idx: int,
    inputs_embeds: list[Tensor],
    attention_mask: Tensor,
    position_ids: Tensor,
    adarms_cond: list[Tensor | None],
    paligemma,
    gemma_expert,
    f_geo: Tensor | None,
    fusion_module: GatedFusion3DMix | None,
) -> list[Tensor]:
    """Run one layer of joint paligemma+expert attention with optional fused tokens.

    This mirrors `compute_layer_complete` in `pi05.modeling_pi05` but appends
    F_fused-derived K/V (and matching Q so the per-position rotary embedding
    machinery still applies cleanly) into the joint attention. Output for the
    fused positions is discarded.
    """

    models = [paligemma.model.language_model, gemma_expert.model]
    query_states: list[Tensor] = []
    key_states: list[Tensor] = []
    value_states: list[Tensor] = []
    gates: list[Tensor | None] = []

    # --- prefix + suffix Q/K/V (identical to vanilla pi05) ---
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_states.append(layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2))
        key_states.append(layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2))
        value_states.append(layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2))

    # --- F_fused tokens for this layer (computed from prefix input states) ---
    fused_len = 0
    if fusion_module is not None and f_geo is not None and inputs_embeds[0] is not None:
        prefix_layer = models[0].layers[layer_idx]
        # Compute F_fused in the prefix's working dtype.
        f_fused = fusion_module(inputs_embeds[0], f_geo)
        if f_fused.dtype != prefix_layer.self_attn.q_proj.weight.dtype:
            f_fused = f_fused.to(dtype=prefix_layer.self_attn.q_proj.weight.dtype)
        fused_normed, _ = layernorm_forward(prefix_layer.input_layernorm, f_fused, adarms_cond[0])
        fused_shape = (*fused_normed.shape[:-1], -1, prefix_layer.self_attn.head_dim)
        q_fused = prefix_layer.self_attn.q_proj(fused_normed).view(fused_shape).transpose(1, 2)
        k_fused = prefix_layer.self_attn.k_proj(fused_normed).view(fused_shape).transpose(1, 2)
        v_fused = prefix_layer.self_attn.v_proj(fused_normed).view(fused_shape).transpose(1, 2)
        query_states.append(q_fused)
        key_states.append(k_fused)
        value_states.append(v_fused)
        fused_len = f_fused.shape[1]

    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)

    # --- positional embeddings: extend position_ids for fused tokens ---
    seq_len_total = query_states.shape[2]
    if fused_len > 0:
        # Reuse the prefix's last position id range for fused tokens. Position
        # within F_geo is intentionally synthetic; rotary just needs valid ids.
        device = position_ids.device
        prefix_last = position_ids.max(dim=1, keepdim=True).values + 1  # [B, 1]
        fused_position_ids = prefix_last + torch.arange(fused_len, device=device)[None, :]
        full_position_ids = torch.cat([position_ids, fused_position_ids], dim=1)
    else:
        full_position_ids = position_ids

    dummy = torch.zeros(
        query_states.shape[0],
        seq_len_total,
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy, full_position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )

    # --- attention mask: extend so all queries can attend to fused K/V ---
    if fused_len > 0:
        # attention_mask has shape [B, 1, L_orig, L_orig] with 0.0 / mask_value.
        b, h, lq, lk = attention_mask.shape
        new_lq = lq + fused_len
        new_lk = lk + fused_len
        full_mask = attention_mask.new_full(
            (b, h, new_lq, new_lk), fill_value=OPENPI_ATTENTION_MASK_VALUE
        )
        full_mask[:, :, :lq, :lk] = attention_mask
        # Allow original queries to attend to fused keys.
        full_mask[:, :, :lq, lk:] = 0.0
        # Allow fused queries to attend to everything (their output is dropped).
        full_mask[:, :, lq:, :] = 0.0
    else:
        full_mask = attention_mask

    scaling = paligemma.model.language_model.layers[layer_idx].self_attn.scaling
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.model.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        full_mask,
        scaling,
    )
    head_dim = paligemma.model.language_model.layers[layer_idx].self_attn.head_dim
    batch_size = query_states.shape[0]
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

    # --- finalize prefix and suffix layer outputs (drop fused) ---
    outputs_embeds: list[Tensor] = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        slice_out = att_output[:, start_pos:end_pos]
        if slice_out.dtype != layer.self_attn.o_proj.weight.dtype:
            slice_out = slice_out.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(slice_out)
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


# ---------------------------------------------------------------------------
# pi0.5 + 3D-Mix model
# ---------------------------------------------------------------------------


class PI05VGGT3DMixPytorch(PI05Pytorch):
    """PI05 with VGGT-derived 3D-Mix layer-wise gated fusion."""

    def __init__(self, config: PI05VGGT3DMixConfig, rtc_processor=None):
        # Disable any post-init compile that would freeze our forward signature
        # before we wire in the 3D-Mix path.
        compile_requested = config.compile_model
        config.compile_model = False
        try:
            super().__init__(config, rtc_processor=rtc_processor)
        finally:
            config.compile_model = compile_requested

        # ---- VGGT encoder ----
        self.vggt = VGGTAggregatorEncoder(
            config.vggt_pretrained_name, freeze=config.vggt_freeze
        )
        # Aggregator output dim (2*embed_dim) -> MLLM hidden dim.
        mllm_hidden = self.paligemma_with_expert.paligemma.config.text_config.hidden_size
        self.vggt_proj = nn.Linear(config.vggt_feature_dim, mllm_hidden, bias=True)
        if config.fusion_zero_init:
            # Don't disturb pi0.5 representation initially; W_s/W_g zero-init
            # already nulls the fused contribution, but zero proj also keeps
            # F_geo magnitude bounded early on.
            nn.init.zeros_(self.vggt_proj.weight)
            nn.init.zeros_(self.vggt_proj.bias)

        # ---- per-layer fusion modules ----
        depth = self.paligemma_with_expert.paligemma.config.text_config.num_hidden_layers
        self._fusion_layer_start = max(0, config.fusion_layer_start)
        self._fusion_layer_end = min(depth, config.fusion_layer_end)
        self.fusion_modules = nn.ModuleList(
            [
                GatedFusion3DMix(mllm_hidden, zero_init=config.fusion_zero_init)
                for _ in range(self._fusion_layer_start, self._fusion_layer_end)
            ]
        )

        self._vggt_max_tokens_per_view = config.vggt_max_tokens_per_view

        # Re-apply compile if requested (now that fusion is wired).
        if compile_requested:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    # -- helpers ---------------------------------------------------------------

    def _fusion_for_layer(self, layer_idx: int) -> GatedFusion3DMix | None:
        if self._fusion_layer_start <= layer_idx < self._fusion_layer_end:
            return self.fusion_modules[layer_idx - self._fusion_layer_start]
        return None

    def _preprocess_for_vggt(self, images: list[Tensor], img_masks: list[Tensor]) -> Tensor:
        """Stack per-camera images into the [B, S, 3, H_v, W_v] tensor VGGT expects."""
        # Pi05 already passes [B, C, H, W] in [-1, 1] (post-SigLIP normalization).
        # VGGT wants [0, 1].
        target = self.config.vggt_image_size
        stacked = []
        for img, mask in zip(images, img_masks, strict=True):
            x = (img.to(torch.float32) + 1.0) * 0.5
            x = x.clamp(0.0, 1.0)
            if x.shape[-1] != target or x.shape[-2] != target:
                x = resize_with_pad_torch(x, target, target)
            # Zero out padded cameras so they contribute no spurious geometry.
            mask = mask.to(x.dtype).view(-1, 1, 1, 1)
            x = x * mask
            stacked.append(x)
        # [num_views, B, 3, H, W] -> [B, num_views, 3, H, W]
        stacked_t = torch.stack(stacked, dim=1)
        return stacked_t

    def _compute_f_geo(self, images: list[Tensor], img_masks: list[Tensor]) -> Tensor:
        vggt_input = self._preprocess_for_vggt(images, img_masks)
        feats = self.vggt(vggt_input)  # [B, S*P, C_vggt]
        # Optional uniform-stride subsample to bound token count.
        if self._vggt_max_tokens_per_view is not None:
            num_views = vggt_input.shape[1]
            target_total = self._vggt_max_tokens_per_view * num_views
            if feats.shape[1] > target_total:
                stride = feats.shape[1] // target_total
                feats = feats[:, ::stride, :][:, :target_total, :]
        f_geo = self.vggt_proj(feats)
        # Match prefix dtype downstream.
        return f_geo

    # -- joint attention forward override --------------------------------------

    def _paligemma_expert_forward(
        self,
        prefix_embs: Tensor,
        suffix_embs: Tensor | None,
        att_2d_masks_4d: Tensor,
        position_ids: Tensor,
        adarms_cond: list[Tensor | None],
        f_geo: Tensor | None,
    ) -> tuple[list[Tensor], None]:
        """Run the joint paligemma+expert stack with per-layer 3D-Mix fusion."""
        paligemma = self.paligemma_with_expert.paligemma
        gemma_expert = self.paligemma_with_expert.gemma_expert
        models = [paligemma.model.language_model, gemma_expert.model]
        num_layers = paligemma.config.text_config.num_hidden_layers

        if prefix_embs is None or suffix_embs is None:
            # We don't currently support 3D-Mix during cache-only prefix forwards
            # (used at first inference step). Fall back to the parent path.
            return self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=(prefix_embs is not None and suffix_embs is None),
                adarms_cond=adarms_cond,
            )

        inputs_embeds = [prefix_embs, suffix_embs]
        use_grad_ckpt = (
            hasattr(gemma_expert.model, "gradient_checkpointing")
            and gemma_expert.model.gradient_checkpointing
            and self.training
        )

        for layer_idx in range(num_layers):
            fusion = self._fusion_for_layer(layer_idx)
            if use_grad_ckpt:
                inputs_embeds = torch.utils.checkpoint.checkpoint(
                    _compute_layer_3dmix,
                    layer_idx,
                    inputs_embeds,
                    att_2d_masks_4d,
                    position_ids,
                    adarms_cond,
                    paligemma,
                    gemma_expert,
                    f_geo,
                    fusion,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                inputs_embeds = _compute_layer_3dmix(
                    layer_idx,
                    inputs_embeds,
                    att_2d_masks_4d,
                    position_ids,
                    adarms_cond,
                    paligemma,
                    gemma_expert,
                    f_geo,
                    fusion,
                )

        # Final norms (mirrors PI05Pytorch.forward).
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            out_emb, _ = layernorm_forward(models[i].norm, hidden_states, adarms_cond[i])
            outputs_embeds.append(out_emb)
        return outputs_embeds, None

    # -- training forward ------------------------------------------------------

    def forward(self, images, img_masks, tokens, masks, actions, noise, time) -> Tensor:
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # 3D-Mix: compute F_geo once, share across layers.
        f_geo = self._compute_f_geo(images, img_masks)
        if f_geo.dtype != prefix_embs.dtype:
            f_geo = f_geo.to(dtype=prefix_embs.dtype)

        def fwd(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond, f_geo):
            outs, _ = self._paligemma_expert_forward(
                prefix_embs=prefix_embs,
                suffix_embs=suffix_embs,
                att_2d_masks_4d=att_2d_masks_4d,
                position_ids=position_ids,
                adarms_cond=[None, adarms_cond],
                f_geo=f_geo,
            )
            return outs[1]

        suffix_out = self._apply_checkpoint(
            fwd, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond, f_geo
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        v_t = self.action_out_proj(suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    # -- inference -------------------------------------------------------------

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Iterative flow-matching sampler with 3D-Mix fusion at each step.

        For simplicity we don't cache prefix K/V here — F_fused depends on the
        prefix hidden states at every layer, which would require also caching
        per-layer fused K/V. We just re-run the joint forward each denoising
        step with both prefix and suffix tokens present. This is the same cost
        per step as training but no cache speedup.
        """
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks
        )
        f_geo = self._compute_f_geo(images, img_masks)
        if f_geo.dtype != prefix_embs.dtype:
            f_geo = f_geo.to(dtype=prefix_embs.dtype)

        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            v_t = self._denoise_step_3dmix(
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
                f_geo=f_geo,
                x_t=x_t,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t

        return x_t

    def _denoise_step_3dmix(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        f_geo: Tensor,
        x_t: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)
        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs_local = prefix_embs.to(dtype=torch.bfloat16)
        else:
            prefix_embs_local = prefix_embs

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        outs, _ = self._paligemma_expert_forward(
            prefix_embs=prefix_embs_local,
            suffix_embs=suffix_embs,
            att_2d_masks_4d=att_2d_masks_4d,
            position_ids=position_ids,
            adarms_cond=[None, adarms_cond],
            f_geo=f_geo,
        )
        suffix_out = outs[1][:, -self.config.chunk_size :].to(torch.float32)
        return self.action_out_proj(suffix_out)


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------


class PI05VGGT3DMixPolicy(PI05Policy):
    """Policy wrapper around PI05VGGT3DMixPytorch."""

    config_class = PI05VGGT3DMixConfig
    name = "pi05_vggt_3d_mix"

    def __init__(self, config: PI05VGGT3DMixConfig, **kwargs):
        require_package("transformers", extra="pi")
        # Skip PI05Policy.__init__ (which would build a vanilla PI05Pytorch);
        # call the grandparent (PreTrainedPolicy.__init__) directly.
        from ..pretrained import PreTrainedPolicy

        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config

        self.init_rtc_processor()
        self.model = PI05VGGT3DMixPytorch(config, rtc_processor=self.rtc_processor)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        # Default strict=False because pi0.5 checkpoints lack the new VGGT and
        # fusion-module parameters; missing keys are expected.
        return super().from_pretrained(
            pretrained_name_or_path=pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            **kwargs,
        )
