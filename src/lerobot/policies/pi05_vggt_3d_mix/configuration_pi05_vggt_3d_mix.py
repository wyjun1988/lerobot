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
# See the License for the specific language governing permissions and
# limitations under the License.

"""3D-Mix configuration: pi05 + VGGT layer-wise gated fusion."""

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("pi05_vggt_3d_mix")
@dataclass
class PI05VGGT3DMixConfig(PI05Config):
    """
    Configuration for pi0.5 with VGGT-based 3D-Mix gated fusion.

    See "3D-Mix for VLA: A Plug-and-Play Module for Integrating VGGT-based 3D
    Information into Vision-Language-Action Models" (arXiv:2603.24393).

    The VGGT backbone is run once per forward pass on the camera images. Its
    aggregator patch tokens are projected to the MLLM hidden dim D and reused
    across all transformer layers as F_geo. At each fusion layer i, a per-layer
    gated fusion module computes a layer-specific F_fused^(i) from the layer's
    semantic context, and these tokens are appended to the joint-attention K/V
    so the action expert can attend to enriched (semantic + geometric) context.
    """

    # ---- VGGT encoder ----
    vggt_pretrained_name: str = "facebook/VGGT-1B"
    # VGGT-1B aggregator outputs tokens with dim 2 * embed_dim = 2048.
    vggt_feature_dim: int = 2048
    # VGGT input image size. The released checkpoint uses 518; using a smaller
    # size is supported via RoPE position interpolation but feature quality
    # degrades. 224 keeps tokens and memory tractable, mirroring PaliGemma.
    vggt_image_size: int = 224
    vggt_freeze: bool = True
    # Optionally subsample (uniform stride) the VGGT patch tokens per view to
    # keep extra-token count manageable at training time. None = keep all.
    vggt_max_tokens_per_view: int | None = 256

    # ---- 3D-Mix gated fusion ----
    # Inclusive start, exclusive end. PaliGemma/Gemma-300M have depth=18 layers,
    # so the default applies fusion to every transformer layer.
    fusion_layer_start: int = 0
    fusion_layer_end: int = 18
    # Initialize the fused-token contribution to zero so the model boots up
    # behaving like vanilla pi0.5 and learns to mix in geometry over training.
    fusion_zero_init: bool = True

    # ---- LR schedule for fusion params ----
    # The 3D-Mix paper uses a 10x larger LR for the fusion modules compared to
    # the MLLM. We expose this here so optimizer setup can pick it up; lerobot's
    # optimizer infra is single-LR by default, so this is informational.
    fusion_lr_multiplier: float = 10.0

    def __post_init__(self):
        super().__post_init__()
        if self.fusion_layer_start < 0 or self.fusion_layer_end < self.fusion_layer_start:
            raise ValueError(
                f"Invalid fusion layer range: [{self.fusion_layer_start}, {self.fusion_layer_end})"
            )
        if self.fusion_layer_end > 18:
            raise ValueError(
                f"fusion_layer_end ({self.fusion_layer_end}) cannot exceed PaliGemma depth (18)"
            )
