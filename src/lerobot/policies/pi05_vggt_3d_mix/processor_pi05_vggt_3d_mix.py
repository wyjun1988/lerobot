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

"""Processor for the pi0.5 + VGGT 3D-Mix policy.

The pre/post-processing pipeline is identical to pi0.5: VGGT consumes the same
camera images that the SigLIP vision tower already sees, and the action/state
spaces are unchanged. We delegate to the pi0.5 factory so the tokenization /
normalization steps stay in lockstep with the upstream policy.
"""

from typing import Any

import torch

from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from ..pi05.processor_pi05 import make_pi05_pre_post_processors
from .configuration_pi05_vggt_3d_mix import PI05VGGT3DMixConfig


def make_pi05_vggt_3d_mix_pre_post_processors(
    config: PI05VGGT3DMixConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return make_pi05_pre_post_processors(config=config, dataset_stats=dataset_stats)
