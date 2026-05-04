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

"""CPU-only tests for pi05 + VGGT 3D-Mix.

Covers the parts of the policy that don't need a GPU, transformers heads, or
the VGGT backbone weights:

* factory + registry wiring (the hookpoints the multi-GPU launcher relies on
  to load this policy via its checkpoint path),
* config validation,
* the ``GatedFusion3DMix`` module — shapes, zero-init invariant, dtype.

Anything that requires actual VLA forwards (PaliGemma / Gemma expert / VGGT-1B
weights) is left for the H100 integration runs.
"""

from __future__ import annotations

import pytest

# transformers is a transitive dep of pi05's modeling module. We don't need
# the gemma weights here — just need the import chain to resolve so the
# factory can return the policy class.
pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch  # noqa: E402

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_policy_config  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05_vggt_3d_mix import PI05VGGT3DMixConfig, PI05VGGT3DMixPolicy  # noqa: E402
from lerobot.policies.pi05_vggt_3d_mix.modeling_pi05_vggt_3d_mix import (  # noqa: E402
    GatedFusion3DMix,
)


# ---------------------------------------------------------------------------
# Factory + registry wiring
# ---------------------------------------------------------------------------


def test_config_is_registered_choice():
    assert "pi05_vggt_3d_mix" in PreTrainedConfig.get_known_choices()
    assert PreTrainedConfig.get_choice_class("pi05_vggt_3d_mix") is PI05VGGT3DMixConfig


def test_make_policy_config_returns_correct_subclass():
    cfg = make_policy_config("pi05_vggt_3d_mix")
    assert isinstance(cfg, PI05VGGT3DMixConfig)
    # Inherits PI05 defaults (action/state dims, scheduler config, etc.).
    assert isinstance(cfg, PI05Config)


def test_get_policy_class_returns_3d_mix_policy():
    cls = get_policy_class("pi05_vggt_3d_mix")
    assert cls is PI05VGGT3DMixPolicy


def test_pi05_branch_still_returns_pi05_policy():
    """Sanity: the explicit ``pi05_vggt_3d_mix`` branch must not have shadowed
    the plain pi05 dispatch (they share a config superclass)."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    assert get_policy_class("pi05") is PI05Policy


def test_processor_factory_dispatch_picks_3d_mix_branch(monkeypatch):
    """``make_pre_post_processors`` must dispatch on ``PI05VGGT3DMixConfig``
    BEFORE the ``PI05Config`` branch — the latter is the parent class so a
    naive isinstance order would silently misroute. We spy on both processor
    factories and confirm only the 3d_mix one fires.
    """
    from lerobot.policies import factory as factory_mod
    from lerobot.policies.pi05 import processor_pi05 as pi05_proc_mod
    from lerobot.policies.pi05_vggt_3d_mix import (
        processor_pi05_vggt_3d_mix as pi05_vggt_proc_mod,
    )

    calls: dict[str, int] = {"pi05": 0, "3d_mix": 0}

    def _spy_pi05(*args, **kwargs):
        calls["pi05"] += 1
        return ("pi05_pre", "pi05_post")

    def _spy_3dmix(*args, **kwargs):
        calls["3d_mix"] += 1
        return ("3dmix_pre", "3dmix_post")

    monkeypatch.setattr(pi05_proc_mod, "make_pi05_pre_post_processors", _spy_pi05)
    monkeypatch.setattr(
        pi05_vggt_proc_mod, "make_pi05_vggt_3d_mix_pre_post_processors", _spy_3dmix
    )

    cfg = PI05VGGT3DMixConfig()
    out = factory_mod.make_pre_post_processors(cfg)

    assert calls == {"pi05": 0, "3d_mix": 1}, calls
    assert out == ("3dmix_pre", "3dmix_post")


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_negative_fusion_start():
    with pytest.raises(ValueError, match="Invalid fusion layer range"):
        PI05VGGT3DMixConfig(fusion_layer_start=-1, fusion_layer_end=4)


def test_config_rejects_inverted_fusion_range():
    with pytest.raises(ValueError, match="Invalid fusion layer range"):
        PI05VGGT3DMixConfig(fusion_layer_start=10, fusion_layer_end=2)


def test_config_rejects_fusion_end_beyond_paligemma_depth():
    with pytest.raises(ValueError, match="cannot exceed PaliGemma depth"):
        PI05VGGT3DMixConfig(fusion_layer_end=19)


def test_config_accepts_partial_fusion_window():
    """Fusion is allowed to skip early/late layers."""
    cfg = PI05VGGT3DMixConfig(fusion_layer_start=4, fusion_layer_end=12)
    assert cfg.fusion_layer_start == 4
    assert cfg.fusion_layer_end == 12


# ---------------------------------------------------------------------------
# GatedFusion3DMix module
# ---------------------------------------------------------------------------


def test_gated_fusion_zero_init_is_strictly_zero():
    """With ``zero_init=True`` the policy must boot up behaving like vanilla
    pi0.5. ``F_fused`` is the only path through which VGGT geometry enters the
    joint attention, so it must be exactly zero before any optimizer step.
    """
    torch.manual_seed(0)
    fusion = GatedFusion3DMix(dim=8, zero_init=True)
    semantic = torch.randn(2, 5, 8)
    f_geo = torch.randn(2, 9, 8)

    out = fusion(semantic, f_geo)

    assert out.shape == (2, 9, 8)
    assert torch.equal(out, torch.zeros_like(out))


def test_gated_fusion_non_zero_init_is_nontrivial():
    """Without zero-init, fusion must produce a non-degenerate output."""
    torch.manual_seed(0)
    fusion = GatedFusion3DMix(dim=8, zero_init=False)
    semantic = torch.randn(2, 5, 8)
    f_geo = torch.randn(2, 9, 8)

    out = fusion(semantic, f_geo)

    assert out.shape == (2, 9, 8)
    assert not torch.equal(out, torch.zeros_like(out))
    # The fusion is sigmoid-gated convex combination of two D-dim projections,
    # so values must be finite.
    assert torch.isfinite(out).all()


def test_gated_fusion_broadcasts_semantic_global():
    """Eq. 7 in the paper mean-pools semantic tokens. Two semantic inputs that
    differ only in *order* (same set of vectors) must produce identical fused
    output."""
    torch.manual_seed(123)
    fusion = GatedFusion3DMix(dim=4, zero_init=False)
    semantic = torch.randn(1, 6, 4)
    permuted = semantic[:, torch.tensor([5, 4, 3, 2, 1, 0]), :]
    f_geo = torch.randn(1, 3, 4)

    out_a = fusion(semantic, f_geo)
    out_b = fusion(permuted, f_geo)

    torch.testing.assert_close(out_a, out_b)


def test_gated_fusion_preserves_geo_token_count():
    """Fusion output length must equal F_geo token count (it does NOT include
    the semantic-side tokens)."""
    fusion = GatedFusion3DMix(dim=4, zero_init=False)
    out = fusion(torch.randn(1, 3, 4), torch.randn(1, 17, 4))
    assert out.shape[1] == 17
