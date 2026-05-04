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

"""Tests for the multi-container LIBERO-plus eval helper.

Covers the parts that don't need LIBERO / MuJoCo / torch — partition logic
in ``prepare`` and merge math in ``merge``. The actual ``run`` step shells
out to the worker, which is exercised by other tests.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "eval_libero_plus_multi_container.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("eval_libero_plus_multi_container", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# task-id parsing
# ---------------------------------------------------------------------------


def test_parse_task_ids_none(mod):
    assert mod._parse_task_ids(None) is None
    assert mod._parse_task_ids("") is None


def test_parse_task_ids_flat(mod):
    assert mod._parse_task_ids("0,1,2") == {"__default__": [0, 1, 2]}


def test_parse_task_ids_per_suite(mod):
    out = mod._parse_task_ids("libero_spatial=0,1;libero_10=3,4,5")
    assert out == {"libero_spatial": [0, 1], "libero_10": [3, 4, 5]}


def test_resolve_task_ids_uses_per_suite_first(mod):
    sizes = {"a": 10}
    spec = {"a": [0, 1], "__default__": [9]}
    assert mod._resolve_task_ids("a", sizes, spec) == [0, 1]


def test_resolve_task_ids_falls_back_to_default(mod):
    sizes = {"a": 10}
    spec = {"__default__": [3, 4]}
    assert mod._resolve_task_ids("a", sizes, spec) == [3, 4]


def test_resolve_task_ids_no_spec_uses_all(mod):
    sizes = {"a": 5}
    assert mod._resolve_task_ids("a", sizes, None) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# cmd_prepare partition logic (run end-to-end via monkeypatch on suite probe)
# ---------------------------------------------------------------------------


def test_prepare_writes_plans_and_partitions_round_robin(mod, tmp_path, monkeypatch):
    """``prepare`` must produce N plan files, every (suite, task_id) tuple
    appears in exactly one plan, and shard sizes are within ±1.
    """
    suites = ["libero_spatial", "libero_object", "libero_10"]
    sizes = {"libero_spatial": 5, "libero_object": 3, "libero_10": 4}

    monkeypatch.setattr(mod, "_probe_suite_sizes", lambda _: sizes)

    import argparse

    args = argparse.Namespace(
        num_shards=4,
        suites=",".join(suites),
        task_ids=None,
        output_dir=tmp_path,
    )
    rc = mod.cmd_prepare(args)
    assert rc == 0

    # Every (suite, tid) must appear in exactly one plan.
    seen: set[tuple[str, int]] = set()
    plan_sizes: list[int] = []
    for k in range(4):
        plan_path = tmp_path / "plans" / f"plan_{k:02d}.json"
        assert plan_path.exists(), plan_path
        plan = json.loads(plan_path.read_text())
        n = 0
        for suite, ids in plan:
            for tid in ids:
                key = (suite, tid)
                assert key not in seen, f"{key} assigned to multiple shards"
                seen.add(key)
                n += 1
        plan_sizes.append(n)

    expected = {(s, tid) for s in suites for tid in range(sizes[s])}
    assert seen == expected
    assert max(plan_sizes) - min(plan_sizes) <= 1, plan_sizes


def test_prepare_respects_explicit_task_ids(mod, tmp_path, monkeypatch):
    suites = ["libero_spatial", "libero_10"]
    sizes = {"libero_spatial": 100, "libero_10": 100}
    monkeypatch.setattr(mod, "_probe_suite_sizes", lambda _: sizes)

    import argparse

    args = argparse.Namespace(
        num_shards=2,
        suites=",".join(suites),
        task_ids="libero_spatial=0,1;libero_10=5",
        output_dir=tmp_path,
    )
    assert mod.cmd_prepare(args) == 0

    # Total work units = 2 + 1 = 3, partitioned across 2 shards.
    flat: list[tuple[str, int]] = []
    for k in range(2):
        plan = json.loads((tmp_path / "plans" / f"plan_{k:02d}.json").read_text())
        for suite, ids in plan:
            for tid in ids:
                flat.append((suite, tid))
    assert sorted(flat) == [("libero_10", 5), ("libero_spatial", 0), ("libero_spatial", 1)]


# ---------------------------------------------------------------------------
# cmd_merge: unbiased pc_success across unequal shards
# ---------------------------------------------------------------------------


def _write_shard(root: Path, name: str, per_task: list[dict]) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "eval_info.json").write_text(
        json.dumps(
            {
                "overall": {},
                "per_group": {},
                "per_task": per_task,
                "suite_runs": [
                    {"suite": t["task_group"], "elapsed_s": 1.0, "overall": {}} for t in per_task
                ],
            }
        )
    )


def test_merge_recomputes_pc_success_from_episode_booleans(mod, tmp_path):
    """Same regression guard as the multi-GPU launcher's merge — when shards
    cover unequal task counts, averaging shard means is biased; we must
    aggregate raw episode booleans.
    """
    # Shard 0: 2 tasks, 100% success on each (4 episodes total).
    _write_shard(
        tmp_path,
        "shard_00",
        [
            {
                "task_group": "libero_spatial",
                "task_id": 0,
                "metrics": {
                    "successes": [True, True],
                    "sum_rewards": [1.0, 1.0],
                    "max_rewards": [1.0, 1.0],
                },
            },
            {
                "task_group": "libero_spatial",
                "task_id": 1,
                "metrics": {
                    "successes": [True, True],
                    "sum_rewards": [1.0, 1.0],
                    "max_rewards": [1.0, 1.0],
                },
            },
        ],
    )
    # Shard 1: 1 task, 0% success (2 episodes).
    _write_shard(
        tmp_path,
        "shard_01",
        [
            {
                "task_group": "libero_spatial",
                "task_id": 2,
                "metrics": {
                    "successes": [False, False],
                    "sum_rewards": [0.0, 0.0],
                    "max_rewards": [0.0, 0.0],
                },
            },
        ],
    )

    import argparse

    args = argparse.Namespace(output_dir=tmp_path, skip_perturbation_breakdown=True)
    assert mod.cmd_merge(args) == 0

    merged = json.loads((tmp_path / "eval_info.json").read_text())
    # 4 success / 6 episodes = 66.66...%; per-shard mean would give 50%.
    assert merged["overall"]["n_episodes"] == 6
    assert merged["overall"]["pc_success"] == pytest.approx(100.0 * 4 / 6)
    assert {t["shard"] for t in merged["per_task"]} == {"shard_00", "shard_01"}


def test_merge_tolerates_missing_shards(mod, tmp_path, caplog):
    """If a container died mid-run, its shard dir won't have eval_info.json.
    Merge should still produce a report from whatever IS there — useful
    when re-running ``merge`` after an admin kills one job."""
    _write_shard(
        tmp_path,
        "shard_00",
        [
            {
                "task_group": "libero_goal",
                "task_id": 0,
                "metrics": {
                    "successes": [True],
                    "sum_rewards": [1.0],
                    "max_rewards": [1.0],
                },
            },
        ],
    )
    # Shard 01 dir exists but no eval_info.json (container crashed).
    (tmp_path / "shard_01").mkdir()

    import argparse

    args = argparse.Namespace(output_dir=tmp_path, skip_perturbation_breakdown=True)
    assert mod.cmd_merge(args) == 0

    merged = json.loads((tmp_path / "eval_info.json").read_text())
    assert merged["overall"]["n_episodes"] == 1
    assert merged["overall"]["pc_success"] == pytest.approx(100.0)


def test_merge_errors_on_empty_dir(mod, tmp_path):
    import argparse

    args = argparse.Namespace(output_dir=tmp_path, skip_perturbation_breakdown=True)
    assert mod.cmd_merge(args) == 2


# ---------------------------------------------------------------------------
# Perturbation categorization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        # Base / unperturbed task
        ("KITCHEN_SCENE2_open_the_top_drawer", "clean"),
        # 7 perturbation axes from the LIBERO-plus paper:
        ("KITCHEN_SCENE_view_top_camera", "camera"),
        ("KITCHEN_SCENE_language_paraphrase_v1", "language"),
        ("KITCHEN_SCENE_light_dark", "lighting"),
        ("KITCHEN_SCENE_tb_3", "background"),
        ("KITCHEN_SCENE_table_2", "background"),
        ("KITCHEN_SCENE_add_obj_carrot", "object_added"),
        ("KITCHEN_SCENE_level1", "object_layout"),
        ("KITCHEN_SCENE_robot_init_offset", "robot_init"),
        ("KITCHEN_SCENE_noise_imgnoise_high", "sensor_noise"),
    ],
)
def test_categorize_task_recognizes_libero_plus_perturbation_axes(mod, name, expected):
    assert mod.categorize_task(name) == expected


def test_aggregate_by_perturbation_pools_episode_booleans(mod):
    """Per-perturbation aggregation must compute pc_success from the underlying
    per-episode booleans, NOT from per-task means — same unbiased logic the
    overall and per-suite reports use. With this rule, 1 success / 1 episode
    on one camera task and 0/3 on another camera task gives 25%, not 12.5%
    (which a mean-of-means would produce).
    """
    per_task = [
        {
            "task_group": "libero_spatial",
            "task_id": 0,
            "perturbation": "camera",
            "metrics": {
                "successes": [True],
                "sum_rewards": [1.0],
                "max_rewards": [1.0],
            },
        },
        {
            "task_group": "libero_spatial",
            "task_id": 1,
            "perturbation": "camera",
            "metrics": {
                "successes": [False, False, False],
                "sum_rewards": [0.0, 0.0, 0.0],
                "max_rewards": [0.0, 0.0, 0.0],
            },
        },
        {
            "task_group": "libero_spatial",
            "task_id": 2,
            "perturbation": "language",
            "metrics": {
                "successes": [True, True],
                "sum_rewards": [1.0, 1.0],
                "max_rewards": [1.0, 1.0],
            },
        },
    ]

    out = mod._aggregate_by_perturbation(per_task)

    # 1 success / 4 episodes for camera = 25.0
    assert out["camera"]["n_episodes"] == 4
    assert out["camera"]["pc_success"] == pytest.approx(25.0)

    # 2 / 2 for language = 100.0
    assert out["language"]["n_episodes"] == 2
    assert out["language"]["pc_success"] == pytest.approx(100.0)


def test_attach_perturbation_categories_skips_when_libero_missing(mod, monkeypatch):
    """If LIBERO can't be imported, ``_attach_perturbation_categories`` should
    return False and leave per_task untouched — the merge step still works,
    just without the perturbation breakdown."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "libero.libero" or name.startswith("libero"):
            raise ImportError("simulated: LIBERO not on PYTHONPATH")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    per_task = [
        {"task_group": "libero_spatial", "task_id": 0, "metrics": {"successes": [True]}}
    ]
    assert mod._attach_perturbation_categories(per_task) is False
    # per_task is untouched (no `perturbation` key was added):
    assert "perturbation" not in per_task[0]


def test_merge_with_perturbation_breakdown_via_monkeypatch(mod, tmp_path, monkeypatch):
    """End-to-end merge with category breakdown: stub LIBERO's benchmark dict
    to return tasks with synthetic names, then verify ``per_perturbation`` ends
    up in the merged report alongside ``per_group``."""
    _write_shard(
        tmp_path,
        "shard_00",
        [
            {
                "task_group": "libero_spatial",
                "task_id": 0,
                "metrics": {
                    "successes": [True, False],
                    "sum_rewards": [1.0, 0.0],
                    "max_rewards": [1.0, 0.0],
                },
            },
            {
                "task_group": "libero_spatial",
                "task_id": 1,
                "metrics": {
                    "successes": [False],
                    "sum_rewards": [0.0],
                    "max_rewards": [0.0],
                },
            },
        ],
    )

    # Stub LIBERO so task_id=0 -> camera, task_id=1 -> language.
    class _Task:
        def __init__(self, name):
            self.name = name

    class _Suite:
        tasks = [_Task("KITCHEN_view_top"), _Task("KITCHEN_language_paraphrase")]

    fake_benchmark_dict = {"libero_spatial": _Suite}

    import sys
    import types

    fake_libero = types.ModuleType("libero")
    fake_libero_libero = types.ModuleType("libero.libero")
    fake_benchmark = types.ModuleType("libero.libero.benchmark")
    fake_benchmark.get_benchmark_dict = lambda: fake_benchmark_dict
    fake_libero_libero.benchmark = fake_benchmark
    fake_libero.libero = fake_libero_libero
    monkeypatch.setitem(sys.modules, "libero", fake_libero)
    monkeypatch.setitem(sys.modules, "libero.libero", fake_libero_libero)
    monkeypatch.setitem(sys.modules, "libero.libero.benchmark", fake_benchmark)

    import argparse

    args = argparse.Namespace(output_dir=tmp_path, skip_perturbation_breakdown=False)
    assert mod.cmd_merge(args) == 0

    merged = json.loads((tmp_path / "eval_info.json").read_text())
    # per_group still works (unchanged):
    assert merged["per_group"]["libero_spatial"]["n_episodes"] == 3
    # New per_perturbation breakdown by category:
    assert "camera" in merged["per_perturbation"]
    assert "language" in merged["per_perturbation"]
    # camera = task 0 = 1/2 = 50%
    assert merged["per_perturbation"]["camera"]["n_episodes"] == 2
    assert merged["per_perturbation"]["camera"]["pc_success"] == pytest.approx(50.0)
    # language = task 1 = 0/1 = 0%
    assert merged["per_perturbation"]["language"]["n_episodes"] == 1
    assert merged["per_perturbation"]["language"]["pc_success"] == pytest.approx(0.0)
