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

"""Pure-Python tests for the multi-GPU LIBERO-plus launcher.

The launcher script lives under top-level ``scripts/`` (it is not part of the
installable ``lerobot`` package), so we load it via importlib + file path.
These tests cover only its planning / arg-translation / merging logic — the
parts that don't need MuJoCo, CUDA, or any heavy lerobot deps.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

LAUNCHER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "eval_libero_plus_multi_gpu.py"


@pytest.fixture(scope="module")
def launcher():
    spec = importlib.util.spec_from_file_location("eval_libero_plus_multi_gpu", LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# parse_task_ids_arg
# ---------------------------------------------------------------------------


def test_parse_task_ids_none(launcher):
    assert launcher.parse_task_ids_arg(None) is None
    assert launcher.parse_task_ids_arg("") is None


def test_parse_task_ids_flat(launcher):
    assert launcher.parse_task_ids_arg("0,1,2") == {"__default__": [0, 1, 2]}


def test_parse_task_ids_per_suite(launcher):
    out = launcher.parse_task_ids_arg("libero_spatial=0,1;libero_10=3,4,5")
    assert out == {"libero_spatial": [0, 1], "libero_10": [3, 4, 5]}


# ---------------------------------------------------------------------------
# make_plan
# ---------------------------------------------------------------------------


def test_make_plan_task_round_robin_covers_all_units(launcher):
    """Round-robin must place every (suite, task_id) tuple onto exactly one shard."""
    suites = ["libero_spatial", "libero_object", "libero_10"]
    counts = {"libero_spatial": 5, "libero_object": 3, "libero_10": 4}
    plan = launcher.make_plan(suites, counts, task_ids=None, num_shards=4, shard_by="task")

    assert len(plan) == 4
    seen: set[tuple[str, int]] = set()
    for shard in plan:
        for suite, ids in shard:
            for tid in ids:
                key = (suite, tid)
                assert key not in seen, f"{key} assigned to multiple shards"
                seen.add(key)
    expected = {(s, tid) for s in suites for tid in range(counts[s])}
    assert seen == expected


def test_make_plan_task_balanced_shard_sizes(launcher):
    """Round-robin should keep shard sizes within ±1 of each other."""
    suites = ["a", "b", "c", "d"]
    counts = {s: 7 for s in suites}
    plan = launcher.make_plan(suites, counts, task_ids=None, num_shards=3, shard_by="task")

    sizes = [sum(len(ids) for _, ids in shard) for shard in plan]
    assert max(sizes) - min(sizes) <= 1


def test_make_plan_suite_assigns_whole_suites(launcher):
    """`shard_by='suite'` should never split a single suite across shards."""
    suites = ["libero_spatial", "libero_object", "libero_10", "libero_goal"]
    counts = {"libero_spatial": 10, "libero_object": 8, "libero_10": 6, "libero_goal": 4}
    plan = launcher.make_plan(suites, counts, task_ids=None, num_shards=4, shard_by="suite")

    suite_to_shard: dict[str, int] = {}
    for shard_idx, shard in enumerate(plan):
        for suite, _ in shard:
            assert suite not in suite_to_shard, f"{suite} split across shards"
            suite_to_shard[suite] = shard_idx
    assert set(suite_to_shard) == set(suites)


def test_make_plan_unknown_strategy_errors(launcher):
    with pytest.raises(ValueError, match="Unknown --shard-by"):
        launcher.make_plan(["a"], {"a": 1}, None, 1, shard_by="bogus")


def test_make_plan_respects_explicit_task_ids(launcher):
    suites = ["libero_spatial", "libero_10"]
    counts = {"libero_spatial": 10, "libero_10": 10}
    task_ids = {"libero_spatial": [0, 1], "libero_10": [5]}
    plan = launcher.make_plan(suites, counts, task_ids=task_ids, num_shards=2, shard_by="task")

    flat: list[tuple[str, int]] = []
    for shard in plan:
        for suite, ids in shard:
            for tid in ids:
                flat.append((suite, tid))
    assert sorted(flat) == [("libero_10", 5), ("libero_spatial", 0), ("libero_spatial", 1)]


# ---------------------------------------------------------------------------
# _translate_to_worker_args
# ---------------------------------------------------------------------------


def test_translate_known_kv_args(launcher):
    forwarded = ["--policy.path=/checkpoints/foo", "--eval.n_episodes=5", "--eval.batch_size=2"]
    out, unknown = launcher._translate_to_worker_args(forwarded)
    assert unknown == []
    assert "--policy-path=/checkpoints/foo" in out
    assert "--n-episodes=5" in out
    assert "--batch-size=2" in out


def test_translate_bool_true_emits_flag(launcher):
    out, unknown = launcher._translate_to_worker_args(["--policy.use_amp=true"])
    assert unknown == []
    assert out == ["--use-amp"]


def test_translate_bool_false_omits_flag(launcher):
    out, unknown = launcher._translate_to_worker_args(["--policy.use_amp=false"])
    assert unknown == []
    assert out == []  # `argparse store_true` defaults to False; nothing to forward.


def test_translate_max_episodes_rendered(launcher):
    """Forwarding ``--eval.max_episodes_rendered=N`` must produce
    ``--max-episodes-rendered=N`` on the worker so videos actually get saved.
    """
    out, unknown = launcher._translate_to_worker_args(["--eval.max_episodes_rendered=2"])
    assert unknown == []
    assert "--max-episodes-rendered=2" in out


def test_translate_unknown_args_are_reported(launcher):
    forwarded = ["--policy.path=/x", "--policy.num_inference_steps=5", "--eval.n_episodes=1"]
    out, unknown = launcher._translate_to_worker_args(forwarded)
    assert unknown == ["--policy.num_inference_steps=5"]
    # Known args still come through.
    assert "--policy-path=/x" in out
    assert "--n-episodes=1" in out


# ---------------------------------------------------------------------------
# _shard_env (CUDA + EGL pinning)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gpu_id", ["0", "1", "3", "7"])
def test_shard_env_pins_cuda_to_global_id_and_egl_to_zero(launcher, gpu_id):
    """Regression test for the EGL pinning bug.

    On NVIDIA, ``CUDA_VISIBLE_DEVICES=N`` filters at the driver level — the
    shard sees one card, indexed locally as 0. ``MUJOCO_EGL_DEVICE_ID`` must
    therefore be ``"0"`` inside the shard, not the global GPU id; otherwise
    MuJoCo errors with "must be an integer between 0 and 0 (inclusive)" on
    every shard except the one that happens to map to GPU 0.
    """
    env = launcher._shard_env(gpu_id)
    assert env["CUDA_VISIBLE_DEVICES"] == gpu_id
    assert env["MUJOCO_EGL_DEVICE_ID"] == "0"
    assert env["EGL_DEVICE_ID"] == "0"
    assert env["MUJOCO_GL"] == "egl"


# ---------------------------------------------------------------------------
# merge_results
# ---------------------------------------------------------------------------


def _write_worker_shard(root: Path, name: str, per_task: list[dict]) -> None:
    """Write a synthetic worker-mode shard layout under ``root/name/``."""
    shard_dir = root / name
    shard_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "overall": {},
        "per_group": {},
        "per_task": per_task,
        "suite_runs": [
            {"suite": t["task_group"], "elapsed_s": 1.0, "overall": {}} for t in per_task
        ],
    }
    (shard_dir / "eval_info.json").write_text(json.dumps(payload))


def test_merge_results_unbiased_pc_success(launcher, tmp_path):
    """``merge_results`` recomputes pc_success from per-episode booleans, not by
    averaging shard means. With unequal shard task counts, this matters: a naive
    mean would overweight the shard with fewer tasks.
    """
    # Shard A: 2 tasks in libero_spatial, 100% success on each (4 episodes).
    _write_worker_shard(
        tmp_path,
        "shard_00_gpu_0",
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
    # Shard B: 1 task in libero_spatial, 0% success (2 episodes).
    _write_worker_shard(
        tmp_path,
        "shard_01_gpu_1",
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

    merged = launcher.merge_results(tmp_path)

    # 4 success / 6 episodes = 66.66...%; per-shard-mean would give (100+0)/2=50%.
    assert merged["overall"]["n_episodes"] == 6
    assert merged["overall"]["pc_success"] == pytest.approx(100.0 * 4 / 6)
    assert merged["per_group"]["libero_spatial"]["n_episodes"] == 6
    assert merged["per_group"]["libero_spatial"]["pc_success"] == pytest.approx(100.0 * 4 / 6)
    # Per-task list should carry shard provenance.
    shards_seen = {t["shard"] for t in merged["per_task"]}
    assert shards_seen == {"shard_00_gpu_0", "shard_01_gpu_1"}


def test_merge_results_handles_empty_dir(launcher, tmp_path):
    merged = launcher.merge_results(tmp_path)
    assert merged["overall"]["n_episodes"] == 0
    assert merged["per_group"] == {}
    assert merged["per_task"] == []
