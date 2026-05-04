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

"""Multi-container LIBERO-plus eval for hosts with EGL hard-capped at 1
device per process.

Why this exists
---------------
On managed clusters (Run:AI, NGC, k8s GPU operator) the container's NVIDIA
driver typically exposes only one EGL device per process regardless of
``CUDA_VISIBLE_DEVICES``. Multi-GPU eval through a single launcher process
(``eval_libero_plus_multi_gpu.py``) cannot work there — every shard's
robosuite EGL collides on the same EGL device 0.

The fix is to use the cluster's own job scheduler to run **N containers in
parallel**, each pinned to one GPU. Each container has its own /dev/nvidiaN
mount and its own EGL world; they don't fight each other. After they all
finish, results are merged from the shared filesystem.

Three subcommands:

* ``prepare``  — probe LIBERO suite sizes, partition (suite, task_id)
                 tuples across N containers, write plan files to
                 ``<output-dir>/plans/plan_K.json``. Run this ONCE on any
                 host that has LIBERO importable.

* ``run``      — inside one container: read ``plan_<rank>.json``, load the
                 policy, iterate the plan suites, write
                 ``<output-dir>/shard_<rank>/eval_info.json``. The cluster's
                 job spec calls this with ``--shard-rank=$JOB_INDEX``.

* ``merge``    — after all containers finish: walk
                 ``<output-dir>/shard_*/eval_info.json``, re-aggregate
                 per-episode booleans, write merged report to
                 ``<output-dir>/eval_info.json``. Same logic the multi-GPU
                 launcher uses, just looking at a different directory shape.

Typical use
-----------
::

    # 1. On any node with LIBERO on PYTHONPATH:
    python scripts/eval_libero_plus_multi_container.py prepare \\
        --num-shards 4 \\
        --suites libero_spatial,libero_object,libero_goal,libero_10 \\
        --output-dir /shared/eval/3dmix_full

    # 2. Submit 4 container jobs to your cluster (Run:AI / k8s / Slurm).
    #    Each job runs (with rank substituted from the job array index):
    python scripts/eval_libero_plus_multi_container.py run \\
        --shard-rank=$JOB_INDEX --num-shards=4 \\
        --output-dir /shared/eval/3dmix_full \\
        --policy-path /path/to/checkpoint \\
        --n-episodes 10 --batch-size 1

    # 3. After all jobs complete (any one container can run this):
    python scripts/eval_libero_plus_multi_container.py merge \\
        --output-dir /shared/eval/3dmix_full
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("multi_container")


# ---------------------------------------------------------------------------
# LIBERO-plus perturbation categorization
# ---------------------------------------------------------------------------
#
# LIBERO-plus encodes each variant's perturbation in the task filename.
# Patterns are matched in order; first hit wins. The fork we evaluate against
# defines five axes:
#
#   camera         — _view_*   (alternate camera viewpoints)
#   background     — _tb_N / _table_N  (table textures / scene background)
#   lighting       — _light_*  (lighting changes)
#   object_added   — _add_*    (extra distractor objects, layout in libero_newobj/)
#   object_layout  — _level*   (alternate object placements, also in libero_newobj/)
#
# Tasks without any of these suffixes are unperturbed bases ("clean").
#
# Note: the published LIBERO-plus paper lists seven axes including language
# paraphrases, robot initial states, and sensor noise — those aren't in this
# fork's task filenames (probably injected at runtime in other forks). If your
# fork adds more axes, append patterns here and the merge will pick them up
# automatically; tasks with new suffixes will simply land in "clean" until
# their pattern is registered.
#
# Cross-reference: src/lerobot/envs/libero.py:64 has a similar regex used to
# strip these suffixes when locating the shared init-state file on disk.
PERTURBATION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"_view_"), "camera"),
    (re.compile(r"_(?:tb|table)_\d+"), "background"),
    (re.compile(r"_light_"), "lighting"),
    (re.compile(r"_add_"), "object_added"),
    (re.compile(r"_level\d*"), "object_layout"),
]


def categorize_task(task_name: str) -> str:
    """Map a LIBERO-plus task filename to its perturbation category.

    Returns one of the five category strings or ``"clean"`` when no known
    perturbation suffix is present (the unperturbed base variant).
    """
    for pat, cat in PERTURBATION_PATTERNS:
        if pat.search(task_name):
            return cat
    return "clean"


def _attach_perturbation_categories(per_task: list[dict]) -> bool:
    """Look up each task's name from LIBERO's benchmark dict and tag with
    perturbation category. Returns True on success, False if LIBERO can't
    be imported (in which case per_task is left untouched and the merge
    silently skips the per-category breakdown).
    """
    try:
        from libero.libero import benchmark  # noqa: PLC0415
    except ImportError:
        log.warning(
            "LIBERO not importable — skipping perturbation-category breakdown. "
            "Set PYTHONPATH to include LIBERO-plus to enable it."
        )
        return False

    bench_dict = benchmark.get_benchmark_dict()
    suite_tasks: dict[str, list] = {}
    for t in per_task:
        suite = t["task_group"]
        if suite not in suite_tasks:
            try:
                suite_tasks[suite] = bench_dict[suite]().tasks
            except KeyError:
                log.warning("Suite %s not in LIBERO benchmark dict; skipping.", suite)
                suite_tasks[suite] = []

    for t in per_task:
        tasks = suite_tasks.get(t["task_group"], [])
        tid = t.get("task_id")
        if tid is None or tid >= len(tasks):
            t.setdefault("task_name", None)
            t.setdefault("perturbation", "unknown")
            continue
        # `name` is what LIBERO-plus uses; some forks expose the BDDL filename
        # via `bddl_file` / `init_states_file` instead. Prefer name, fall back.
        name = getattr(tasks[tid], "name", None) or getattr(
            tasks[tid], "bddl_file", ""
        )
        t["task_name"] = name
        t["perturbation"] = categorize_task(name)
    return True


def _aggregate_by_perturbation(per_task: list[dict]) -> dict:
    """Group per-task results by perturbation category and compute pc_success
    over per-episode booleans (NOT mean-of-task-means — same reason the
    overall report does it: avoids biasing categories with unequal task counts).
    """
    by_cat_succ: dict[str, list[bool]] = defaultdict(list)
    by_cat_sum: dict[str, list[float]] = defaultdict(list)
    by_cat_max: dict[str, list[float]] = defaultdict(list)
    for t in per_task:
        cat = t.get("perturbation", "unknown")
        m = t.get("metrics", {})
        by_cat_succ[cat].extend(m.get("successes", []))
        by_cat_sum[cat].extend(m.get("sum_rewards", []))
        by_cat_max[cat].extend(m.get("max_rewards", []))

    def _avg(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    out: dict[str, dict] = {}
    for cat, succ in by_cat_succ.items():
        out[cat] = {
            "n_episodes": len(succ),
            "pc_success": 100.0 * sum(succ) / len(succ) if succ else float("nan"),
            "avg_sum_reward": _avg(by_cat_sum[cat]),
            "avg_max_reward": _avg(by_cat_max[cat]),
        }
    return out


# ---------------------------------------------------------------------------
# prepare: partition (suite, task_id) work across N shards
# ---------------------------------------------------------------------------


def _probe_suite_sizes(suites: list[str]) -> dict[str, int]:
    from libero.libero import benchmark  # noqa: PLC0415

    bench = benchmark.get_benchmark_dict()
    return {s: len(bench[s]().tasks) for s in suites}


def _parse_task_ids(spec: str | None) -> dict[str, list[int]] | None:
    """Same grammar as the multi-GPU launcher: flat ``"0,1,2"`` or per-suite
    ``"libero_spatial=0,1;libero_10=3,4"``. ``None`` means use all tasks.
    """
    if not spec:
        return None
    if "=" not in spec:
        return {"__default__": [int(x) for x in spec.split(",") if x.strip()]}
    out: dict[str, list[int]] = {}
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        suite, ids = chunk.split("=", 1)
        out[suite.strip()] = [int(x) for x in ids.split(",") if x.strip()]
    return out


def _resolve_task_ids(
    suite: str, sizes: dict[str, int], spec: dict[str, list[int]] | None
) -> list[int]:
    if spec and suite in spec:
        return list(spec[suite])
    if spec and "__default__" in spec:
        return list(spec["__default__"])
    return list(range(sizes[suite]))


def cmd_prepare(args: argparse.Namespace) -> int:
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    log.info("Probing LIBERO suite sizes for %s ...", suites)
    sizes = _probe_suite_sizes(suites)
    log.info("Suite sizes: %s", sizes)

    spec = _parse_task_ids(args.task_ids)
    work: list[tuple[str, int]] = []
    for s in suites:
        for tid in _resolve_task_ids(s, sizes, spec):
            work.append((s, tid))
    log.info("Total work units: %d", len(work))

    # Round-robin so every shard gets a roughly even mix of suites — keeps
    # wall time balanced even when suite costs differ (libero_10's 520
    # max-steps vs libero_spatial's 280).
    buckets: list[dict[str, list[int]]] = [defaultdict(list) for _ in range(args.num_shards)]
    for i, (s, tid) in enumerate(work):
        buckets[i % args.num_shards][s].append(tid)

    plans_dir = args.output_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    for k, bucket in enumerate(buckets):
        plan = sorted(((s, sorted(ids)) for s, ids in bucket.items()), key=lambda x: x[0])
        plan_path = plans_dir / f"plan_{k:02d}.json"
        plan_path.write_text(json.dumps(plan, indent=2))
        n = sum(len(ids) for _, ids in plan)
        log.info(
            "  plan_%02d.json: %d tasks across suites %s",
            k,
            n,
            [s for s, _ in plan],
        )
    log.info("Plans written to %s", plans_dir)
    log.info(
        "Submit %d container jobs that run, with $RANK substituted 0..%d:",
        args.num_shards,
        args.num_shards - 1,
    )
    log.info(
        "  python scripts/eval_libero_plus_multi_container.py run "
        "--shard-rank=$RANK --num-shards=%d --output-dir=%s "
        "--policy-path=<ckpt> --n-episodes=<N>",
        args.num_shards,
        args.output_dir,
    )
    return 0


# ---------------------------------------------------------------------------
# run: in-container single-GPU eval against this shard's plan
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    plan_path = args.output_dir / "plans" / f"plan_{args.shard_rank:02d}.json"
    if not plan_path.exists():
        log.error(
            "Plan file %s not found. Run `prepare` first to generate plans, "
            "then re-run this container.",
            plan_path,
        )
        return 2

    shard_dir = args.output_dir / f"shard_{args.shard_rank:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Hand off to the existing shard worker — already does load-once-per-shard
    # plus per-suite iteration plus eval_info.json writing. No reason to
    # duplicate that logic here.
    import subprocess  # noqa: PLC0415

    worker = Path(__file__).resolve().parent / "eval_libero_plus_shard_worker.py"
    cmd = [
        sys.executable,
        str(worker),
        f"--plan-file={plan_path}",
        f"--output-dir={shard_dir}",
        f"--policy-path={args.policy_path}",
        f"--device={args.device}",
        f"--n-episodes={args.n_episodes}",
        f"--batch-size={args.batch_size}",
        f"--seed={args.seed}",
        f"--max-episodes-rendered={args.max_episodes_rendered}",
    ]
    if args.use_amp:
        cmd.append("--use-amp")
    if args.rename_map is not None:
        cmd.append(f"--rename-map={args.rename_map}")
    log.info("Launching worker for shard %d: %s", args.shard_rank, " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


# ---------------------------------------------------------------------------
# merge: re-aggregate from shared FS
# ---------------------------------------------------------------------------


def cmd_merge(args: argparse.Namespace) -> int:
    """Re-aggregate per-shard ``eval_info.json`` files from the shared FS.

    Walks ``<output-dir>/shard_*/eval_info.json``. Each file was written by
    the shard worker with ``per_task`` (per-episode booleans) and
    ``suite_runs`` (per-suite timing). We recompute ``pc_success`` from the
    booleans rather than averaging shard means — that's the only way to stay
    unbiased when shards cover unequal task counts (which is normal under
    round-robin partitioning).
    """
    per_task: list[dict] = []
    shards_summary: list[dict] = []

    shard_dirs = sorted(args.output_dir.glob("shard_*"))
    if not shard_dirs:
        log.error("No shard_*/ directories under %s — nothing to merge.", args.output_dir)
        return 2

    for shard_dir in shard_dirs:
        ei = shard_dir / "eval_info.json"
        if not ei.exists():
            log.warning(
                "%s missing — that container may have crashed or not finished yet. "
                "Including only shards that have written eval_info.json.",
                ei,
            )
            continue
        data = json.loads(ei.read_text())
        for task in data.get("per_task", []):
            per_task.append({**task, "shard": shard_dir.name})
        for run in data.get("suite_runs", []):
            shards_summary.append(
                {
                    "shard": shard_dir.name,
                    "suite": run.get("suite"),
                    "overall": run.get("overall") or {},
                    "elapsed_s": run.get("elapsed_s"),
                }
            )

    by_g_succ: dict[str, list[bool]] = defaultdict(list)
    by_g_sum: dict[str, list[float]] = defaultdict(list)
    by_g_max: dict[str, list[float]] = defaultdict(list)
    for t in per_task:
        g = t["task_group"]
        m = t["metrics"]
        by_g_succ[g].extend(m.get("successes", []))
        by_g_sum[g].extend(m.get("sum_rewards", []))
        by_g_max[g].extend(m.get("max_rewards", []))

    def _avg(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    per_group: dict[str, dict] = {}
    overall_s: list[bool] = []
    overall_sum: list[float] = []
    overall_max: list[float] = []
    for g, succ in by_g_succ.items():
        per_group[g] = {
            "n_episodes": len(succ),
            "pc_success": 100.0 * sum(succ) / len(succ) if succ else float("nan"),
            "avg_sum_reward": _avg(by_g_sum[g]),
            "avg_max_reward": _avg(by_g_max[g]),
        }
        overall_s.extend(succ)
        overall_sum.extend(by_g_sum[g])
        overall_max.extend(by_g_max[g])

    overall = {
        "n_episodes": len(overall_s),
        "pc_success": 100.0 * sum(overall_s) / len(overall_s) if overall_s else float("nan"),
        "avg_sum_reward": _avg(overall_sum),
        "avg_max_reward": _avg(overall_max),
    }

    # LIBERO-plus paper reports per-perturbation breakdown (camera, language,
    # lighting, background, object_added, object_layout, robot_init,
    # sensor_noise). The category for each task is encoded in its name; we
    # look that up from LIBERO's benchmark dict and aggregate alongside the
    # per-suite breakdown so the final report has both views.
    per_perturbation: dict[str, dict] = {}
    if not args.skip_perturbation_breakdown:
        if _attach_perturbation_categories(per_task):
            per_perturbation = _aggregate_by_perturbation(per_task)

    merged = {
        "overall": overall,
        "per_group": per_group,
        "per_perturbation": per_perturbation,
        "per_task": per_task,
        "shards": shards_summary,
    }

    out_path = args.output_dir / "eval_info.json"
    out_path.write_text(json.dumps(merged, indent=2))
    log.info("Wrote merged report -> %s", out_path)
    log.info("Overall: %s", overall)
    log.info("Per suite (LIBERO suites):")
    for g, agg in per_group.items():
        log.info("  %s: %s", g, agg)
    if per_perturbation:
        log.info("Per perturbation category (LIBERO-plus axes):")
        for cat, agg in sorted(per_perturbation.items()):
            log.info("  %s: %s", cat, agg)
    return 0


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    p = argparse.ArgumentParser(
        prog="eval_libero_plus_multi_container.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="Probe suites + write per-container plan files.")
    pp.add_argument("--num-shards", type=int, required=True, help="Number of containers / shards.")
    pp.add_argument(
        "--suites",
        default="libero_spatial,libero_object,libero_goal,libero_10",
        help="Comma-separated LIBERO suite names.",
    )
    pp.add_argument(
        "--task-ids",
        default=None,
        help='Restrict to specific task ids — flat ("0,1,2") or per-suite '
        '("libero_spatial=0,1;libero_10=3,4"). Default: all tasks.',
    )
    pp.add_argument("--output-dir", type=Path, required=True)

    pr = sub.add_parser("run", help="Run this container's shard against its plan.")
    pr.add_argument("--shard-rank", type=int, required=True)
    pr.add_argument("--num-shards", type=int, required=True)
    pr.add_argument("--output-dir", type=Path, required=True)
    pr.add_argument("--policy-path", required=True)
    pr.add_argument("--device", default="cuda")
    pr.add_argument("--use-amp", action="store_true")
    pr.add_argument("--seed", type=int, default=1000)
    pr.add_argument("--n-episodes", type=int, default=10)
    pr.add_argument("--batch-size", type=int, default=1)
    pr.add_argument("--max-episodes-rendered", type=int, default=0)
    pr.add_argument("--rename-map", default=None)

    pm = sub.add_parser("merge", help="Aggregate per-shard outputs into one report.")
    pm.add_argument("--output-dir", type=Path, required=True)
    pm.add_argument(
        "--skip-perturbation-breakdown",
        action="store_true",
        help="Don't compute the LIBERO-plus per-perturbation aggregation. "
        "Default: compute it whenever LIBERO is importable (set PYTHONPATH "
        "to include LIBERO-plus). The breakdown groups tasks by the 5 "
        "perturbation axes this fork supports (camera, background, lighting, "
        "object_added, object_layout) plus 'clean' for unperturbed bases.",
    )

    args = p.parse_args(argv)
    if args.cmd == "prepare":
        return cmd_prepare(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "merge":
        return cmd_merge(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
