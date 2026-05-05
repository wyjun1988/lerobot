#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Multi-GPU launcher for ``lerobot-eval`` on LIBERO / LIBERO-plus.

Why this exists
---------------
``lerobot-eval`` runs a policy through every (suite, task_id, episode) tuple on
one device. A full LIBERO-plus sweep (~10k task variants) takes several days
on a single H100. The (suite, task_id) work units are independent — there is
no shared optimizer state, no gradient sync, nothing — so the cleanest way to
cut wall time is data parallelism by *task*: split the task list across N GPUs,
run N copies of ``lerobot-eval`` in parallel, and merge their JSON outputs at
the end.

This script does exactly that. One subprocess per visible GPU, pinned with
``CUDA_VISIBLE_DEVICES``. Within a shard, suites run sequentially (since
``lerobot-eval`` accepts only one global ``--env.task_ids`` filter); across
shards, everything runs concurrently. After all shards finish, per-shard
``eval_info.json`` files are read back and re-aggregated from per-episode
booleans so the global ``pc_success`` is unbiased even when shards have
different task counts.

Usage
-----
.. code-block:: bash

    # Default: 4 standard suites × 10 tasks, all visible GPUs, 10 ep/task.
    python scripts/eval_libero_plus_multi_gpu.py \
        --output-dir outputs/eval/libero_plus_multi \
        -- \
        --policy.path=lerobot/your_policy \
        --eval.n_episodes=10 \
        --eval.batch_size=1 \
        --policy.device=cuda \
        --policy.use_amp=false

    # Pin to a subset of GPUs and override suites:
    python scripts/eval_libero_plus_multi_gpu.py \
        --gpus 0,1,4,5 \
        --suites libero_spatial,libero_object \
        --output-dir outputs/eval/libero_plus_split \
        -- \
        --policy.path=lerobot/your_policy \
        --eval.n_episodes=10

Flags before ``--`` configure the launcher; everything after ``--`` is
forwarded verbatim to each ``lerobot-eval`` shard. Don't pass
``--env.type``, ``--env.task``, ``--env.task_ids``, or ``--output_dir`` in
the forwarded args — the launcher controls those.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

DEFAULT_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
RESERVED_FORWARD_ARGS = ("--env.type", "--env.task", "--env.task_ids", "--output_dir")

# Forwarded args we actively consume in worker mode (translated to worker
# CLI flags). Anything else in the forwarded list triggers an error in
# worker mode because the worker isn't a draccus pipeline and won't accept
# arbitrary --policy.* flags. Use --use-lerobot-eval to fall back.
_WORKER_KNOWN = {
    "--policy.path": "--policy-path",
    "--policy.device": "--device",
    "--policy.use_amp": "--use-amp",  # bool: --policy.use_amp=true → --use-amp
    # Inference-time override of action-chunk execution length. Useful for
    # action-chunked policies (pi0, pi05, the pi05_vggt_3d_mix variant): a
    # smaller value means more frequent re-planning vs more open-loop.
    "--policy.n_action_steps": "--n-action-steps",
    "--eval.n_episodes": "--n-episodes",
    "--eval.batch_size": "--batch-size",
    "--eval.use_async_envs": "--use-async-envs",  # bool
    # Pass any positive integer to enable video recording; videos land in
    # <shard_dir>/videos/<suite>/. With shard_by=task, each (suite, task_id)
    # tuple lives on exactly one shard, so collecting all videos for a
    # suite means walking shard_*/videos/<suite>/.
    "--eval.max_episodes_rendered": "--max-episodes-rendered",
    "--env.control_mode": "--control-mode",
    "--env.max_parallel_tasks": "--max-parallel-tasks",
    "--env.episode_length": "--episode-length",
    "--seed": "--seed",
    "--rename_map": "--rename-map",
    "--trust_remote_code": "--trust-remote-code",  # bool
}
_WORKER_BOOL_FLAGS = {
    "--policy.use_amp",
    "--eval.use_async_envs",
    "--trust_remote_code",
}

log = logging.getLogger("eval_multi_gpu")


# ---------------------------------------------------------------------------
# GPU + suite discovery
# ---------------------------------------------------------------------------
def detect_gpus(explicit: str | None) -> list[str]:
    """Return a list of GPU IDs (as strings) to fan out across.

    Precedence: explicit ``--gpus`` flag → ``CUDA_VISIBLE_DEVICES`` env var →
    ``nvidia-smi -L``. IDs are kept as strings because that's what we feed
    back into ``CUDA_VISIBLE_DEVICES`` for each shard.
    """
    if explicit:
        ids = [s.strip() for s in explicit.split(",") if s.strip()]
        if ids and all(i.isdigit() for i in ids) and len(ids) == 1 and "," not in explicit:
            n = int(ids[0])
            if n > 0 and len(ids) == 1:
                # Allow `--gpus 4` shorthand for "first 4 GPUs".
                return [str(i) for i in range(n)]
        return ids
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return [s.strip() for s in cvd.split(",") if s.strip()]
    if not shutil.which("nvidia-smi"):
        raise RuntimeError(
            "Cannot detect GPUs: nvidia-smi not on PATH. Pass --gpus explicitly."
        )
    out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    return [str(i) for i, line in enumerate(out.splitlines()) if line.strip()]


def probe_suite_sizes(suites: list[str]) -> dict[str, int]:
    """Run a tiny child Python to ask LIBERO how many tasks each suite has.

    We shell out instead of importing here because (a) the launcher should not
    pull in MuJoCo / EGL on import, and (b) probing in a child process keeps
    any LIBERO import side-effects from leaking into the coordinator.
    """
    snippet = (
        "import json, sys\n"
        "from libero.libero import benchmark\n"
        "bench = benchmark.get_benchmark_dict()\n"
        "out = {s: len(bench[s]().tasks) for s in sys.argv[1:]}\n"
        "print(json.dumps(out))\n"
    )
    raw = subprocess.check_output([sys.executable, "-c", snippet, *suites], text=True)
    # Some LIBERO versions print banner lines on import; take the last JSON line.
    for line in reversed(raw.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"Could not parse suite sizes from probe output:\n{raw}")


# ---------------------------------------------------------------------------
# Work planning
# ---------------------------------------------------------------------------
def parse_task_ids_arg(spec: str | None) -> dict[str, list[int]] | None:
    """Parse ``--task-ids`` into a per-suite dict.

    Accepted forms:
      * ``"0,1,2"`` — same task ids for every suite
      * ``"libero_spatial=0,1;libero_10=3,4,5"`` — per-suite override
    """
    if not spec:
        return None
    if "=" not in spec:
        flat = [int(x) for x in spec.split(",") if x.strip()]
        return {"__default__": flat}
    out: dict[str, list[int]] = {}
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        suite, ids = chunk.split("=", 1)
        out[suite.strip()] = [int(x) for x in ids.split(",") if x.strip()]
    return out


def _resolve_ids(
    suite: str, task_counts: dict[str, int], task_ids: dict[str, list[int]] | None
) -> list[int]:
    if task_ids and suite in task_ids:
        return list(task_ids[suite])
    if task_ids and "__default__" in task_ids:
        return list(task_ids["__default__"])
    return list(range(task_counts[suite]))


def make_plan(
    suites: list[str],
    task_counts: dict[str, int],
    task_ids: dict[str, list[int]] | None,
    num_shards: int,
    shard_by: str = "task",
) -> list[list[tuple[str, list[int]]]]:
    """Partition (suite, task_id) work units across shards.

    Two strategies, exposed via ``shard_by``:

    ``"task"`` (default, recommended) — round-robin every (suite, task_id)
    tuple across shards. Every shard gets a roughly even mix of suites, which
    keeps wall time balanced even when suites have very different costs (e.g.
    libero_10 has 520 max-steps vs libero_spatial's 280, so a shard that runs
    only libero_10 takes ~2× as long as one that runs only libero_spatial,
    and the whole sweep is bounded by the slowest shard).

    ``"suite"`` — assign whole suites to shards (suite i → shard i % N). With
    N == #suites this is one-suite-per-GPU. Useful for A/B comparison against
    ``"task"`` and for cases where you want to measure per-suite throughput
    in isolation. Expect *worse* wall time when suite costs are uneven.
    """
    if shard_by == "task":
        work: list[tuple[str, int]] = []
        for s in suites:
            for tid in _resolve_ids(s, task_counts, task_ids):
                work.append((s, tid))
        shards: list[dict[str, list[int]]] = [defaultdict(list) for _ in range(num_shards)]
        for i, (s, tid) in enumerate(work):
            shards[i % num_shards][s].append(tid)
        return [
            sorted(((s, sorted(ids)) for s, ids in d.items()), key=lambda x: x[0])
            for d in shards
        ]

    if shard_by == "suite":
        shards_s: list[list[tuple[str, list[int]]]] = [[] for _ in range(num_shards)]
        for i, suite in enumerate(suites):
            ids = sorted(_resolve_ids(suite, task_counts, task_ids))
            shards_s[i % num_shards].append((suite, ids))
        # Sort each shard's suites alphabetically for deterministic output.
        return [sorted(plan, key=lambda x: x[0]) for plan in shards_s]

    raise ValueError(f"Unknown --shard-by: {shard_by!r} (expected 'task' or 'suite').")


# ---------------------------------------------------------------------------
# Shard execution
# ---------------------------------------------------------------------------
def _build_lerobot_eval_cmd(
    suite: str,
    task_ids: list[int],
    output_dir: Path,
    forwarded_args: list[str],
) -> list[str]:
    return [
        "lerobot-eval",
        "--env.type=libero_plus",
        f"--env.task={suite}",
        f"--env.task_ids={json.dumps(task_ids)}",
        f"--output_dir={output_dir}",
        *forwarded_args,
    ]


def _translate_to_worker_args(forwarded: list[str]) -> tuple[list[str], list[str]]:
    """Translate --foo.bar=value forwarded args into worker --foo-bar flags.

    Returns ``(worker_args, unknown_args)``. Worker mode rejects unknown
    forwarded args because the worker isn't a draccus pipeline and silently
    dropping flags would be a footgun (the user thinks ``--policy.use_amp=true``
    took effect when it didn't).
    """
    worker_args: list[str] = []
    unknown: list[str] = []
    for raw in forwarded:
        key, sep, val = raw.partition("=")
        if key not in _WORKER_KNOWN:
            unknown.append(raw)
            continue
        wkey = _WORKER_KNOWN[key]
        if key in _WORKER_BOOL_FLAGS:
            v = (val if sep else "true").strip().lower()
            if v in ("true", "1", "yes"):
                worker_args.append(wkey)
            # false → omit (argparse store_true defaults to False)
        else:
            if not sep:
                # Allow `--foo bar` style too.
                worker_args.append(wkey)
            else:
                worker_args.append(f"{wkey}={val}")
    return worker_args, unknown


def _shard_worker_path() -> Path:
    return Path(__file__).resolve().parent / "eval_libero_plus_shard_worker.py"


def _shard_env(
    gpu_id: str, all_gpus: list[str], egl_filter: bool = False
) -> dict[str, str]:
    """Per-shard environment: pin CUDA + EGL to the same physical GPU.

    Two modes, gated by ``egl_filter``:

    ``egl_filter=False`` (default, "cuda-rank" mode)
    ------------------------------------------------
    For typical NVIDIA workstations where EGL respects ``CUDA_VISIBLE_DEVICES``.
    Keeps CVD as the *full* set of selected GPUs across all shards (so MuJoCo's
    Python wrapper can membership-check ``MEDI`` against the CVD list and
    renumber internally), sets ``MEDI=$gpu_id`` (a value in CVD), and isolates
    per-shard CUDA via ``torch.cuda.set_device(rank)`` inside the worker. The
    launcher forwards ``--cuda-rank`` for that.

    ``egl_filter=True`` ("cvd-filter" mode, for managed clusters)
    -------------------------------------------------------------
    For hosts where EGL exposes only 1 device per process *regardless* of
    ``CUDA_VISIBLE_DEVICES`` — typical of managed GPU clusters / containers
    that decouple EGL device enumeration from CUDA. Even with ``CVD=0,1,2,3``,
    ``eglQueryDevicesEXT`` returns 1, so robosuite's count check rejects any
    ``MEDI > 0``. The only working layout in that case is to filter CVD to a
    single GPU per shard (so each shard's "EGL device 0" maps to that physical
    card) and set ``MEDI=0``. ``--cuda-rank`` is unnecessary here because the
    CVD filter already isolates CUDA: the shard sees ``cuda:0 == physical $gpu_id``.

    Detect the right mode for a host by running::

        from robosuite.renderers.context.egl_context import \
            create_initialized_egl_device_display
        # Try device_id=1 with CVD=0,1,2,3. If it fails with "0 and 0
        # (inclusive)", the host needs egl_filter=True.
    """
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    if egl_filter:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["MUJOCO_EGL_DEVICE_ID"] = "0"
        env["EGL_DEVICE_ID"] = "0"
    else:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(all_gpus)
        env["MUJOCO_EGL_DEVICE_ID"] = gpu_id
        env["EGL_DEVICE_ID"] = gpu_id
    return env


def _cuda_rank_in(gpu_id: str, all_gpus: list[str]) -> int:
    """Index of ``gpu_id`` in the ordered ``all_gpus`` list.

    With ``CUDA_VISIBLE_DEVICES`` set to ``",".join(all_gpus)``, PyTorch
    re-numbers physical devices to ``cuda:0..K-1`` in CVD order. The shard's
    physical card is at this position; the worker must ``set_device(rank)``
    before any CUDA op, otherwise tensors land on whichever device PyTorch
    happens to default to (typically cuda:0 = first CVD value, causing every
    shard to collide on one card).
    """
    return all_gpus.index(gpu_id)


def _shard_env_legacy(gpu_id: str) -> dict[str, str]:
    """Env for the legacy ``--use-lerobot-eval`` path: filter CVD per shard.

    The legacy path spawns ``lerobot-eval`` subprocesses that don't know about
    ``--cuda-rank``, so we can't use the worker-mode trick. Instead we filter
    CUDA per shard (the original setup) and let LIBERO/robosuite go through
    MuJoCo's Python EGL wrapper, which membership-checks ``MEDI`` against CVD
    and renumbers it internally — that works fine when CVD has a single value.
    The pre-flight ``--check-egl`` snippet (which exercises the C++ raw path
    via ``mujoco.Renderer`` and would fail under filtered CVD) is run with
    the *worker-mode* env, not this one, so the divergence is contained.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env.setdefault("MUJOCO_GL", "egl")
    env["MUJOCO_EGL_DEVICE_ID"] = gpu_id
    env["EGL_DEVICE_ID"] = gpu_id
    return env


def run_shard_worker(
    shard_idx: int,
    gpu_id: str,
    all_gpus: list[str],
    plan: list[tuple[str, list[int]]],
    worker_args: list[str],
    output_root: Path,
    dry_run: bool,
    rc_holder: list[int],
    timing_holder: list[dict] | None = None,
    egl_filter: bool = False,
) -> None:
    """Spawn one Python worker per shard. Worker loads the policy ONCE
    and iterates the shard's (suite, task_ids) plan in-process, so model
    load + GPU init + processor setup are paid once per *shard*, not once
    per suite as in the legacy ``run_shard_per_suite`` path.
    """
    shard_dir = output_root / f"shard_{shard_idx:02d}_gpu_{gpu_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_path = shard_dir / "shard.log"
    plan_file = shard_dir / "plan.json"
    plan_file.write_text(json.dumps(plan, indent=2))

    env = _shard_env(gpu_id, all_gpus, egl_filter=egl_filter)
    cmd = [
        sys.executable,
        str(_shard_worker_path()),
        f"--plan-file={plan_file}",
        f"--output-dir={shard_dir}",
        *worker_args,
    ]
    # In cuda-rank mode the worker must call torch.cuda.set_device(rank) so
    # tensors land on the right physical GPU after CVD reordering. In
    # egl-filter mode CVD is already filtered to one card, so cuda:0 IS the
    # right physical GPU; --cuda-rank would be a no-op (and asking for >0
    # would error since torch only sees 1 device).
    if not egl_filter:
        cmd.insert(4, f"--cuda-rank={_cuda_rank_in(gpu_id, all_gpus)}")

    started = time.time()
    rc = 0
    with log_path.open("w", buffering=1) as logf:
        line = (
            f"[shard {shard_idx:02d} gpu {gpu_id}] "
            f"plan={[(s, len(ids)) for s, ids in plan]}\n"
            f"  cmd: {shlex.join(cmd)}\n"
        )
        logf.write(line)
        print(line, end="", flush=True)
        if not dry_run:
            proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
            rc = proc.returncode
        elapsed = time.time() - started
        logf.write(f"[shard {shard_idx:02d} gpu {gpu_id}] finished in {elapsed:.1f}s rc={rc}\n")
        print(f"[shard {shard_idx:02d} gpu {gpu_id}] finished in {elapsed:.1f}s rc={rc}",
              flush=True)

    # Worker emits its own per-suite timing inside eval_info.json under
    # "suite_runs". Mirror at the shard-timing level for the launcher
    # summary's wall-time table.
    suites: list[dict] = []
    ei_path = shard_dir / "eval_info.json"
    if ei_path.exists():
        try:
            data = json.loads(ei_path.read_text())
            suites = data.get("suite_runs", [])
        except json.JSONDecodeError:
            pass

    timing = {
        "shard": shard_idx,
        "gpu_id": gpu_id,
        "elapsed_s": elapsed,
        "rc": rc,
        "suites": suites,
    }
    with (shard_dir / "shard_timing.json").open("w") as f:
        json.dump(timing, f, indent=2)
    rc_holder[0] = rc
    if timing_holder is not None:
        timing_holder.append(timing)


def run_shard_per_suite(
    shard_idx: int,
    gpu_id: str,
    all_gpus: list[str],
    plan: list[tuple[str, list[int]]],
    forwarded_args: list[str],
    output_root: Path,
    dry_run: bool,
    rc_holder: list[int],
    timing_holder: list[dict] | None = None,
) -> None:
    """Legacy path: one ``lerobot-eval`` subprocess per (shard, suite).

    Pays full startup cost (Python import, hub download, policy load, GPU
    init, processor setup) once per *suite*. Kept as ``--use-lerobot-eval``
    for debugging — the worker path (``run_shard_worker``) is the default.
    """
    del all_gpus  # unused in legacy path; CVD is filtered per shard instead
    shard_dir = output_root / f"shard_{shard_idx:02d}_gpu_{gpu_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_path = shard_dir / "shard.log"
    env = _shard_env_legacy(gpu_id)

    rc_total = 0
    started = time.time()
    suite_timings: list[dict] = []
    with log_path.open("w", buffering=1) as logf:

        def emit(msg: str) -> None:
            line = f"[shard {shard_idx:02d} gpu {gpu_id}] {msg}\n"
            logf.write(line)
            print(line, end="", flush=True)

        emit(f"plan: {[(s, len(ids)) for s, ids in plan]}")
        for suite, ids in plan:
            suite_dir = shard_dir / suite
            cmd = _build_lerobot_eval_cmd(suite, ids, suite_dir, forwarded_args)
            emit(f"suite={suite} n_tasks={len(ids)} ids={ids}")
            emit(f"cmd: {shlex.join(cmd)}")
            if dry_run:
                continue
            t0 = time.time()
            proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
            elapsed = time.time() - t0
            emit(f"suite={suite} done in {elapsed:.1f}s rc={proc.returncode}")
            suite_timings.append(
                {"suite": suite, "n_tasks": len(ids), "elapsed_s": elapsed, "rc": proc.returncode}
            )
            if proc.returncode != 0:
                rc_total = proc.returncode

        total_elapsed = time.time() - started
        emit(f"shard finished in {total_elapsed:.1f}s rc={rc_total}")
        with (shard_dir / "shard_timing.json").open("w") as f:
            json.dump(
                {
                    "shard": shard_idx,
                    "gpu_id": gpu_id,
                    "elapsed_s": total_elapsed,
                    "rc": rc_total,
                    "suites": suite_timings,
                },
                f,
                indent=2,
            )
    rc_holder[0] = rc_total
    if timing_holder is not None:
        timing_holder.append(
            {
                "shard": shard_idx,
                "gpu_id": gpu_id,
                "elapsed_s": total_elapsed,
                "rc": rc_total,
                "suites": suite_timings,
            }
        )


# ---------------------------------------------------------------------------
# Result merging
# ---------------------------------------------------------------------------
def merge_results(output_root: Path) -> dict:
    """Re-aggregate per-shard ``eval_info.json`` files into one global report.

    We recompute ``pc_success`` from per-episode booleans rather than averaging
    pre-aggregated shard means — that's the only way to stay unbiased when
    shards have different task counts (which is normal under round-robin).
    """
    per_task: list[dict] = []
    shards_summary: list[dict] = []

    for shard_dir in sorted(output_root.glob("shard_*")):
        # Worker mode: one eval_info.json at the shard root, covering all
        # suites the shard owns. We still record per-suite "overall" via
        # the worker's "suite_runs" field.
        shard_ei = shard_dir / "eval_info.json"
        if shard_ei.exists():
            with shard_ei.open() as f:
                data = json.load(f)
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
            continue

        # Legacy --use-lerobot-eval mode: one eval_info.json per (shard, suite).
        for suite_dir in sorted(p for p in shard_dir.iterdir() if p.is_dir()):
            ei = suite_dir / "eval_info.json"
            if not ei.exists():
                continue
            with ei.open() as f:
                data = json.load(f)
            for task in data.get("per_task", []):
                per_task.append({**task, "shard": shard_dir.name})
            shards_summary.append(
                {
                    "shard": shard_dir.name,
                    "suite": suite_dir.name,
                    "overall": data.get("overall", {}),
                }
            )

    by_group_succ: dict[str, list[bool]] = defaultdict(list)
    by_group_sum: dict[str, list[float]] = defaultdict(list)
    by_group_max: dict[str, list[float]] = defaultdict(list)
    for t in per_task:
        g = t["task_group"]
        m = t["metrics"]
        by_group_succ[g].extend(m.get("successes", []))
        by_group_sum[g].extend(m.get("sum_rewards", []))
        by_group_max[g].extend(m.get("max_rewards", []))

    def _avg(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    per_group: dict[str, dict] = {}
    overall_succ: list[bool] = []
    overall_sum: list[float] = []
    overall_max: list[float] = []
    for g, succ in by_group_succ.items():
        per_group[g] = {
            "n_episodes": len(succ),
            "pc_success": 100.0 * (sum(succ) / len(succ)) if succ else float("nan"),
            "avg_sum_reward": _avg(by_group_sum[g]),
            "avg_max_reward": _avg(by_group_max[g]),
        }
        overall_succ.extend(succ)
        overall_sum.extend(by_group_sum[g])
        overall_max.extend(by_group_max[g])

    overall = {
        "n_episodes": len(overall_succ),
        "pc_success": 100.0 * (sum(overall_succ) / len(overall_succ))
        if overall_succ
        else float("nan"),
        "avg_sum_reward": _avg(overall_sum),
        "avg_max_reward": _avg(overall_max),
    }

    return {
        "overall": overall,
        "per_group": per_group,
        "per_task": per_task,
        "shards": shards_summary,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def check_egl_per_gpu(gpus: list[str]) -> list[tuple[str, bool, str]]:
    """For each GPU, spawn a child that binds an EGL display + a tiny MuJoCo
    scene on that physical device and reports success/failure.

    Uses the same env layout as the worker mode: ``CUDA_VISIBLE_DEVICES`` is
    the *full* list of selected GPUs (so MuJoCo's Python wrapper accepts
    ``MEDI`` as a member, then renumbers internally), and ``MEDI`` is the
    global id of the GPU under test. The pre-flight binds whichever physical
    card the wrapper resolves to — same logic the real shards will use.
    """
    snippet = (
        "import os, sys, json\n"
        "os.environ.setdefault('MUJOCO_GL', 'egl')\n"
        "try:\n"
        "    import mujoco\n"
        "    # Tiny model — just enough to force EGL ctx + GPU buffer alloc.\n"
        "    xml = '<mujoco><worldbody><geom type=\"sphere\" size=\"0.1\"/>'\n"
        "    xml += '</worldbody></mujoco>'\n"
        "    m = mujoco.MjModel.from_xml_string(xml)\n"
        "    d = mujoco.MjData(m)\n"
        "    r = mujoco.Renderer(m, height=64, width=64)\n"
        "    mujoco.mj_step(m, d)\n"
        "    r.update_scene(d)\n"
        "    r.render()\n"
        "    print(json.dumps({'ok': True,\n"
        "        'cuda_visible': os.environ.get('CUDA_VISIBLE_DEVICES'),\n"
        "        'egl': os.environ.get('MUJOCO_EGL_DEVICE_ID')}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'ok': False, 'err': repr(e)}))\n"
        "    sys.exit(1)\n"
    )
    results: list[tuple[str, bool, str]] = []
    cvd_all = ",".join(gpus)
    for gpu in gpus:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cvd_all
        env["MUJOCO_GL"] = "egl"
        # MEDI is the global id; CVD lists all selected GPUs so the wrapper's
        # membership check passes and it can renumber MEDI to its CVD index.
        env["MUJOCO_EGL_DEVICE_ID"] = gpu
        env["EGL_DEVICE_ID"] = gpu
        try:
            out = subprocess.check_output(
                [sys.executable, "-c", snippet],
                env=env,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=60,
            )
            last = out.strip().splitlines()[-1] if out.strip() else "{}"
            results.append((gpu, True, last))
        except subprocess.CalledProcessError as e:
            results.append((gpu, False, e.output.strip().splitlines()[-1] if e.output else repr(e)))
        except Exception as e:
            results.append((gpu, False, repr(e)))
    return results


def check_egl_per_gpu_filtered(gpus: list[str]) -> list[tuple[str, bool, str]]:
    """Filter-mode pre-flight: for each GPU, spawn a child with the egl-filter
    env layout (CVD=$gpu_id, MEDI=0) and call robosuite's own EGL helper —
    the *same code path* eval will use when robosuite creates the offscreen
    renderer. This catches host configs where only EGL device 0 is visible
    and rejects MEDI>0.
    """
    snippet = (
        "import os, sys, json\n"
        "os.environ.setdefault('MUJOCO_GL', 'egl')\n"
        "try:\n"
        "    from robosuite.renderers.context.egl_context import (\n"
        "        create_initialized_egl_device_display,\n"
        "    )\n"
        "    create_initialized_egl_device_display(device_id=0)\n"
        "    print(json.dumps({'ok': True,\n"
        "        'cuda_visible': os.environ.get('CUDA_VISIBLE_DEVICES'),\n"
        "        'egl': os.environ.get('MUJOCO_EGL_DEVICE_ID')}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'ok': False, 'err': repr(e)}))\n"
        "    sys.exit(1)\n"
    )
    results: list[tuple[str, bool, str]] = []
    for gpu in gpus:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["MUJOCO_GL"] = "egl"
        env["MUJOCO_EGL_DEVICE_ID"] = "0"
        env["EGL_DEVICE_ID"] = "0"
        try:
            out = subprocess.check_output(
                [sys.executable, "-c", snippet],
                env=env,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=60,
            )
            last = out.strip().splitlines()[-1] if out.strip() else "{}"
            results.append((gpu, True, last))
        except subprocess.CalledProcessError as e:
            results.append((gpu, False, e.output.strip().splitlines()[-1] if e.output else repr(e)))
        except Exception as e:
            results.append((gpu, False, repr(e)))
    return results


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" in argv:
        sep = argv.index("--")
        head, forwarded = argv[:sep], argv[sep + 1 :]
    else:
        head, forwarded = argv, []

    p = argparse.ArgumentParser(
        prog="eval_libero_plus_multi_gpu.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--gpus",
        default=None,
        help='GPU ids, e.g. "0,1,2,3", or shorthand count "4". '
        "Defaults to $CUDA_VISIBLE_DEVICES, then `nvidia-smi -L`.",
    )
    p.add_argument(
        "--suites",
        default=",".join(DEFAULT_SUITES),
        help=f"Comma-separated LIBERO suite names. Default: {','.join(DEFAULT_SUITES)}",
    )
    p.add_argument(
        "--task-ids",
        default=None,
        help='Restrict to specific task ids. Either a flat list ("0,1,2") '
        'applied to every suite, or per-suite ("libero_spatial=0,1;libero_10=3,4").',
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Root directory for shard outputs and merged eval_info.json.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and print commands but do not execute lerobot-eval.",
    )
    p.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip launching shards; just re-merge the existing shard_*/ "
        "eval_info.json files under --output-dir.",
    )
    p.add_argument(
        "--use-lerobot-eval",
        action="store_true",
        help="Legacy mode: spawn one ``lerobot-eval`` subprocess per "
        "(shard, suite) pair. This pays the model-load + GPU-init cost "
        "once per suite. Default is the in-process worker which loads "
        "the policy once per shard.",
    )
    p.add_argument(
        "--shard-by",
        choices=["task", "suite"],
        default="task",
        help="Sharding strategy. 'task' (default): round-robin every "
        "(suite, task_id) tuple across GPUs — every GPU runs a mix of "
        "suites, balanced wall time. 'suite': assign whole suites to GPUs "
        "(suite i → GPU i %% N) — useful for A/B comparison and isolated "
        "per-suite throughput. Expect 'suite' to be slower when suite "
        "max_steps differ (e.g. libero_10:520 vs libero_spatial:280).",
    )
    p.add_argument(
        "--check-egl",
        action="store_true",
        help="Sanity-check that each GPU can independently bind an EGL "
        "context with CUDA_VISIBLE_DEVICES + MUJOCO_EGL_DEVICE_ID before "
        "launching the full sweep. Cheap (~1s/GPU). Recommended on a fresh "
        "host because EGL device→CUDA device mapping can drift.",
    )
    p.add_argument(
        "--egl-filter",
        action="store_true",
        help="Switch to per-shard CUDA_VISIBLE_DEVICES filtering with "
        "MUJOCO_EGL_DEVICE_ID=0. Use this on managed clusters / containers "
        "where the host exposes only 1 EGL device per process regardless of "
        "CVD (the default cuda-rank mode would have every shard's robosuite "
        "EGL collide on the same card). Detect this case by running, with "
        "CVD=0,1,2,3, robosuite's create_initialized_egl_device_display "
        "for device_id in 0..N-1 — if only device_id=0 succeeds, use "
        "--egl-filter. Pre-flight (--check-egl) is automatically routed "
        "through robosuite's helper in this mode.",
    )

    ns = p.parse_args(head[1:])  # head[0] is the script path
    return ns, forwarded


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args, forwarded = parse_args(argv if argv is not None else sys.argv)

    bad = [a for a in forwarded if a.split("=", 1)[0] in RESERVED_FORWARD_ARGS]
    if bad:
        log.error(
            "These args are controlled by the launcher and must not be forwarded: %s",
            bad,
        )
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        merged = merge_results(args.output_dir)
        out_path = args.output_dir / "eval_info.json"
        with out_path.open("w") as f:
            json.dump(merged, f, indent=2)
        print(f"Wrote merged report → {out_path}")
        print(json.dumps(merged["overall"], indent=2))
        return 0

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    gpus = detect_gpus(args.gpus)
    if not gpus:
        log.error("No GPUs detected. Pass --gpus explicitly.")
        return 2

    log.info("Using GPUs: %s", gpus)

    if args.check_egl:
        log.info(
            "Checking EGL binding per GPU (mode: %s)...",
            "egl-filter" if args.egl_filter else "cuda-rank",
        )
        if args.egl_filter:
            results = check_egl_per_gpu_filtered(gpus)
        else:
            results = check_egl_per_gpu(gpus)
        bad = [(g, msg) for g, ok, msg in results if not ok]
        for gpu, ok, msg in results:
            log.info("  GPU %s: %s %s", gpu, "OK" if ok else "FAIL", msg)
        if bad:
            log.error(
                "EGL check failed on %d GPU(s). Aborting before sweep.%s",
                len(bad),
                "" if args.egl_filter else (
                    " If only GPU 0 succeeds and others fail with 'between 0 "
                    "and 0 (inclusive)', the host exposes only 1 EGL device "
                    "per process regardless of CVD — re-run with --egl-filter."
                ),
            )
            return 2

    log.info("Probing suite sizes...")
    task_counts = probe_suite_sizes(suites)
    log.info("Suite sizes: %s", task_counts)

    plan = make_plan(
        suites,
        task_counts,
        parse_task_ids_arg(args.task_ids),
        len(gpus),
        shard_by=args.shard_by,
    )
    log.info("Sharding strategy: %s", args.shard_by)
    for i, (gpu, plan_i) in enumerate(zip(gpus, plan, strict=True)):
        total = sum(len(ids) for _, ids in plan_i)
        log.info("Shard %d → GPU %s: %d tasks (%s)", i, gpu, total, plan_i)

    # Resolve mode + forwarded-arg translation BEFORE the dry-run gate so a
    # dry run still validates user input. Otherwise typos in forwarded args
    # only surface after a real launch.
    if args.use_lerobot_eval:
        if args.egl_filter:
            log.info(
                "Mode: legacy lerobot-eval (egl-filter mode is the same env "
                "layout this path already uses — CVD per shard, MEDI=$gpu_id)."
            )
        else:
            log.info(
                "Mode: legacy lerobot-eval (one subprocess per (shard, suite); "
                "model loaded per-suite — slower)."
            )
        shard_runner = run_shard_per_suite
        runner_kwargs: dict = {}
        runner_args: tuple = (forwarded,)
    else:
        worker_args, unknown = _translate_to_worker_args(forwarded)
        if unknown:
            log.error(
                "These forwarded args aren't supported in worker mode: %s. "
                "Either remove them or pass --use-lerobot-eval to use the "
                "legacy per-suite subprocess path that accepts arbitrary "
                "draccus flags.",
                unknown,
            )
            return 2
        log.info(
            "Mode: in-process worker (model loaded ONCE per shard — fast). "
            "EGL: %s. Translated forwarded args: %s",
            "egl-filter (single-CVD per shard, MEDI=0)"
            if args.egl_filter
            else "cuda-rank (full-CVD shared, MEDI=$gpu_id, set_device(rank))",
            worker_args,
        )
        shard_runner = run_shard_worker
        runner_kwargs = {"egl_filter": args.egl_filter}
        runner_args = (worker_args,)

    if args.dry_run:
        log.info("Dry run; not launching.")
        return 0

    threads: list[threading.Thread] = []
    rc_holders: list[list[int]] = []
    timing_holders: list[list[dict]] = []
    sweep_started = time.time()
    for i, (gpu, plan_i) in enumerate(zip(gpus, plan, strict=True)):
        rc_holder = [0]
        rc_holders.append(rc_holder)
        timing_holder: list[dict] = []
        timing_holders.append(timing_holder)
        t = threading.Thread(
            target=shard_runner,
            args=(
                i,
                gpu,
                gpus,
                plan_i,
                *runner_args,
                args.output_dir,
                False,
                rc_holder,
                timing_holder,
            ),
            kwargs=runner_kwargs,
            daemon=False,
            name=f"shard-{i}-gpu-{gpu}",
        )
        t.start()
        threads.append(t)
        # Tiny stagger so Hub downloads / GPU init don't all hit at once.
        time.sleep(2.0)

    for t in threads:
        t.join()

    sweep_elapsed = time.time() - sweep_started
    rcs = [h[0] for h in rc_holders]
    if any(rcs):
        log.warning("Some shards exited non-zero: %s — merging partial results.", rcs)

    merged = merge_results(args.output_dir)
    out_path = args.output_dir / "eval_info.json"
    with out_path.open("w") as f:
        json.dump(merged, f, indent=2)

    # Wall-time summary — the actual point of running multi-GPU.
    shard_elapsed: list[tuple[int, str, float]] = []
    sum_suite_s = 0.0
    for h in timing_holders:
        if not h:
            continue
        rec = h[0]
        shard_elapsed.append((rec["shard"], rec["gpu_id"], rec["elapsed_s"]))
        for s in rec["suites"]:
            sum_suite_s += s["elapsed_s"]

    print("\n========== Wall-time summary ==========")
    print(f"Strategy:        {args.shard_by}")
    print(f"GPUs:            {len(gpus)} ({','.join(gpus)})")
    print(f"Sweep wall time: {sweep_elapsed:.1f}s "
          f"({sweep_elapsed/3600:.2f}h)")
    if shard_elapsed:
        slowest = max(s for _, _, s in shard_elapsed)
        fastest = min(s for _, _, s in shard_elapsed)
        print(f"Slowest shard:   {slowest:.1f}s "
              f"(slowest/fastest = {slowest/max(1e-9, fastest):.2f}× — "
              f"closer to 1.00 = better balanced)")
        # If we'd run the same work on 1 GPU sequentially, it would take
        # ~ sum of all (shard, suite) elapsed. Speedup is that / wall.
        print(f"Sequential est.: {sum_suite_s:.1f}s "
              f"→ speedup vs 1×GPU ≈ {sum_suite_s/max(1e-9, sweep_elapsed):.2f}×")
        print("Per-shard:")
        for sid, gid, s in sorted(shard_elapsed):
            print(f"  shard {sid:02d} gpu {gid}: {s:.1f}s")

    print("\n========== Merged eval ==========")
    print(f"Output: {out_path}")
    print("Overall:", json.dumps(merged["overall"], indent=2))
    print("Per group:")
    for g, agg in merged["per_group"].items():
        print(f"  {g}: {json.dumps(agg)}")

    # Persist a tiny benchmarking record so two runs (e.g. --shard-by task
    # vs --shard-by suite) can be compared after the fact.
    bench_path = args.output_dir / "sweep_timing.json"
    with bench_path.open("w") as f:
        json.dump(
            {
                "strategy": args.shard_by,
                "gpus": gpus,
                "sweep_elapsed_s": sweep_elapsed,
                "sum_shard_suite_s": sum_suite_s,
                "shards": [
                    {"shard": sid, "gpu_id": gid, "elapsed_s": s}
                    for sid, gid, s in shard_elapsed
                ],
            },
            f,
            indent=2,
        )
    print(f"\nSweep timing → {bench_path}")

    return 0 if not any(rcs) else 1


if __name__ == "__main__":
    sys.exit(main())
