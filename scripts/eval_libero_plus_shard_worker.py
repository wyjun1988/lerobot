#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""In-process shard worker for multi-GPU LIBERO-plus evaluation.

The companion launcher (``eval_libero_plus_multi_gpu.py``) used to spawn one
``lerobot-eval`` subprocess per (shard, suite) pair. That paid the full
startup cost — Python imports, hub download, policy load, GPU init,
processor setup — once per *suite*, not once per *shard*. For VLAs that's
30–120s of pure overhead per call, multiplied by the number of suites in
the shard.

This worker fixes that. One process per shard, model loaded once, then a
loop over the shard's ``(suite, task_ids)`` plan that calls
``cfg.create_envs()`` and ``eval_policy_all`` directly — keeping the same
policy + processors across suites. All LIBERO suites share the same
observation/action spaces, so the policy doesn't need to be reconstructed.

Inputs are passed via:
  * ``--plan-file`` — JSON: ``[["libero_spatial", [0,2,4]], ...]``
  * Standard CLI flags for the bits of ``EvalPipelineConfig`` we actually
    use. Anything fancier than these flags (custom rename maps with deep
    structure, exotic policy overrides) should still go through the
    regular ``lerobot-eval`` path.

Output is one ``eval_info.json`` written to ``--output-dir`` with the same
shape as ``lerobot-eval`` produces, plus a ``suite_runs`` field timing
each (suite, task_ids) iteration so the launcher can attribute wall time.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import close_envs, make_env_pre_post_processors
from lerobot.envs.configs import LiberoPlusEnv
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

log = logging.getLogger("eval_shard_worker")


def _build_env_cfg(suite: str, task_ids: list[int], args) -> LiberoPlusEnv:
    """Construct a single-suite LiberoPlusEnv config matching ``--env.*`` CLI flags.

    All LIBERO suites share the same observation/action spaces, so we use
    one cfg per suite only because ``create_libero_envs`` reads ``task`` and
    ``task_ids`` off it. Building a fresh cfg per iteration is cheap — no
    GPU work happens until ``create_envs`` runs.
    """
    cfg = LiberoPlusEnv(task=suite, task_ids=list(task_ids))
    cfg.control_mode = args.control_mode
    cfg.max_parallel_tasks = args.max_parallel_tasks
    if args.episode_length is not None:
        cfg.episode_length = args.episode_length
    return cfg


def _aggregate_global(per_task: list[dict]) -> dict:
    """Recompute per-group + overall metrics from per-episode booleans.

    Mirrors what the launcher does at merge time, but for a single shard so
    the worker writes a self-contained eval_info.json compatible with the
    upstream lerobot-eval shape.
    """
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
        "pc_success": 100.0 * sum(overall_s) / len(overall_s)
        if overall_s
        else float("nan"),
        "avg_sum_reward": _avg(overall_sum),
        "avg_max_reward": _avg(overall_max),
    }
    return {"overall": overall, "per_group": per_group}


def main(argv: list[str] | None = None) -> int:
    init_logging()
    register_third_party_plugins()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-file", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--policy-path", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--use-amp", action="store_true")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--use-async-envs", action="store_true")
    p.add_argument("--control-mode", default="relative")
    p.add_argument("--max-parallel-tasks", type=int, default=1)
    p.add_argument("--episode-length", type=int, default=None)
    p.add_argument("--max-episodes-rendered", type=int, default=0)
    p.add_argument(
        "--rename-map",
        default=None,
        help="JSON dict, e.g. '{\"observation.images.image\": \"observation.images.cam0\"}'.",
    )
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Override policy.n_action_steps at eval time — number of actions "
        "from each predicted action chunk that get executed before the policy "
        "is re-queried. Smaller values = more frequent re-planning (more "
        "compute, potentially more reactive). Larger values = open-loop "
        "execution of more steps. Default: None (use whatever the saved "
        "policy config has).",
    )
    p.add_argument(
        "--cuda-rank",
        type=int,
        default=None,
        help="Index of this shard's GPU within CUDA_VISIBLE_DEVICES. The "
        "launcher leaves CVD as the full set of selected GPUs across all "
        "shards (so MuJoCo's EGL wrapper can membership-check + renumber "
        "MUJOCO_EGL_DEVICE_ID), and uses --cuda-rank to tell each worker "
        "which physical card to pin its CUDA tensors to. Without this, all "
        "shards default to cuda:0 and collide.",
    )
    args = p.parse_args(argv)

    # Pin CUDA to this shard's physical GPU BEFORE any other CUDA op (policy
    # construction, dataset_stats tensors, etc). After this, ``cfg.device =
    # "cuda"`` and ``torch.tensor(..., device="cuda")`` both land on the
    # right card. Done early so even module-level CUDA touches from imports
    # below pick up the right default.
    if args.cuda_rank is not None and torch.cuda.is_available():
        torch.cuda.set_device(int(args.cuda_rank))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan_file.read_text())
    if not plan:
        log.error("Empty plan; nothing to evaluate.")
        return 0
    rename_map = json.loads(args.rename_map) if args.rename_map else {}

    # ---- Set up device / determinism ----
    template_suite, template_ids = plan[0][0], list(plan[0][1])
    template_env_cfg = _build_env_cfg(template_suite, template_ids, args)

    # PreTrainedConfig.from_pretrained applies CLI overrides via cli_overrides;
    # we instead patch fields directly because we own this Python entry point.
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = Path(args.policy_path)
    policy_cfg.device = args.device
    policy_cfg.use_amp = bool(args.use_amp)
    if args.n_action_steps is not None:
        if not hasattr(policy_cfg, "n_action_steps"):
            log.warning(
                "Policy config %s has no n_action_steps attribute; "
                "--n-action-steps=%d will be ignored.",
                type(policy_cfg).__name__,
                args.n_action_steps,
            )
        else:
            log.info(
                "Override n_action_steps: %s -> %d",
                policy_cfg.n_action_steps,
                args.n_action_steps,
            )
            policy_cfg.n_action_steps = args.n_action_steps

    device = get_safe_torch_device(policy_cfg.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)

    # ---- Load policy + processors ONCE (the whole point of this worker) ----
    log.info("Loading policy from %s ...", args.policy_path)
    t_load = time.time()
    policy = make_policy(cfg=policy_cfg, env_cfg=template_env_cfg, rename_map=rename_map)
    policy.eval()
    load_s = time.time() - t_load
    log.info("Policy loaded in %.1fs (this used to be paid per suite)", load_s)

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=template_env_cfg, policy_cfg=policy_cfg
    )

    # ---- Iterate plan, building env per (suite, task_ids) ----
    autocast_ctx = (
        torch.autocast(device_type=device.type) if policy_cfg.use_amp else nullcontext()
    )
    per_task: list[dict] = []
    suite_runs: list[dict] = []
    started = time.time()
    rc = 0

    with torch.no_grad(), autocast_ctx:
        for suite, task_ids in plan:
            task_ids = list(task_ids)
            log.info("Suite=%s n_tasks=%d ids=%s", suite, len(task_ids), task_ids)
            t0 = time.time()
            suite_cfg = _build_env_cfg(suite, task_ids, args)
            envs = suite_cfg.create_envs(
                n_envs=args.batch_size,
                use_async_envs=args.use_async_envs,
            )
            videos_dir = (
                args.output_dir / "videos" / suite if args.max_episodes_rendered > 0 else None
            )
            try:
                info = eval_policy_all(
                    envs=envs,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=args.n_episodes,
                    max_episodes_rendered=args.max_episodes_rendered,
                    videos_dir=videos_dir,
                    start_seed=args.seed,
                    max_parallel_tasks=suite_cfg.max_parallel_tasks,
                )
            except Exception:  # noqa: BLE001 — log + continue across suites
                log.exception("Suite %s failed; continuing with next suite.", suite)
                # log.exception relies on the configured logging handler
                # picking up exc_info; lerobot's init_logging() format string
                # has historically dropped tracebacks on this code path. Force
                # a copy to stderr AND a per-suite error file so the actual
                # exception is always recoverable from the shard output.
                tb_text = traceback.format_exc()
                print(f"\n===== Suite {suite} traceback =====\n{tb_text}", file=sys.stderr, flush=True)
                err_path = args.output_dir / f"error_{suite}.txt"
                try:
                    err_path.write_text(tb_text)
                except OSError:
                    pass
                # eval_policy_all closes envs internally, but in the failure
                # path it may not have. Best-effort cleanup:
                try:
                    close_envs(envs)
                except Exception:  # noqa: BLE001
                    pass
                rc = 1
                suite_runs.append(
                    {
                        "suite": suite,
                        "n_tasks": len(task_ids),
                        "elapsed_s": time.time() - t0,
                        "rc": 1,
                        "overall": None,
                    }
                )
                continue

            elapsed = time.time() - t0
            for t in info.get("per_task", []):
                per_task.append(t)
            suite_runs.append(
                {
                    "suite": suite,
                    "n_tasks": len(task_ids),
                    "elapsed_s": elapsed,
                    "rc": 0,
                    "overall": info.get("overall", {}),
                }
            )
            log.info("Suite=%s done in %.1fs overall=%s", suite, elapsed, info.get("overall"))

    total_s = time.time() - started
    aggregated = _aggregate_global(per_task)
    out = {
        **aggregated,
        "per_task": per_task,
        "suite_runs": suite_runs,
        "worker_timing": {
            "policy_load_s": load_s,
            "rollout_total_s": total_s,
            "n_suites": len(plan),
        },
    }
    out_path = args.output_dir / "eval_info.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    log.info("Wrote %s", out_path)
    log.info("Overall: %s", aggregated["overall"])
    log.info(
        "Saved %.1fs of policy-reload by amortizing across %d suites "
        "(load=%.1fs vs %d×load=%.1fs)",
        max(0.0, (len(plan) - 1) * load_s),
        len(plan),
        load_s,
        len(plan),
        len(plan) * load_s,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
