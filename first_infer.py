#!/usr/bin/env python3
"""APXInf π0.5 BF16 首次推理取证；需要在已构建的 NVIDIA GPU 环境运行。"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import platform
import time
import traceback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("warmup 必须非负，samples 必须大于零")

    import numpy as np
    from apxinf_robo import build_robot_policy

    model_dir = args.model_dir.resolve()
    for filename in ("config.json", "norm_stats.json", "tokenizer.model", "model.safetensors", "verified-assets.json"):
        if not (model_dir / filename).is_file():
            parser.error(f"缺少模型资产：{model_dir / filename}")
    # 每次运行使用新目录，保留之前的取证结果。
    args.out_dir.mkdir(parents=True, exist_ok=False)
    policy = None
    stage = "checkpoint_load"
    try:
        print("REAL_CHECKPOINT_LOAD_START", dt.datetime.now(dt.timezone.utc).isoformat(), flush=True)
        print("checkpoint:", model_dir, flush=True)
        started = time.perf_counter()
        policy = build_robot_policy(
            "franka_libero", model_dir, precision="bf16", device="cuda:0",
            action_horizon=10, num_flow_steps=10,
            norm_stats=model_dir / "norm_stats.json",
            tokenizer_path=model_dir / "tokenizer.model",
            discrete_state=False, seed=0,
        )
        load_ms = (time.perf_counter() - started) * 1000
        metadata = dict(policy.metadata)
        (args.out_dir / "policy-metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
        print(f"REAL_CHECKPOINT_LOADED load_ms={load_ms:.3f}", flush=True)
        print("resolved metadata:", json.dumps(metadata, default=str), flush=True)
        stage = "prepare_synthetic_observation"
        observation = {
            key: np.zeros((256, 256, 3), dtype=np.uint8)
            for key in metadata["image_keys"]
        }
        prompt_key = metadata.get("prompt_key", "prompt")
        observation[prompt_key] = "put both moka pots on the stove"
        state_key = metadata.get("state_key")
        if state_key is not None:
            observation[state_key] = np.zeros(int(metadata["state_dim"]), np.float32)
        arrays = {key: value for key, value in observation.items() if isinstance(value, np.ndarray)}
        np.savez_compressed(args.out_dir / "synthetic_observation.npz", **arrays)
        (args.out_dir / "observation.json").write_text(
            json.dumps({
                "source": "synthetic interface check: black RGB frames and zero state",
                "prompt_key": prompt_key,
                "prompt": observation[prompt_key],
                "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                           for key, value in arrays.items()},
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        stage = "first_infer"
        print("FIRST_INFER_START | real weights / synthetic black RGB and zero state", flush=True)
        started = time.perf_counter()
        first = policy.infer(observation)
        first_call_wall_ms = (time.perf_counter() - started) * 1000

        def checked_actions(result):
            actions = np.asarray(result["actions"])
            if actions.shape != (10, 7) or not np.isfinite(actions).all():
                raise RuntimeError(f"动作输出不符合 H=10、7 维 Franka 约定：{actions.shape}")
            if actions.dtype != np.float32:
                raise RuntimeError(f"预期 float32 部署动作，实际为 {actions.dtype}")
            return actions

        actions = checked_actions(first).copy()
        np.save(args.out_dir / "first_actions.npy", actions, allow_pickle=False)
        if "normalized_actions" in first:
            np.save(args.out_dir / "first_normalized_actions.npy",
                    np.asarray(first["normalized_actions"]), allow_pickle=False)
        first_summary = {"first_call_wall_ms": first_call_wall_ms, "timing": first.get("timing", {}),
                         "actions_shape": list(actions.shape), "actions_dtype": str(actions.dtype),
                         "actions_finite": True, "first_action": actions[0].tolist()}
        (args.out_dir / "first-call.json").write_text(json.dumps(first_summary, indent=2, default=str) + "\n")
        print("FIRST_INFER_OK", json.dumps(first_summary, default=str), flush=True)
        stage = "warmup"
        for index in range(args.warmup):
            checked_actions(policy.infer(observation))
            print(f"warmup {index + 1}/{args.warmup}", flush=True)

        stage = "steady_state_samples"
        samples = []
        for index in range(args.samples):
            started = time.perf_counter()
            result = policy.infer(observation)
            wall_ms = (time.perf_counter() - started) * 1000
            checked_actions(result)
            samples.append({"policy_wall_ms": wall_ms, "timing": result.get("timing", {})})
            with (args.out_dir / "measurements.jsonl").open("a") as stream:
                stream.write(json.dumps({"sample": index + 1, **samples[-1]}, default=str) + "\n")
            if (index + 1) % 5 == 0 or index + 1 == args.samples:
                print(f"measured {index + 1}/{args.samples} | last policy wall {wall_ms:.3f} ms", flush=True)

        wall_values = [item["policy_wall_ms"] for item in samples]
        summary = {
            "scope": "real checkpoint, synthetic observations; no robot task evaluation",
            "robot": "franka_libero", "precision": "bf16", "device": "cuda:0",
            "action_horizon": 10, "num_flow_steps": 10, "seed": 0,
            "discrete_state": False,
            "state_handling": "state is saved as input but is not injected into the prompt; package default made explicit",
            "model_dir": str(model_dir), "python": platform.python_version(),
            "load_ms": load_ms,
            "first_call_wall_ms": first_call_wall_ms,
            "first_call_timing": first.get("timing", {}),
            "actions_shape": list(actions.shape), "actions_dtype": str(actions.dtype),
            "actions_finite": True, "first_action": actions[0].tolist(),
            "warmup_after_first": args.warmup, "samples": args.samples,
            "policy_wall_p50_ms": float(np.percentile(wall_values, 50)),
            "policy_wall_p95_ms": float(np.percentile(wall_values, 95)),
            "percentile_method": "numpy.percentile default linear interpolation",
            "metadata": metadata, "measurements": samples,
        }
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({key: summary[key] for key in (
            "scope", "actions_shape", "actions_dtype", "actions_finite", "load_ms",
            "first_call_wall_ms", "policy_wall_p50_ms", "policy_wall_p95_ms",
        )}, ensure_ascii=False, indent=2))
        print(f"saved: {args.out_dir.resolve()}")
        print("REAL_CHECKPOINT_FIRST_INFER_VERIFIED", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        failure = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "stage": stage,
                   "error_type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        (args.out_dir / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    finally:
        if policy is not None:
            policy.close()


if __name__ == "__main__":
    main()
