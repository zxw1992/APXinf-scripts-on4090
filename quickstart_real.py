#!/usr/bin/env python3
"""A small real-weight API example. Black frames test the interface, not a task."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from apxinf_robo import build_robot_policy

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-dir", required=True, type=Path)
parser.add_argument("--out-dir", required=True, type=Path)
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=False)

policy = build_robot_policy(
    "franka_libero", args.model_dir, precision="bf16", device="cuda:0",
    action_horizon=10, num_flow_steps=10, discrete_state=False, seed=0,
    norm_stats=args.model_dir / "norm_stats.json",
    tokenizer_path=args.model_dir / "tokenizer.model",
)
try:
    metadata = policy.metadata
    observation = {key: np.zeros((256, 256, 3), np.uint8) for key in metadata["image_keys"]}
    observation[metadata.get("prompt_key", "prompt")] = "put both moka pots on the stove"
    if metadata.get("state_key"):
        observation[metadata["state_key"]] = np.zeros(metadata["state_dim"], np.float32)
    started = time.perf_counter()
    result = policy.infer(observation)
    wall_ms = (time.perf_counter() - started) * 1000
    actions = np.asarray(result["actions"])
    assert actions.shape == (10, 7)
    assert actions.dtype == np.float32 and np.isfinite(actions).all()
    np.save(args.out_dir / "actions.npy", actions, allow_pickle=False)
    summary = {"scope": "real weights, synthetic input, API check only",
               "shape": list(actions.shape), "dtype": str(actions.dtype), "finite": True,
               "first_action": actions[0].tolist(), "first_call_wall_ms": wall_ms,
               "timing": result.get("timing", {}), "metadata": metadata}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print("QUICKSTART_REAL_WEIGHTS_OK", flush=True)
finally:
    policy.close()
