from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reward_hack_honeypot_env.train.train import RunConfig, run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Run macro-action demo training + save artifacts.")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task-id", type=str, default="caesar_cipher")
    p.add_argument("--split", type=str, default="train")
    p.add_argument(
        "--out-png",
        type=str,
        default=str(Path(__file__).resolve().parent / "reward_curve.png"),
    )
    p.add_argument(
        "--out-json",
        type=str,
        default=str(Path(__file__).resolve().parent / "metrics.json"),
    )
    args = p.parse_args()

    cfg = RunConfig(
        episodes=args.episodes,
        seed=args.seed,
        task_id=args.task_id,
        split=args.split,
        out_png=args.out_png,
        out_json=args.out_json,
    )
    metrics = run(cfg)
    # Print a compact JSON summary to stdout for CI / judge convenience
    print(json.dumps({k: metrics[k] for k in ("first10_mean_return", "last10_mean_return")}, indent=2))


if __name__ == "__main__":
    main()
