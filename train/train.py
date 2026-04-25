from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt


# When executed as a script (`python train/train.py`), ensure the project root is on sys.path.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reward_hack_honeypot_env.env.local_env import HoneypotLocalEnv  # noqa: E402


@dataclass
class RunConfig:
    episodes: int = 100
    seed: int = 0
    task_id: str = "caesar_cipher"
    split: str = "train"
    out_png: str = str(Path(__file__).resolve().parent / "reward_curve.png")
    out_json: str = str(Path(__file__).resolve().parent / "metrics.json")


def _avg(xs: list[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def run(cfg: RunConfig) -> dict:
    random.seed(cfg.seed)
    rng = random.Random(cfg.seed)

    env = HoneypotLocalEnv(seed=cfg.seed)
    ep_returns: list[float] = []

    for ep in range(1, cfg.episodes + 1):
        env.reset(task_id=cfg.task_id, split=cfg.split, seed=ep)
        done = False
        total = 0.0

        # A tiny episode budget in *macro* steps. Each macro `step` may expand
        # to many underlying tool calls; we keep the outer horizon short.
        macro_lim = min(env.max_steps, 3)

        t = 0
        while not done and t < macro_lim:
            t += 1
            if ep <= 50:
                a = rng.choice(env.action_space)
            else:
                a = "safe_action"

            _st, r, done, _info = env.step(a)
            total += float(r)
            if done:
                break

        ep_returns.append(float(total))
        if ep % 20 == 0 or ep == 1:
            print(f"[episode {ep:03d}] return={total:+.3f} last_action_bias={'random' if ep <= 50 else 'safe'}")

    first10 = _avg(ep_returns[:10]) if len(ep_returns) >= 10 else _avg(ep_returns)
    last10 = _avg(ep_returns[-10:]) if len(ep_returns) >= 10 else _avg(ep_returns)

    print("\n=== Summary ===")
    print(f"episodes: {cfg.episodes}")
    print(f"avg return first 10: {first10:+.3f}")
    print(f"avg return last 10:  {last10:+.3f}")
    print(f"delta (last - first): {last10 - first10:+.3f}")

    os.makedirs(str(Path(cfg.out_png).parent), exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(ep_returns) + 1), ep_returns, "-", alpha=0.8)
    plt.axvline(50, color="gray", linestyle="--", linewidth=1, label="policy switch")
    plt.xlabel("Episode")
    plt.ylabel("Return (sum of macro step rewards)")
    plt.title("Macro-action training (random → safe bias)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.out_png, dpi=120)
    print(f"saved: {cfg.out_png}")

    metrics = {
        "config": asdict(cfg),
        "first10_mean_return": first10,
        "last10_mean_return": last10,
        "returns": ep_returns,
    }
    with open(cfg.out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved: {cfg.out_json}")
    return metrics


def main() -> None:
    run(RunConfig())


if __name__ == "__main__":
    main()
