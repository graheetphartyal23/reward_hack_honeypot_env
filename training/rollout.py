"""Rollout harness: run an episode with a pluggable policy function.

A policy is a function `policy(obs, state) -> HoneypotAction`. The harness
handles the reset/step loop and returns a trajectory summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from ..models import HoneypotAction, HoneypotObservation, HoneypotState


class Policy(Protocol):
    name: str

    def __call__(
        self, obs: HoneypotObservation, state: HoneypotState
    ) -> HoneypotAction: ...


@dataclass
class Trajectory:
    policy_name: str
    task_id: str
    actions: list[dict] = field(default_factory=list)
    step_rewards: list[float] = field(default_factory=list)
    final_reward: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0
    cheat_flags: list[str] = field(default_factory=list)
    task_reward: float = 0.0
    cheat_score: float = 0.0
    truncated: bool = False
    honeypots_placed: list[str] = field(default_factory=list)


def run_episode(
    env,  # HoneypotEnvironment-like (direct, not client)
    policy: Policy,
    task_id: Optional[str] = None,
    seed: Optional[int] = None,
    max_steps: int = 30,
) -> Trajectory:
    obs = env.reset(task_id=task_id, seed=seed)
    task = env._sandbox.task
    traj = Trajectory(
        policy_name=getattr(policy, "name", policy.__class__.__name__),
        task_id=task.task_id,
        honeypots_placed=list(env._sandbox.honeypots),
    )

    for _ in range(max_steps + 2):
        state = env.state
        action = policy(obs, state)
        traj.actions.append(
            {"tool": action.tool, "path": action.path, "has_content": bool(action.content)}
        )
        obs = env.step(action)
        traj.step_rewards.append(obs.reward or 0.0)
        if obs.done:
            break

    traj.final_reward = sum(traj.step_rewards)
    traj.tests_passed = obs.tests_passed or 0
    traj.tests_total = obs.tests_total or 0
    traj.task_reward = obs.task_reward or 0.0
    traj.cheat_score = obs.cheat_score or 0.0
    traj.cheat_flags = list(obs.cheat_flags or [])
    traj.truncated = obs.truncated
    return traj
