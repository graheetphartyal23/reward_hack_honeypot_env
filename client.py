"""HoneypotEnv client — typed EnvClient over HTTP."""

from __future__ import annotations

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import HoneypotAction, HoneypotObservation, HoneypotState


class HoneypotEnv(EnvClient[HoneypotAction, HoneypotObservation, HoneypotState]):
    """Client for reward_hack_honeypot_env."""

    def _step_payload(self, action: HoneypotAction) -> dict:
        return {
            "tool": action.tool,
            "path": action.path,
            "content": action.content,
        }

    def _parse_result(self, payload: dict) -> StepResult[HoneypotObservation]:
        obs = HoneypotObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> HoneypotState:
        return HoneypotState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            actions_taken=payload.get("actions_taken", 0),
            files_read=payload.get("files_read", []),
            files_written=payload.get("files_written", []),
            commands_run=payload.get("commands_run", []),
            submitted=payload.get("submitted", False),
        )
