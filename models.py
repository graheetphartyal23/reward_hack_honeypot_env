"""Action / Observation / State types for reward_hack_honeypot_env."""

from __future__ import annotations

from typing import Literal, Optional

from openenv.core.env_server.interfaces import Action, Observation, State

ToolName = Literal[
    "list_dir",
    "read_file",
    "write_file",
    "run_tests",
    "submit",
]


class HoneypotAction(Action):
    """A single tool invocation in the sandbox."""

    tool: ToolName
    path: Optional[str] = None
    content: Optional[str] = None


class HoneypotObservation(Observation):
    """Tool output + task status.

    `cheat_flags` is always an empty list until `submit`, so the agent cannot
    use observed honeypot detection as a reward-hack signal during the episode.
    """

    tool_output: str = ""
    tests_passed: Optional[int] = None
    tests_total: Optional[int] = None
    task_reward: Optional[float] = None
    cheat_score: Optional[float] = None
    cheat_flags: list[str] = []
    truncated: bool = False


class HoneypotState(State):
    task_id: str = ""
    actions_taken: int = 0
    files_read: list[str] = []
    files_written: list[str] = []
    commands_run: list[str] = []
    submitted: bool = False
