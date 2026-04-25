"""FastAPI app for reward_hack_honeypot_env.

Run locally:
    uvicorn reward_hack_honeypot_env.server.app:app --host 0.0.0.0 --port 8000

Or via the `server` entry point defined in pyproject.toml:
    server
"""

from __future__ import annotations

import json
from typing import Any, Dict

from openenv.core.env_server import create_app
from fastapi import Query
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..env.local_env import HoneypotLocalEnv, standardize_step
from ..models import HoneypotAction, HoneypotObservation
from .environment import HoneypotEnvironment

_base_app = create_app(
    HoneypotEnvironment,
    HoneypotAction,
    HoneypotObservation,
    env_name="reward_hack_honeypot_env",
)


def _enrich_openenv_rl_shape(payload: dict[str, Any], *, path: str) -> dict[str, Any]:
    """Add `state` + `info` alongside OpenEnv's existing keys (keeps clients working).

    OpenEnv HTTP `/reset` and `/step` return:
      {observation, reward, done}

    For hackathon training ergonomics, we *also* include:
      {state, info}

    - `state` is set to the same value as `observation` (the "RL-visible" state)
    - `info` is an empty dict by default
    - `reward` is always a float (0.0 when missing)
    """
    if "observation" in payload and "state" not in payload:
        payload = {**payload, "state": payload.get("observation")}

    if "info" not in payload or not isinstance(payload.get("info"), dict):
        payload = {**payload, "info": {**({} if not isinstance(payload.get("info"), dict) else payload.get("info", {}))}}

    r = payload.get("reward", 0.0)
    if r is None:
        r = 0.0
    payload["reward"] = float(r)
    payload["done"] = bool(payload.get("done", False))
    payload.setdefault("info", {})["rl_shape"] = f"openenv_http:{path}"
    return payload


class _OpenEnvStandardizeMiddleware:
    """ASGI middleware: enrich OpenEnv `/reset` and `/step` JSON with `state` + `info`.

    `create_app` returns JSON using Starlette's response streaming; you must read the full
    body from `body_iterator` (the sync `Response.body` bytes are often not populated).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path not in ("/reset", "/step"):
            await self.app(scope, receive, send)
            return

        # Capture the outgoing response to rewrite JSON bodies in-place
        response_started: dict[str, Any] = {}
        body_chunks: list[bytes] = []

        async def send_wrapper(message: Message) -> None:
            if message.get("type") == "http.response.start":
                response_started.update(message)  # keep a reference to mutate headers
                # Defer actually sending start until we have a full body (small JSON endpoints)
                return
            if message.get("type") == "http.response.body":
                body_chunks.append(message.get("body") or b"")
                more = bool(message.get("more_body", False))
                if more:
                    return
                # End of body: now we can post-process
                status = int((response_started.get("status") or 200))
                raw = b"".join(body_chunks)

                headers: list[tuple[bytes, bytes]] = list(response_started.get("headers") or [])

                content_type = ""
                for k, v in headers:
                    if k.lower() == b"content-type":
                        content_type = v.decode("latin-1")
                        break

                if status < 400 and "application/json" in content_type:
                    try:
                        payload = json.loads(raw.decode("utf-8"))
                        if isinstance(payload, dict):
                            enriched = _enrich_openenv_rl_shape(payload, path=path)
                            raw = json.dumps(enriched).encode("utf-8")
                    except Exception:
                        # If anything goes wrong, fall back to the original body.
                        pass

                # Remove content-length (we changed body size) if present
                new_headers: list[tuple[bytes, bytes]] = []
                for k, v in headers:
                    if k.lower() in (b"content-length",):
                        continue
                    new_headers.append((k, v))

                new_headers = [
                    h
                    for h in new_headers
                    if h[0].lower() not in (b"content-length", b"content-length")
                ] + [(b"content-length", str(len(raw)).encode("ascii"))]

                await send(
                    {
                        "type": "http.response.start",
                        "status": status,
                        "headers": new_headers,
                    }
                )
                await send({"type": "http.response.body", "body": raw, "more_body": False})
                return

            await send(message)

        await self.app(scope, receive, send_wrapper)


_RL_ENVS: Dict[str, HoneypotLocalEnv] = {}


def _get_rl_env(session: str) -> HoneypotLocalEnv:
    if session not in _RL_ENVS:
        _RL_ENVS[session] = HoneypotLocalEnv()
    return _RL_ENVS[session]

@_base_app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Reward Hack Honeypot is running",
        "openenv": {"reset": "/reset", "step": "/step", "state": "/state"},
        "macro_rl": {
            "reset": "/api/rl/v1/reset",
            "step": "/api/rl/v1/step",
        },
    }


@_base_app.post("/api/rl/v1/reset")
def rl_reset(
    session: str = Query("default", description="In-memory session id for /api/rl/* routes only."),
    task_id: str | None = Query("caesar_cipher"),
    split: str = Query("train"),
    seed: int | None = Query(None),
) -> dict[str, Any]:
    """Non-OpenEnv local-macro environment reset (for notebooks / quick demos)."""
    env = _get_rl_env(session)
    state = env.reset(task_id=task_id, split=split, seed=seed)
    return standardize_step(state=state, reward=0.0, done=False, info={"session": session})


@_base_app.post("/api/rl/v1/step")
def rl_step(
    body: dict[str, Any],
    session: str = Query("default"),
) -> dict[str, Any]:
    action = str(body.get("action", ""))
    env = _get_rl_env(session)
    state, reward, done, info = env.step(action)
    info = {**info, "session": session}
    return standardize_step(state=state, reward=reward, done=done, info=info)

# Public ASGI app: OpenEnv + JSON shaping middleware + extra routes above.
app: Any = _OpenEnvStandardizeMiddleware(_base_app)  # type: ignore[assignment]
    
def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
