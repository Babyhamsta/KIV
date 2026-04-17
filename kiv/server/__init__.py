"""Ollama-compatible HTTP server for KIV."""

from .session import KIVSession

__all__ = ["KIVSession", "create_app", "run_server"]


def __getattr__(name: str):
    if name == "create_app":
        from .app import create_app

        globals()["create_app"] = create_app
        return create_app
    if name == "run_server":
        from .cli import run_server

        globals()["run_server"] = run_server
        return run_server
    raise AttributeError(f"module 'kiv.server' has no attribute {name!r}")
