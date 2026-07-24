"""Plugin protocol — the contract every plugin implements."""

from typing import Protocol, runtime_checkable
from fastapi import FastAPI


@runtime_checkable
class Plugin(Protocol):
    """Standard interface for platform plugins.

    Plugins implement only the hooks they need. All methods have default
    no-op implementations via the BasePlugin helper class.
    """

    name: str

    def register(self, app: FastAPI, config: dict) -> None:
        """Called at startup — register HTTP routes, DB tables, etc."""
        ...

    def on_ticket(self, ticket_key: str, event: str, payload: dict) -> None:
        """Called when a ticket event arrives (created, updated, commented)."""
        ...

    def on_schedule(self) -> None:
        """Called on each pipeline cycle (scheduled or webhook-triggered)."""
        ...


class BasePlugin:
    """Convenience base — subclass and override only the hooks you need."""

    name: str = "unnamed"

    def register(self, app: FastAPI, config: dict) -> None:
        pass

    def on_ticket(self, ticket_key: str, event: str, payload: dict) -> None:
        pass

    def on_schedule(self) -> None:
        pass
