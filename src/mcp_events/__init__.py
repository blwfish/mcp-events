"""mcp_events — lightweight out-of-band event channel for MCP servers.

Wire format stability: the level/code/message/context fields in the event envelope
are the stable wire contract. Adding new fields is backward-compatible; renaming
or removing fields is breaking and requires a version bump.
"""
from __future__ import annotations

import asyncio
import functools
import sys
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable

__version__ = "0.1.0"
__all__ = [
    "Event",
    "EventAccumulator",
    "event_context",
    "get_current",
    "soft_failure",
    "emit_event",
    "with_events",
]

_SEVERITY_RANK: dict[str, int] = {"info": 0, "warn": 1, "error": 2}


@dataclass
class Event:
    severity: str  # "info" | "warn" | "error"
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


class EventAccumulator:
    """Collects events during a single tool call."""

    def __init__(self) -> None:
        self._events: list[Event] = []

    def emit(
        self,
        severity: str,
        code: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        self._events.append(
            Event(severity=severity, code=code, message=message, context=context or {})
        )

    def info(self, code: str, message: str, context: dict[str, Any] | None = None) -> None:
        self.emit("info", code, message, context)

    def warn(self, code: str, message: str, context: dict[str, Any] | None = None) -> None:
        self.emit("warn", code, message, context)

    def error(self, code: str, message: str, context: dict[str, Any] | None = None) -> None:
        self.emit("error", code, message, context)

    def soft_failure(
        self, code: str, message: str, context: dict[str, Any] | None = None
    ) -> None:
        """Alias for warn(). Use at previously-silent except blocks."""
        self.warn(code, message, context)

    def has_any(self, min_severity: str = "info") -> bool:
        min_rank = _SEVERITY_RANK.get(min_severity, 0)
        return any(_SEVERITY_RANK.get(e.severity, 0) >= min_rank for e in self._events)

    def to_envelope(self, min_severity: str = "warn") -> list[dict[str, Any]]:
        """Render events at or above min_severity as JSON-serializable dicts.

        The returned list is empty (not None) when no events meet the threshold.
        The context key is omitted from each entry when the event's context is empty.
        """
        min_rank = _SEVERITY_RANK.get(min_severity, 0)
        result = []
        for e in self._events:
            if _SEVERITY_RANK.get(e.severity, 0) >= min_rank:
                entry: dict[str, Any] = {
                    "level": e.severity,
                    "code": e.code,
                    "message": e.message,
                }
                if e.context:
                    entry["context"] = e.context
                result.append(entry)
        return result

    @property
    def events(self) -> list[Event]:
        """Snapshot of all collected events (any severity). For persistence layers."""
        return list(self._events)


_current: ContextVar[EventAccumulator | None] = ContextVar(
    "_mcp_events_current", default=None
)


class event_context:
    """Context manager establishing an EventAccumulator for the current call.

    Nested use: if an outer event_context is already active, the inner one is a
    no-op and both share the outer accumulator. Events from nested helpers
    accumulate into the caller's context rather than being discarded.

    Usage::

        @mcp.tool()
        def my_tool(args):
            with event_context() as events:
                response = {"status": "ok", "data": ...}
                if events.has_any("warn"):
                    response["events"] = events.to_envelope()
                return response
    """

    def __init__(self) -> None:
        self._acc: EventAccumulator | None = None
        self._token: Any = None
        self._is_nested = False

    def __enter__(self) -> EventAccumulator:
        existing = _current.get()
        if existing is not None:
            self._is_nested = True
            self._acc = existing
        else:
            self._acc = EventAccumulator()
            self._token = _current.set(self._acc)
        return self._acc  # type: ignore[return-value]

    def __exit__(self, *exc: object) -> None:
        if not self._is_nested and self._token is not None:
            _current.reset(self._token)


def get_current() -> EventAccumulator | None:
    """Returns the active EventAccumulator if inside an event_context, else None."""
    return _current.get()


def _stderr_emit(
    severity: str, code: str, message: str, _context: dict[str, Any] | None
) -> None:
    if not code and not message:
        print(
            "[mcp_events] meta-warning: emit_event called with empty code and message",
            file=sys.stderr,
        )
    print(f"[mcp_events] {severity}: {code}: {message}", file=sys.stderr)


def soft_failure(
    code: str, message: str, context: dict[str, Any] | None = None
) -> None:
    """Record a warn-level soft failure. Falls back to stderr when no context is active."""
    acc = _current.get()
    if acc is not None:
        acc.soft_failure(code, message, context)
    else:
        _stderr_emit("warn", code, message, context)


def emit_event(
    severity: str, code: str, message: str, context: dict[str, Any] | None = None
) -> None:
    """Emit an event at any severity. Falls back to stderr when no context is active."""
    acc = _current.get()
    if acc is not None:
        acc.emit(severity, code, message, context)
    else:
        _stderr_emit(severity, code, message, context)


def with_events(
    min_severity: str = "warn",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator wrapping a tool with an event_context and auto-augmenting the response.

    Supports both sync and async tool functions. The ``events`` key is added to
    the response dict only when at least one event meets min_severity; it is
    omitted entirely when the envelope is empty. Non-dict return values are
    returned unchanged (events are silently dropped — use explicit event_context
    if you need to handle that case).

    Usage::

        @with_events()
        @mcp.tool()
        def my_tool(args):
            soft_failure("cache_miss", "Fell back to stale data")
            return {"status": "ok", "data": ...}
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with event_context() as events:
                    result = await fn(*args, **kwargs)
                    envelope = events.to_envelope(min_severity)
                    if envelope and isinstance(result, dict):
                        result["events"] = envelope
                    return result

            return async_wrapper
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with event_context() as events:
                    result = fn(*args, **kwargs)
                    envelope = events.to_envelope(min_severity)
                    if envelope and isinstance(result, dict):
                        result["events"] = envelope
                    return result

            return sync_wrapper

    return decorator
