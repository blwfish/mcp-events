"""Canonical test suite for mcp_events.

Covers all cases specified in SPEC_OOB_Events.md:
  - Basic accumulation and severity thresholds
  - Nested context semantics (inner shares outer accumulator)
  - Stderr fallback when no context is active
  - Decorator (sync + async), envelope shape, omit-when-empty
  - Async-safety: concurrent coroutines are isolated
  - Boundary: threshold edges, empty-string inputs, context defaults
"""
from __future__ import annotations

import asyncio

import pytest

from mcp_events import (
    Event,
    EventAccumulator,
    emit_event,
    event_context,
    get_current,
    soft_failure,
    with_events,
)


# ---------------------------------------------------------------------------
# EventAccumulator — basic accumulation
# ---------------------------------------------------------------------------


class TestEventAccumulator:
    def test_emit_all_severities(self) -> None:
        acc = EventAccumulator()
        acc.info("c1", "info msg")
        acc.warn("c2", "warn msg")
        acc.error("c3", "error msg")
        assert len(acc.events) == 3
        assert acc.events[0].severity == "info"
        assert acc.events[1].severity == "warn"
        assert acc.events[2].severity == "error"

    def test_to_envelope_default_excludes_info(self) -> None:
        acc = EventAccumulator()
        acc.info("c_info", "info msg")
        acc.warn("c_warn", "warn msg")
        acc.error("c_error", "error msg")
        codes = {e["code"] for e in acc.to_envelope()}
        assert "c_info" not in codes
        assert "c_warn" in codes
        assert "c_error" in codes

    def test_to_envelope_info_threshold_includes_info(self) -> None:
        acc = EventAccumulator()
        acc.info("c_info", "info msg")
        acc.warn("c_warn", "warn msg")
        codes = {e["code"] for e in acc.to_envelope("info")}
        assert "c_info" in codes
        assert "c_warn" in codes

    def test_to_envelope_error_threshold_excludes_warn(self) -> None:
        """Boundary: warn-level event is excluded when min_severity='error'."""
        acc = EventAccumulator()
        acc.warn("c_warn", "warn msg")
        acc.error("c_error", "error msg")
        codes = {e["code"] for e in acc.to_envelope("error")}
        assert "c_warn" not in codes
        assert "c_error" in codes

    def test_to_envelope_default_is_warn_not_info(self) -> None:
        """Confirm default min_severity is 'warn', not 'info'."""
        acc = EventAccumulator()
        acc.info("only_info", "info only")
        assert acc.to_envelope() == []

    def test_to_envelope_returns_empty_list_not_none(self) -> None:
        acc = EventAccumulator()
        result = acc.to_envelope()
        assert result == []
        assert result is not None

    def test_has_any_default_info(self) -> None:
        acc = EventAccumulator()
        assert not acc.has_any()
        acc.info("c1", "info")
        assert acc.has_any()

    def test_has_any_warn_false_with_only_info(self) -> None:
        acc = EventAccumulator()
        acc.info("c1", "info msg")
        assert not acc.has_any("warn")

    def test_has_any_warn_true_with_warn(self) -> None:
        acc = EventAccumulator()
        acc.warn("c1", "warn msg")
        assert acc.has_any("warn")

    def test_soft_failure_is_warn(self) -> None:
        acc = EventAccumulator()
        acc.soft_failure("fail_code", "something failed", {"key": "val"})
        assert len(acc.events) == 1
        e = acc.events[0]
        assert e.severity == "warn"
        assert e.code == "fail_code"
        assert e.message == "something failed"
        assert e.context == {"key": "val"}

    def test_events_property_is_snapshot(self) -> None:
        """events returns a copy — later emissions don't modify prior snapshots."""
        acc = EventAccumulator()
        acc.warn("c1", "first")
        snapshot = acc.events
        acc.warn("c2", "second")
        assert len(snapshot) == 1

    def test_context_defaults_to_empty_dict(self) -> None:
        acc = EventAccumulator()
        acc.warn("c1", "warn msg")
        assert acc.events[0].context == {}

    def test_context_absent_from_envelope_when_empty(self) -> None:
        acc = EventAccumulator()
        acc.warn("c1", "warn msg")  # no context kwarg
        envelope = acc.to_envelope()
        assert "context" not in envelope[0]

    def test_context_present_in_envelope_when_nonempty(self) -> None:
        acc = EventAccumulator()
        acc.warn("c1", "warn msg", {"age_days": 5})
        envelope = acc.to_envelope()
        assert envelope[0]["context"] == {"age_days": 5}

    def test_envelope_shape(self) -> None:
        acc = EventAccumulator()
        acc.warn("stale_cache_fallback", "Using cached snapshot", {"age_days": 5})
        envelope = acc.to_envelope()
        assert len(envelope) == 1
        e = envelope[0]
        assert e["level"] == "warn"
        assert e["code"] == "stale_cache_fallback"
        assert e["message"] == "Using cached snapshot"
        assert e["context"] == {"age_days": 5}

    def test_emission_order_preserved(self) -> None:
        acc = EventAccumulator()
        acc.warn("first", "first msg")
        acc.warn("second", "second msg")
        acc.warn("third", "third msg")
        codes = [e["code"] for e in acc.to_envelope()]
        assert codes == ["first", "second", "third"]

    def test_empty_code_and_message_accepted(self) -> None:
        """Empty code+message inside a context is accepted without raising."""
        acc = EventAccumulator()
        acc.warn("", "")
        assert len(acc.events) == 1


# ---------------------------------------------------------------------------
# event_context — context management
# ---------------------------------------------------------------------------


class TestEventContext:
    def test_basic_context_manager(self) -> None:
        with event_context() as events:
            events.warn("c1", "warn msg")
        assert len(events.events) == 1

    def test_two_sequential_contexts_are_isolated(self) -> None:
        with event_context() as ev1:
            ev1.warn("c1", "msg1")
        with event_context() as ev2:
            ev2.warn("c2", "msg2")
        codes1 = {e.code for e in ev1.events}
        codes2 = {e.code for e in ev2.events}
        assert codes1 == {"c1"}
        assert codes2 == {"c2"}

    def test_nested_context_shares_outer_accumulator(self) -> None:
        """Inner event_context is a no-op; events accumulate into outer."""
        with event_context() as outer:
            with event_context() as inner:
                assert inner is outer
                inner.warn("inner_warn", "from inner")
            outer.warn("outer_warn", "from outer")
        codes = {e.code for e in outer.events}
        assert "inner_warn" in codes
        assert "outer_warn" in codes

    def test_get_current_inside_context(self) -> None:
        assert get_current() is None
        with event_context() as events:
            assert get_current() is events
        assert get_current() is None

    def test_context_cleaned_up_after_exception(self) -> None:
        try:
            with event_context():
                raise ValueError("deliberate test error")
        except ValueError:
            pass
        assert get_current() is None


# ---------------------------------------------------------------------------
# Top-level shortcuts
# ---------------------------------------------------------------------------


class TestTopLevelFunctions:
    def test_soft_failure_inside_context(self) -> None:
        with event_context() as events:
            soft_failure("test_fail", "something failed", {"k": "v"})
        assert len(events.events) == 1
        e = events.events[0]
        assert e.severity == "warn"
        assert e.code == "test_fail"

    def test_emit_event_inside_context(self) -> None:
        with event_context() as events:
            emit_event("error", "test_error", "something broke")
        assert events.events[0].severity == "error"

    def test_soft_failure_outside_context_writes_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert get_current() is None
        soft_failure("oob_fail", "outside-context failure")
        captured = capsys.readouterr()
        assert "[mcp_events]" in captured.err
        assert "oob_fail" in captured.err

    def test_emit_event_outside_context_writes_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert get_current() is None
        emit_event("warn", "oob_warn", "outside-context warning")
        captured = capsys.readouterr()
        assert "[mcp_events]" in captured.err
        assert "oob_warn" in captured.err

    def test_soft_failure_outside_context_does_not_raise(self) -> None:
        assert get_current() is None
        soft_failure("test_code", "test message")  # must not raise

    def test_empty_code_and_message_outside_context_triggers_meta_warning(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert get_current() is None
        emit_event("warn", "", "")
        captured = capsys.readouterr()
        assert "meta-warning" in captured.err


# ---------------------------------------------------------------------------
# with_events decorator
# ---------------------------------------------------------------------------


class TestWithEventsDecorator:
    def test_sync_tool_with_warn_gets_events(self) -> None:
        @with_events()
        def my_tool() -> dict:  # type: ignore[return]
            soft_failure("test_fail", "something failed")
            return {"status": "ok"}

        result = my_tool()
        assert "events" in result
        assert result["events"][0]["code"] == "test_fail"
        assert result["events"][0]["level"] == "warn"

    def test_sync_tool_no_events_omits_field(self) -> None:
        @with_events()
        def my_tool() -> dict:
            return {"status": "ok"}

        result = my_tool()
        assert "events" not in result

    def test_async_tool_with_warn_gets_events(self) -> None:
        @with_events()
        async def my_async_tool() -> dict:  # type: ignore[return]
            soft_failure("async_fail", "async failure")
            return {"status": "ok"}

        result = asyncio.run(my_async_tool())
        assert "events" in result
        assert result["events"][0]["code"] == "async_fail"

    def test_async_tool_no_events_omits_field(self) -> None:
        @with_events()
        async def my_async_tool() -> dict:
            return {"status": "ok"}

        result = asyncio.run(my_async_tool())
        assert "events" not in result

    def test_decorator_preserves_function_name(self) -> None:
        @with_events()
        def my_named_tool() -> dict:
            return {"status": "ok"}

        assert my_named_tool.__name__ == "my_named_tool"

    def test_decorator_with_info_min_severity_includes_info(self) -> None:
        @with_events(min_severity="info")
        def my_tool() -> dict:  # type: ignore[return]
            emit_event("info", "info_code", "info message")
            return {"status": "ok"}

        result = my_tool()
        assert "events" in result
        assert result["events"][0]["level"] == "info"

    def test_decorator_default_excludes_info(self) -> None:
        """Default min_severity='warn' means info events don't trigger events field."""
        @with_events()
        def my_tool() -> dict:
            emit_event("info", "info_code", "info message")
            return {"status": "ok"}

        result = my_tool()
        assert "events" not in result

    def test_non_dict_result_not_modified(self) -> None:
        @with_events()
        def my_tool() -> str:
            soft_failure("f", "fail")
            return "plain string result"

        result = my_tool()
        assert result == "plain string result"

    def test_error_events_included_in_response(self) -> None:
        @with_events()
        def my_tool() -> dict:  # type: ignore[return]
            emit_event("error", "write_failed", "telemetry write failed")
            return {"status": "ok"}

        result = my_tool()
        assert "events" in result
        assert result["events"][0]["level"] == "error"

    def test_nested_helper_events_reach_decorator(self) -> None:
        """Helper that uses soft_failure contributes to the decorator's envelope."""

        def helper() -> None:
            soft_failure("helper_fail", "helper soft failure")

        @with_events()
        def my_tool() -> dict:  # type: ignore[return]
            helper()
            return {"status": "ok"}

        result = my_tool()
        assert "events" in result
        assert result["events"][0]["code"] == "helper_fail"


# ---------------------------------------------------------------------------
# Async-safety: concurrent coroutines must not bleed events across tasks
# ---------------------------------------------------------------------------


class TestAsyncSafety:
    def test_concurrent_coroutines_isolated(self) -> None:
        """Two concurrent coroutines each see only their own events."""

        async def task1() -> list[dict]:
            with event_context() as events:
                await asyncio.sleep(0)  # yield to task2
                events.warn("task1_warn", "Task 1 warning")
                return events.to_envelope()

        async def task2() -> list[dict]:
            with event_context() as events:
                events.warn("task2_warn", "Task 2 warning")
                await asyncio.sleep(0)  # yield to task1
                return events.to_envelope()

        async def run() -> tuple[list[dict], list[dict]]:
            results = await asyncio.gather(task1(), task2())
            return results[0], results[1]

        r1, r2 = asyncio.run(run())
        assert [e["code"] for e in r1] == ["task1_warn"]
        assert [e["code"] for e in r2] == ["task2_warn"]

    def test_async_nested_context_shares_accumulator(self) -> None:
        """Nested event_context inside async coroutine returns outer accumulator."""

        async def inner() -> EventAccumulator:
            with event_context() as events:
                soft_failure("inner_fail", "inner failure")
                return events

        async def outer() -> list[dict]:
            with event_context() as events:
                inner_acc = await inner()
                assert inner_acc is events
                return events.to_envelope()

        result = asyncio.run(outer())
        assert any(e["code"] == "inner_fail" for e in result)

    def test_with_events_async_isolation(self) -> None:
        """Two concurrent @with_events tasks don't share events."""

        @with_events()
        async def tool_a() -> dict:  # type: ignore[return]
            await asyncio.sleep(0)
            soft_failure("a_fail", "failure in A")
            return {"tool": "a"}

        @with_events()
        async def tool_b() -> dict:  # type: ignore[return]
            soft_failure("b_fail", "failure in B")
            await asyncio.sleep(0)
            return {"tool": "b"}

        async def run() -> tuple[dict, dict]:
            results = await asyncio.gather(tool_a(), tool_b())
            return results[0], results[1]

        ra, rb = asyncio.run(run())
        codes_a = {e["code"] for e in ra.get("events", [])}
        codes_b = {e["code"] for e in rb.get("events", [])}
        assert codes_a == {"a_fail"}
        assert codes_b == {"b_fail"}


# ---------------------------------------------------------------------------
# Boundary tests (threshold edges, empty inputs, Event defaults)
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_warn_threshold_includes_warn_excludes_info(self) -> None:
        """At-value boundary: min_severity='warn' includes warn, not info."""
        acc = EventAccumulator()
        acc.info("i", "info")
        acc.warn("w", "warn")
        codes = {e["code"] for e in acc.to_envelope("warn")}
        assert "i" not in codes
        assert "w" in codes

    def test_error_threshold_excludes_warn(self) -> None:
        """Above-value boundary: min_severity='error' excludes warn."""
        acc = EventAccumulator()
        acc.warn("w", "warn")
        acc.error("e", "error")
        codes = {e["code"] for e in acc.to_envelope("error")}
        assert "w" not in codes
        assert "e" in codes

    def test_to_envelope_returns_list_not_none_when_empty(self) -> None:
        acc = EventAccumulator()
        result = acc.to_envelope("warn")
        assert isinstance(result, list)
        assert result is not None

    def test_event_context_defaults(self) -> None:
        e = Event(severity="warn", code="c", message="m")
        assert e.context == {}

    def test_has_any_empty_accumulator(self) -> None:
        acc = EventAccumulator()
        assert not acc.has_any("info")
        assert not acc.has_any("warn")
        assert not acc.has_any("error")

    def test_emit_event_info_inside_context(self) -> None:
        with event_context() as events:
            emit_event("info", "info_code", "info message")
        assert events.events[0].severity == "info"
        assert events.events[0].code == "info_code"
