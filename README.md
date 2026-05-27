# mcp-events

Lightweight out-of-band event channel for MCP servers.

A tool call can accumulate warnings and errors that don't belong in the main
return value but also can't be silently swallowed. `mcp-events` gives those
signals a coherent path to the calling LLM.

## Quick start

```python
from mcp_events import with_events, soft_failure, emit_event

@with_events()
@mcp.tool()
def my_tool(args):
    try:
        result = risky_thing()
    except Exception as e:
        soft_failure("risky_thing_failed", str(e), {"context": "..."})
        result = fallback_result()
    return {"status": "ok", "data": result}
```

When warnings or errors are emitted, the decorator appends them to the response:

```json
{
  "status": "ok",
  "data": "...",
  "events": [
    {
      "level": "warn",
      "code": "risky_thing_failed",
      "message": "connection reset by peer",
      "context": {"context": "..."}
    }
  ]
}
```

The `events` field is omitted entirely when nothing notable happened.

## API

### Emitting events

```python
from mcp_events import soft_failure, emit_event

# soft_failure is an alias for warn — use at previously-silent except blocks
soft_failure("code", "message", {"optional": "context"})

# emit at any severity
emit_event("info" | "warn" | "error", "code", "message", {...})
```

Both functions route to the active `EventAccumulator` when inside an
`event_context`, or fall back to stderr when called outside one.

### Context management

```python
from mcp_events import event_context, get_current

with event_context() as events:
    events.warn("code", "message")
    response = {"status": "ok"}
    if events.has_any("warn"):
        response["events"] = events.to_envelope()
    return response
```

Nested `event_context` blocks share the outer accumulator — events from
deeply-nested helpers accumulate into the top-level context rather than
being discarded.

### Decorator

```python
from mcp_events import with_events

@with_events()          # default: surface warn+error events
@with_events("info")    # surface all events including info
```

Works on both sync and async functions. The `events` key is omitted when no
events meet the threshold.

### Persistence

```python
acc.events  # list[Event] snapshot — iterate for local persistence
```

Each `Event` has `severity`, `code`, `message`, `context` (dict, defaults to `{}`).

## Response envelope

```json
{
  "status": "ok",
  "events": [
    {
      "level": "warn",
      "code": "stale_cache_fallback",
      "message": "Using cached snapshot from 5 days ago",
      "context": {"snapshot_age_days": 5}
    }
  ]
}
```

- `events` is omitted when empty — never `"events": []`
- `context` is omitted when empty — never `"context": {}`
- Emission order is chronological

## Distribution

In development, install from the local path in each consuming project:

```toml
# pyproject.toml
[project]
dependencies = ["mcp-events>=0.1.0"]

[tool.uv.sources]
mcp-events = { path = "/path/to/mcp-events", editable = true }
```

Downstream users (installing from PyPI) ignore the `[tool.uv.sources]` block
and get the published package.

## Running tests

```
uv run pytest
```
