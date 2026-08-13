"""Lifecycle record emitter (hermes-gitops #476, ADR-96/ADR-102).

Emits metadata-only lifecycle records — the ``lifecycle-record/v1alpha1``
envelope the hermes-gitops platform's observer reads — for the planes that
live inside this process: cron scheduling, agent runs, tool execution, and
gateway authorization refusals. The event router emits its own (router
plane) records with the same envelope; this module is the agent-side
counterpart.

Design constraints, in order:

1. **Zero cost when unconfigured.** ``HERMES_OBSERVER_URL`` unset means
   every call is a single ``os.environ`` read returning immediately.
   No thread, no socket, no import-time work.
2. **Never blocks, never raises into the caller.** Emission is a
   fire-and-forget POST from a single daemon worker thread with a short
   timeout and a BOUNDED queue — a dead observer drops records (counted),
   it never slows a cron tick or a tool call. Observability must not
   consume the budget of the thing it observes.
3. **Metadata only.** Record fields are ids, types, outcomes, durations
   and scalar detail values. No prompts, no tool arguments, no results,
   no message content — the redaction guard on the reading side
   (``hg communication trace``) treats value-shaped payloads as
   violations, and the honest way to pass that check is to never emit
   them.

The envelope mirrors ``plugin/schemas/lifecycle-record/v1alpha1`` in the
hermes-gitops-plugin repository; ``cli/src/lifecycle.ts``'s
``isLifecycleRecord`` is the runtime contract (version 1, UUID record and
trace ids, a closed plane enum, dotted lowercase type).
"""

from __future__ import annotations

import contextvars
import json
import os
import queue
import threading
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

__all__ = [
    "emit",
    "new_trace",
    "pending_count",
    "dropped_count",
    "set_current_trace",
    "reset_current_trace",
]

# The ambient trace (contextvar, so it survives the executor's contextvar
# propagation into tool worker threads): a run-level scope - cron's
# run_one_job, a gateway turn - sets it, and every emit() inside that
# scope that names no trace of its own joins it. This is what makes
# `hg communication trace <id>` show cron.triggered -> agent.started ->
# tool.* as ONE causal tree instead of a scatter of one-record traces.
_current_trace: "contextvars.ContextVar[Optional[Dict[str, str]]]" = contextvars.ContextVar(
    "hermes_lifecycle_trace", default=None
)


def set_current_trace(trace_id: str, causation_id: Optional[str] = None):
    """Install the ambient trace for this context; returns a reset token."""
    return _current_trace.set({"traceId": trace_id, **({"causationId": causation_id} if causation_id else {})})


def reset_current_trace(token) -> None:
    try:
        _current_trace.reset(token)
    except Exception:
        pass

_PLANES = {"gateway", "cron", "router", "alert", "agent", "tool"}

# Bounded: a dead observer must cap memory, not grow a backlog. 256 is
# generous for the burstiest real producer (a parallel tool batch).
_QUEUE_MAX = 256
_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=_QUEUE_MAX)
_worker_started = False
_worker_lock = threading.Lock()
_dropped = 0


def _observer_url() -> str:
    """Read per call, not at import: tests and long-lived gateways may
    set or clear it after this module loads."""
    return (os.environ.get("HERMES_OBSERVER_URL") or "").strip()


def new_trace() -> str:
    """A fresh trace id, for callers that start a causal chain."""
    return str(uuid.uuid4())


def pending_count() -> int:
    return _queue.qsize()


def dropped_count() -> int:
    return _dropped


def _worker() -> None:
    while True:
        record = _queue.get()
        url = _observer_url()
        if not url:
            continue
        try:
            req = urllib.request.Request(
                f"{url.rstrip('/')}/observer",
                data=json.dumps(record).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2.0):
                pass
        except Exception:
            global _dropped
            _dropped += 1


def _ensure_worker() -> None:
    global _worker_started
    if _worker_started:
        return
    with _worker_lock:
        if _worker_started:
            return
        t = threading.Thread(target=_worker, name="hermes-lifecycle-emit", daemon=True)
        t.start()
        _worker_started = True


def _scalar_detail(detail: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Scalars only, clipped — the schema's detail contract. Anything
    non-scalar is DROPPED rather than serialized: a dict or list smuggled
    into detail is exactly how payloads leak into a metadata stream."""
    if not detail:
        return None
    out: Dict[str, Any] = {}
    for key, value in detail.items():
        if isinstance(value, bool) or isinstance(value, (int, float)):
            out[str(key)[:60]] = value
        elif isinstance(value, str):
            out[str(key)[:60]] = value[:200]
    return out or None


def emit(
    plane: str,
    type_: str,
    *,
    trace_id: Optional[str] = None,
    causation_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    outcome: Optional[str] = None,
    duration_ms: Optional[float] = None,
    profile: Optional[str] = None,
    test_run: Optional[str] = None,
    source: Optional[Dict[str, str]] = None,
    target: Optional[Dict[str, str]] = None,
    detail: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Emit one lifecycle record, fire-and-forget.

    Returns ``{"recordId": ..., "traceId": ...}`` ALWAYS — even when the
    observer is unconfigured — so callers can thread causation
    unconditionally instead of branching on whether observability is on.
    """
    record_id = str(uuid.uuid4())
    ambient = _current_trace.get()
    if trace_id is None and ambient:
        trace_id = ambient["traceId"]
        if causation_id is None:
            causation_id = ambient.get("causationId")
    tid = trace_id or str(uuid.uuid4())
    if not _observer_url():
        return {"recordId": record_id, "traceId": tid}
    if plane not in _PLANES:
        return {"recordId": record_id, "traceId": tid}
    record: Dict[str, Any] = {
        "version": 1,
        "recordId": record_id,
        "traceId": tid,
        "plane": plane,
        "type": type_,
        "occurredAt": datetime.now(timezone.utc).isoformat(),
    }
    if causation_id:
        record["causationId"] = causation_id
    if correlation_id:
        record["correlationId"] = correlation_id
    if outcome:
        record["outcome"] = outcome
    if duration_ms is not None:
        record["durationMs"] = round(float(duration_ms), 1)
    if profile:
        record["profile"] = str(profile)[:80]
    if test_run:
        record["testRun"] = str(test_run)[:80]
    if source:
        record["redactedSource"] = {k: str(v)[:120] for k, v in source.items() if k in ("kind", "ref", "label")}
    if target:
        record["redactedTarget"] = {k: str(v)[:120] for k, v in target.items() if k in ("kind", "ref", "label")}
    scalars = _scalar_detail(detail)
    if scalars:
        record["detail"] = scalars
    _ensure_worker()
    try:
        _queue.put_nowait(record)
    except queue.Full:
        global _dropped
        _dropped += 1
    return {"recordId": record_id, "traceId": tid}
