"""hermes_lifecycle: the agent-side lifecycle record emitter (#476).

The contract under test is the reading side's (hermes-gitops
``isLifecycleRecord``): version 1, UUID ids, closed plane enum, dotted
type — plus this module's own three promises: free when unconfigured,
never blocking, metadata only.
"""

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import hermes_lifecycle


class _Sink(BaseHTTPRequestHandler):
    records = []

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).records.append((self.path, json.loads(body)))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):  # silence
        pass


@pytest.fixture()
def sink(monkeypatch):
    _Sink.records = []
    server = HTTPServer(("127.0.0.1", 0), _Sink)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("HERMES_OBSERVER_URL", url)
    yield _Sink
    server.shutdown()


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_unconfigured_is_free_and_still_returns_ids(monkeypatch):
    monkeypatch.delenv("HERMES_OBSERVER_URL", raising=False)
    before = hermes_lifecycle.pending_count()
    ids = hermes_lifecycle.emit("cron", "cron.triggered")
    # Callers thread causation unconditionally - ids come back either way.
    uuid.UUID(ids["recordId"])
    uuid.UUID(ids["traceId"])
    assert hermes_lifecycle.pending_count() == before  # nothing enqueued


def test_a_record_reaches_the_observer_with_the_v1_envelope(sink):
    ids = hermes_lifecycle.emit(
        "cron",
        "cron.completed",
        outcome="success",
        duration_ms=1234.5,
        profile="marketing-manager",
        detail={"job": "communication-proof"},
    )
    assert _wait_for(lambda: len(sink.records) == 1)
    path, record = sink.records[0]
    assert path == "/observer"
    assert record["version"] == 1
    assert record["recordId"] == ids["recordId"]
    assert record["traceId"] == ids["traceId"]
    assert record["plane"] == "cron"
    assert record["type"] == "cron.completed"
    assert record["outcome"] == "success"
    assert record["durationMs"] == 1234.5
    assert record["profile"] == "marketing-manager"
    assert record["detail"] == {"job": "communication-proof"}
    assert "occurredAt" in record


def test_causation_threads_across_records(sink):
    first = hermes_lifecycle.emit("cron", "cron.triggered")
    hermes_lifecycle.emit(
        "cron", "cron.started", trace_id=first["traceId"], causation_id=first["recordId"]
    )
    assert _wait_for(lambda: len(sink.records) == 2)
    second = sink.records[1][1]
    assert second["traceId"] == first["traceId"]
    assert second["causationId"] == first["recordId"]


def test_detail_is_scalars_only_and_clipped(sink):
    # A dict or list smuggled into detail is exactly how payloads leak
    # into a metadata stream - they are DROPPED, not serialized.
    hermes_lifecycle.emit(
        "tool",
        "tool.completed",
        detail={
            "tool": "web_search",
            "args": {"query": "secret payload"},
            "results": ["a", "b"],
            "long": "x" * 500,
        },
    )
    assert _wait_for(lambda: len(sink.records) == 1)
    detail = sink.records[0][1]["detail"]
    assert detail["tool"] == "web_search"
    assert "args" not in detail
    assert "results" not in detail
    assert len(detail["long"]) == 200


def test_an_unknown_plane_emits_nothing(sink):
    hermes_lifecycle.emit("business", "made.up")
    hermes_lifecycle.emit("cron", "cron.triggered")
    assert _wait_for(lambda: len(sink.records) == 1)
    assert sink.records[0][1]["plane"] == "cron"


def test_a_dead_observer_never_blocks_the_caller(monkeypatch):
    # An unroutable observer: emit() must return immediately; the worker
    # eats the failure and counts it.
    monkeypatch.setenv("HERMES_OBSERVER_URL", "http://127.0.0.1:1")
    start = time.monotonic()
    for _ in range(20):
        hermes_lifecycle.emit("agent", "agent.started")
    assert time.monotonic() - start < 0.5  # no network wait on the caller
