"""The two #349 negative paths that had no test (hermes-gitops #476):

1. An unauthorized actor is REFUSED with a lifecycle record (gateway
   plane, outcome refused) - never a silent drop - and the record carries
   identifiers only, no message content.
2. A delivery failure is recorded SEPARATELY from job completion
   (cron.delivery.failed alongside cron.completed) - a delivered-nothing
   run that reports success is the failure mode #349 exists to close.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest


class _Sink(BaseHTTPRequestHandler):
    records = []

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).records.append(json.loads(body))
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
    monkeypatch.setenv("HERMES_OBSERVER_URL", f"http://127.0.0.1:{server.server_port}")
    yield _Sink
    server.shutdown()


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_unauthorized_actor_is_refused_with_a_record_not_a_silent_drop(sink):
    from gateway.authz_mixin import GatewayAuthorizationMixin
    from gateway.config import Platform
    from gateway.session import SessionSource

    class _Gateway(GatewayAuthorizationMixin):
        def _is_user_authorized_inner(self, source):
            return False

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="123456",
        chat_type="channel",
        user_id="999888777",
    )
    assert _Gateway()._is_user_authorized(source) is False

    assert _wait_for(
        lambda: any(r.get("type") == "gateway.message.refused" for r in _Sink.records)
    ), "refusal produced no lifecycle record - a silent drop"
    record = next(r for r in _Sink.records if r["type"] == "gateway.message.refused")
    assert record["outcome"] == "refused"
    assert record["plane"] == "gateway"
    # Identifiers only, never content: the actor ref is the id, and the
    # record has no content-bearing fields at all.
    assert record["redactedSource"]["ref"] == "999888777"
    assert not any(k in record for k in ("content", "text", "body", "payload"))


def test_delivery_failure_is_recorded_separately_from_job_completion(sink):
    import cron.scheduler as sched

    job = {"id": "job-df", "name": "t", "prompt": "do work", "deliver": "discord"}
    with patch("cron.scheduler.claim_dispatch", return_value=True), \
         patch("agent.secret_scope.set_secret_scope", return_value=None), \
         patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
         patch("agent.secret_scope.reset_secret_scope"), \
         patch("cron.scheduler.run_job", return_value=(True, "full output", "final response", None)), \
         patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
         patch("cron.scheduler._is_cron_silence_response", return_value=False), \
         patch("cron.scheduler._deliver_result", side_effect=Exception("adapter down")), \
         patch("cron.scheduler.mark_job_run"):
        assert sched.run_one_job(job) is True

    assert _wait_for(
        lambda: any(r.get("type") == "cron.delivery.failed" for r in _Sink.records)
        and any(r.get("type") in ("cron.completed", "cron.failed") for r in _Sink.records)
    ), "delivery failure and job completion were not both recorded"
    failed = next(r for r in _Sink.records if r["type"] == "cron.delivery.failed")
    done = next(r for r in _Sink.records if r["type"] in ("cron.completed", "cron.failed"))
    assert failed["outcome"] == "failure"
    # SEPARATE records sharing one trace: the delivery verdict never
    # overwrites or masquerades as the job verdict.
    assert failed["recordId"] != done["recordId"]
    assert failed["traceId"] == done["traceId"]
