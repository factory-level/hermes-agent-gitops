"""The three Codex catches on the #476 lifecycle/temporary-enablement work.

1. Lifecycle records never carry DERIVED job names (prompt/skill/script text
   is content, the record envelope is metadata-only) - `lifecycle_job_label`
   falls back to the opaque id unless the name is provably authored.
2. `claim_job_for_fire` enforces `resume_until` - external scheduler
   providers and the manual cronjob tool claim directly, without the tick,
   so expiry must live at the atomic claim seam and fail closed.
3. `resume_job(until=...)` rejects malformed and already-past deadlines -
   an unparseable deadline used to become a permanent enablement.

Real store against a temp HERMES_HOME, per the E2E-over-mocks discipline.
"""
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def test_derived_name_never_reaches_lifecycle_label(temp_home):
    from cron.jobs import create_job, lifecycle_job_label

    job = create_job(prompt="rotate token sk-live-SECRET", schedule="every 5m")
    assert job["name_derived"] is True
    assert lifecycle_job_label(job) == job["id"]
    assert "sk-live" not in lifecycle_job_label(job)


def test_authored_name_reaches_lifecycle_label(temp_home):
    from cron.jobs import create_job, lifecycle_job_label

    job = create_job(prompt="x", schedule="every 5m", name="communication-proof")
    assert job["name_derived"] is False
    assert lifecycle_job_label(job) == "communication-proof"


def test_legacy_job_without_flag_falls_back_to_id():
    from cron.jobs import lifecycle_job_label

    legacy = {"id": "job-123", "name": "whatever the prompt was"}
    assert lifecycle_job_label(legacy) == "job-123"


def test_claim_refuses_and_pauses_expired_enablement(temp_home):
    from cron.jobs import create_job, pause_job, resume_job, claim_job_for_fire, get_job

    job = create_job(prompt="x", schedule="every 5m", name="t")
    pause_job(job["id"])
    resume_job(job["id"], until=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat())
    # Simulate the deadline passing without a tick: rewrite it into the past.
    from cron.jobs import update_job

    update_job(job["id"], {"resume_until": (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()})

    assert claim_job_for_fire(job["id"]) is False
    after = get_job(job["id"])
    assert after["enabled"] is False
    assert after["state"] == "paused"
    assert "expired" in (after.get("paused_reason") or "")
    assert after.get("resume_until") is None


def test_claim_honors_future_enablement(temp_home):
    from cron.jobs import create_job, pause_job, resume_job, claim_job_for_fire

    job = create_job(prompt="x", schedule="every 5m", name="t2")
    pause_job(job["id"])
    resume_job(job["id"], until=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat())
    assert claim_job_for_fire(job["id"]) is True


def test_resume_rejects_malformed_deadline(temp_home):
    from cron.jobs import create_job, pause_job, resume_job

    job = create_job(prompt="x", schedule="every 5m", name="t3")
    pause_job(job["id"])
    with pytest.raises(ValueError, match="not an ISO-8601"):
        resume_job(job["id"], until="in thirty minutes")


def test_resume_rejects_past_deadline(temp_home):
    from cron.jobs import create_job, pause_job, resume_job

    job = create_job(prompt="x", schedule="every 5m", name="t4")
    pause_job(job["id"])
    with pytest.raises(ValueError, match="already in the past"):
        resume_job(job["id"], until=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat())
