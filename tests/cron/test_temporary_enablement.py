"""Temporary enablement actually expires (hermes-gitops #476).

`resume_job(..., until=...)` used to be announced by the platform CLI and
revoked by nothing — a "30 minutes for debugging" enable was a standing
schedule with a comforting message. The scheduler's tick now re-pauses
past-deadline jobs before computing dueness, so a temporary enable can
never fire past its deadline.
"""

from datetime import datetime, timedelta, timezone

from cron.jobs import (
    create_job,
    expire_temporary_enablements,
    get_job,
    pause_job,
    resume_job,
)


def _iso(delta_minutes: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=delta_minutes)).isoformat()


def _make_paused_job():
    job = create_job(prompt="probe", schedule="every 1h")
    pause_job(job["id"])
    return job


class TestResumeUntil:
    def test_resume_with_until_stores_the_deadline(self):
        job = _make_paused_job()
        resumed = resume_job(job["id"], until=_iso(30))
        assert resumed["enabled"] is True
        assert resumed["resume_until"]

    def test_a_plain_resume_clears_a_previous_deadline(self):
        # Re-enabling for real must not inherit last week's expiry.
        job = _make_paused_job()
        resume_job(job["id"], until=_iso(30))
        pause_job(job["id"])
        resumed = resume_job(job["id"])
        assert resumed["resume_until"] is None


class TestExpiry:
    def test_a_past_deadline_re_pauses_with_a_self_explaining_reason(self):
        job = _make_paused_job()
        resume_job(job["id"], until=_iso(-5))
        assert expire_temporary_enablements() == 1
        after = get_job(job["id"])
        assert after["enabled"] is False
        assert after["state"] == "paused"
        assert "temporary enablement expired" in (after["paused_reason"] or "")
        # The deadline is consumed - a later manual resume is a fresh,
        # untimed enablement, not a replay of the expired one.
        assert after.get("resume_until") is None

    def test_a_future_deadline_is_left_alone(self):
        job = _make_paused_job()
        resume_job(job["id"], until=_iso(30))
        assert expire_temporary_enablements() == 0
        assert get_job(job["id"])["enabled"] is True

    def test_jobs_without_a_deadline_are_never_touched(self):
        job = create_job(prompt="steady", schedule="every 1h")
        assert expire_temporary_enablements() == 0
        assert get_job(job["id"])["enabled"] is True

    def test_a_malformed_deadline_never_pauses_the_job(self):
        # Fail open on parse: a garbage deadline is visible in `cron list`
        # and must not silently disable a schedule someone relies on.
        job = _make_paused_job()
        resume_job(job["id"], until="not-a-date")
        assert expire_temporary_enablements() == 0
        assert get_job(job["id"])["enabled"] is True
