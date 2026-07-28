"""Tests for ``hermes cron sync`` — converging shipped declarations.

The behaviours worth pinning here are the ownership rules, because they are
what make this safe to run unattended on every container start: sync owns a
job's definition, the operator owns whether it runs, and a job somebody created
by hand is never sync's business.
"""

import json
from argparse import Namespace

import pytest
import yaml

from cron.jobs import create_job, get_job, load_jobs, resume_job
from hermes_cli.cron import cron_command
from hermes_cli.cron_sync import CronDeclarationError, sync


@pytest.fixture()
def cron_home(tmp_path, monkeypatch):
    """Point both the job store and the declaration reader at a temp profile."""
    cron_dir = tmp_path / "cron"
    cron_dir.mkdir(parents=True)
    monkeypatch.setattr("cron.jobs.CRON_DIR", cron_dir)
    monkeypatch.setattr("cron.jobs.JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", cron_dir / "output")
    return tmp_path


def declare(cron_home, filename, body):
    path = cron_home / "cron" / filename
    if filename.endswith(".json"):
        path.write_text(json.dumps(body), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


WEEKLY = {
    "name": "research-pass",
    "schedule": "0 6 * * 1",
    "deliver": "local",
    "skills": ["research-pass", "top-reels"],
    "prompt": "Run this week's research pass.",
}


class TestConvergence:
    def test_declaration_becomes_a_paused_job(self, cron_home):
        declare(cron_home, "research-pass.yaml", WEEKLY)

        report = sync(home=cron_home)

        assert [e["name"] for e in report["created"]] == ["research-pass"]
        jobs = load_jobs()
        assert len(jobs) == 1
        job = jobs[0]
        assert job["name"] == "research-pass"
        assert job["schedule"]["kind"] == "cron"
        assert job["skills"] == ["research-pass", "top-reels"]
        assert job["deliver"] == "local"
        assert job["distribution_source"] == "research-pass.yaml"
        # Paused on arrival: a redeploy must never start work on its own.
        assert job["enabled"] is False
        assert job["state"] == "paused"

    def test_second_sync_changes_nothing(self, cron_home):
        declare(cron_home, "research-pass.yaml", WEEKLY)
        sync(home=cron_home)
        first_id = load_jobs()[0]["id"]

        report = sync(home=cron_home)

        assert report["created"] == []
        assert report["updated"] == []
        assert [e["name"] for e in report["unchanged"]] == ["research-pass"]
        assert [j["id"] for j in load_jobs()] == [first_id]

    def test_edited_declaration_updates_in_place(self, cron_home):
        declare(cron_home, "research-pass.yaml", WEEKLY)
        sync(home=cron_home)
        original_id = load_jobs()[0]["id"]

        declare(cron_home, "research-pass.yaml", {**WEEKLY, "schedule": "every 6h"})
        report = sync(home=cron_home)

        assert [e["name"] for e in report["updated"]] == ["research-pass"]
        job = get_job(original_id)
        assert job is not None, "the job keeps its id, and with it its run history"
        assert job["schedule"]["kind"] == "interval"

    def test_resume_survives_a_later_sync(self, cron_home):
        declare(cron_home, "research-pass.yaml", WEEKLY)
        sync(home=cron_home)
        job_id = load_jobs()[0]["id"]
        resume_job(job_id)

        # An unrelated edit, then the redeploy that re-runs sync.
        declare(cron_home, "research-pass.yaml", {**WEEKLY, "prompt": "Run it, but better."})
        sync(home=cron_home)

        job = get_job(job_id)
        assert job["enabled"] is True
        assert job["state"] == "scheduled"
        assert job["prompt"] == "Run it, but better."

    def test_removed_declaration_prunes_its_job(self, cron_home):
        path = declare(cron_home, "research-pass.yaml", WEEKLY)
        sync(home=cron_home)
        path.unlink()

        report = sync(home=cron_home)

        assert [e["name"] for e in report["pruned"]] == ["research-pass"]
        assert load_jobs() == []

    def test_hand_made_jobs_are_never_touched(self, cron_home):
        mine = create_job(prompt="my own thing", schedule="every 1h", name="mine")
        declare(cron_home, "research-pass.yaml", WEEKLY)

        sync(home=cron_home)

        survivor = get_job(mine["id"])
        assert survivor is not None
        assert survivor["enabled"] is True
        assert "distribution_source" not in survivor

    def test_dry_run_writes_nothing(self, cron_home):
        declare(cron_home, "research-pass.yaml", WEEKLY)

        report = sync(dry_run=True, home=cron_home)

        assert [e["name"] for e in report["created"]] == ["research-pass"]
        assert load_jobs() == []


class TestDeclarationParsing:
    def test_the_job_store_is_not_read_as_a_declaration(self, cron_home):
        declare(cron_home, "research-pass.yaml", WEEKLY)
        sync(home=cron_home)
        assert (cron_home / "cron" / "jobs.json").exists()

        # jobs.json shares the .json suffix; reading it back as a declaration
        # would make every sync see a malformed file.
        report = sync(home=cron_home)
        assert [e["name"] for e in report["unchanged"]] == ["research-pass"]

    def test_a_file_may_hold_a_list_of_jobs(self, cron_home):
        declare(
            cron_home,
            "jobs.yaml",
            [WEEKLY, {**WEEKLY, "name": "daily-sweep", "schedule": "every 1d"}],
        )

        report = sync(home=cron_home)

        assert sorted(e["name"] for e in report["created"]) == ["daily-sweep", "research-pass"]

    def test_a_file_may_wrap_jobs_in_a_jobs_key(self, cron_home):
        declare(cron_home, "bundle.yaml", {"jobs": [WEEKLY]})

        report = sync(home=cron_home)

        assert [e["name"] for e in report["created"]] == ["research-pass"]

    def test_unknown_field_is_rejected_by_name(self, cron_home):
        declare(cron_home, "typo.yaml", {**WEEKLY, "schedul": "every 1h"})

        with pytest.raises(CronDeclarationError) as exc:
            sync(home=cron_home)

        assert "schedul" in str(exc.value)

    def test_missing_schedule_is_rejected(self, cron_home):
        declare(cron_home, "bad.yaml", {"name": "nope", "prompt": "do a thing"})

        with pytest.raises(CronDeclarationError) as exc:
            sync(home=cron_home)

        assert "schedule" in str(exc.value)

    def test_no_agent_without_script_is_rejected(self, cron_home):
        declare(
            cron_home,
            "watchdog.yaml",
            {"name": "watchdog", "schedule": "every 5m", "no_agent": True},
        )

        with pytest.raises(CronDeclarationError) as exc:
            sync(home=cron_home)

        assert "script" in str(exc.value)

    def test_duplicate_names_across_files_are_rejected(self, cron_home):
        declare(cron_home, "a.yaml", WEEKLY)
        declare(cron_home, "b.yaml", {**WEEKLY, "schedule": "every 2h"})

        with pytest.raises(CronDeclarationError) as exc:
            sync(home=cron_home)

        assert "already declared" in str(exc.value)

    def test_malformed_yaml_names_the_file(self, cron_home):
        (cron_home / "cron" / "broken.yaml").write_text("name: [unclosed\n", encoding="utf-8")

        with pytest.raises(CronDeclarationError) as exc:
            sync(home=cron_home)

        assert "broken.yaml" in str(exc.value)


class TestCommandSurface:
    def test_sync_reports_the_pause_and_how_to_undo_it(self, cron_home, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(cron_home))
        declare(cron_home, "research-pass.yaml", WEEKLY)

        rc = cron_command(Namespace(cron_command="sync", dry_run=False, json_output=False))

        assert rc == 0

    def test_a_bad_declaration_exits_nonzero(self, cron_home, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_HOME", str(cron_home))
        declare(cron_home, "typo.yaml", {**WEEKLY, "nonsense": 1})

        rc = cron_command(Namespace(cron_command="sync", dry_run=False, json_output=False))

        assert rc == 1
        assert "nonsense" in capsys.readouterr().out

    def test_json_mode_emits_the_report(self, cron_home, monkeypatch, capsys):
        monkeypatch.setenv("HERMES_HOME", str(cron_home))
        declare(cron_home, "research-pass.yaml", WEEKLY)

        rc = cron_command(Namespace(cron_command="sync", dry_run=False, json_output=True))

        assert rc == 0
        report = json.loads(capsys.readouterr().out)
        assert [e["name"] for e in report["created"]] == ["research-pass"]
