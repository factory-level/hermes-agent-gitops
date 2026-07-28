"""Converge distribution-shipped cron declarations into the job store.

``hermes profile install`` copies a distribution's ``cron/`` directory onto the
profile verbatim, but the scheduler only ever reads ``cron/jobs.json`` — so a
shipped ``cron/weekly-digest.yaml`` lands on disk and never runs. Nothing in
``hermes cron`` imports one, which is why every shipped job has historically
been a dead file. This module closes that gap.

Ownership is split on purpose, because this runs unattended on every container
start:

* **sync owns the definition.** Schedule, prompt, skills, delivery. Editing a
  declaration and re-syncing updates the job in place, keeping its id and run
  history.
* **the operator owns whether it runs.** A job is created **paused**, and sync
  never writes ``enabled``/``state``/``paused_at`` again. So ``hermes cron
  resume`` is a permanent decision that survives every reboot — the alternative
  would have a redeploy silently re-arm a job somebody deliberately stopped.

Jobs created here carry ``distribution_source`` (which file declared them) and
``distribution_spec`` (the normalized declaration). The first lets sync retire
its own orphans without ever touching a hand-made job; the second makes the
"has this changed?" comparison exact rather than a guess about how the store
normalized a field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Every field a declaration may carry, each mapping onto a field the job store
# already understands. Deliberately not a superset: inventing declaration-only
# vocabulary would mean maintaining a translation layer, and there is nothing a
# declaration needs to say that `create_job` cannot already express.
DECLARATION_FIELDS = frozenset(
    {
        "name",
        "schedule",
        "prompt",
        "skill",
        "skills",
        "deliver",
        "workdir",
        "enabled_toolsets",
        "script",
        "no_agent",
        "repeat",
        "context_from",
        "model",
        "provider",
        "base_url",
        "attach_to_session",
    }
)

# `cron/` holds runtime state alongside declarations: the store itself, the
# per-job output tree, the executions database, the tick lock and the ticker
# heartbeats. Only the store shares a declaration suffix, so it is the only one
# that needs naming — but dotfiles are skipped too, since an editor swap file
# should never be read as a job.
STORE_FILENAME = "jobs.json"
DECLARATION_SUFFIXES = (".yaml", ".yml", ".json")


class CronDeclarationError(ValueError):
    """A declaration file is unreadable or does not describe a valid job."""


def declaration_dir(home: Optional[Path] = None) -> Path:
    """The directory declarations are read from, honouring the active profile."""
    if home is not None:
        return Path(home) / "cron"
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "cron"


def _iter_declaration_files(cron_dir: Path) -> List[Path]:
    if not cron_dir.is_dir():
        return []
    files = [
        p
        for p in sorted(cron_dir.iterdir())
        if p.is_file()
        and not p.name.startswith(".")
        and p.name != STORE_FILENAME
        and p.suffix.lower() in DECLARATION_SUFFIXES
    ]
    return files


def _load_declaration_file(path: Path) -> List[Dict[str, Any]]:
    """Read one file into a list of raw declarations.

    A file may hold a single job mapping, a list of them, or a ``jobs:`` key
    wrapping a list — all three read naturally, and picking one arbitrarily
    would just make authors look it up.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CronDeclarationError(f"{path.name}: cannot read ({exc})") from exc

    try:
        if path.suffix.lower() == ".json":
            data = json.loads(text) if text.strip() else None
        else:
            data = yaml.safe_load(text)
    except (ValueError, yaml.YAMLError) as exc:
        raise CronDeclarationError(f"{path.name}: not valid {path.suffix.lstrip('.')} ({exc})") from exc

    if data is None:
        return []
    if isinstance(data, dict) and "jobs" in data and len(data) == 1:
        data = data["jobs"]
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                raise CronDeclarationError(
                    f"{path.name}: every entry must be a mapping, got {type(entry).__name__}"
                )
        return list(data)
    raise CronDeclarationError(
        f"{path.name}: expected a job mapping or a list of them, got {type(data).__name__}"
    )


def _normalize_declaration(raw: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Validate one declaration and return it in canonical form.

    Unknown keys are rejected rather than ignored: a typo'd ``schedul:`` that
    silently produced a job with no schedule would be far worse to debug than a
    failed sync that names the key.
    """
    unknown = sorted(set(raw) - DECLARATION_FIELDS)
    if unknown:
        raise CronDeclarationError(
            f"{source}: unknown field(s) {', '.join(unknown)} — "
            f"allowed: {', '.join(sorted(DECLARATION_FIELDS))}"
        )

    name = str(raw.get("name") or "").strip()
    if not name:
        raise CronDeclarationError(f"{source}: 'name' is required — it is how sync identifies the job")

    schedule = raw.get("schedule")
    if not isinstance(schedule, str) or not schedule.strip():
        raise CronDeclarationError(
            f"{source}: '{name}' needs a 'schedule' string "
            "(e.g. 'every 2h', '0 6 * * 1', '2026-06-01T09:00:00Z')"
        )

    no_agent = bool(raw.get("no_agent") or False)
    prompt = raw.get("prompt")
    prompt = str(prompt) if prompt is not None else None
    script = raw.get("script")
    script = str(script).strip() if isinstance(script, str) else None
    if no_agent and not script:
        raise CronDeclarationError(
            f"{source}: '{name}' sets no_agent but declares no script — "
            "with no agent and no script there is nothing to run"
        )
    if not no_agent and not (prompt and prompt.strip()) and not (raw.get("skills") or raw.get("skill")):
        raise CronDeclarationError(
            f"{source}: '{name}' needs a 'prompt' (or at least a skill to run) — "
            "a cron prompt must be self-contained, since the job runs with no chat history"
        )

    skills = raw.get("skills")
    if skills is None and raw.get("skill"):
        skills = [raw["skill"]]
    if skills is not None:
        if isinstance(skills, str):
            skills = [skills]
        if not isinstance(skills, list) or not all(isinstance(s, str) for s in skills):
            raise CronDeclarationError(f"{source}: '{name}' skills must be a list of strings")
        skills = [s.strip() for s in skills if s and s.strip()] or None

    toolsets = raw.get("enabled_toolsets")
    if toolsets is not None:
        if not isinstance(toolsets, list) or not all(isinstance(t, str) for t in toolsets):
            raise CronDeclarationError(
                f"{source}: '{name}' enabled_toolsets must be a list of strings"
            )
        toolsets = [t.strip() for t in toolsets if t and t.strip()] or None

    repeat = raw.get("repeat")
    if repeat is not None:
        if isinstance(repeat, bool) or not isinstance(repeat, int):
            raise CronDeclarationError(f"{source}: '{name}' repeat must be an integer")

    context_from = raw.get("context_from")
    if isinstance(context_from, str):
        context_from = [context_from]
    if context_from is not None and (
        not isinstance(context_from, list) or not all(isinstance(c, str) for c in context_from)
    ):
        raise CronDeclarationError(
            f"{source}: '{name}' context_from must be a job name/id or a list of them"
        )

    spec: Dict[str, Any] = {
        "name": name,
        "schedule": schedule.strip(),
        "prompt": prompt.strip() if isinstance(prompt, str) else None,
        "skills": skills,
        # A distribution job has no originating conversation to reply into, so
        # "local" is the only sane default — "origin" would resolve to nothing.
        "deliver": (str(raw["deliver"]).strip() if raw.get("deliver") else "local"),
        "workdir": (str(raw["workdir"]).strip() if raw.get("workdir") else None),
        "enabled_toolsets": toolsets,
        "script": script,
        "no_agent": no_agent,
        "repeat": repeat,
        "context_from": context_from,
        "model": (str(raw["model"]).strip() if raw.get("model") else None),
        "provider": (str(raw["provider"]).strip() if raw.get("provider") else None),
        "base_url": (str(raw["base_url"]).strip() if raw.get("base_url") else None),
        "attach_to_session": (
            bool(raw["attach_to_session"]) if isinstance(raw.get("attach_to_session"), bool) else None
        ),
    }
    return spec


def load_declarations(home: Optional[Path] = None) -> List[Tuple[str, Dict[str, Any]]]:
    """Read every declaration under ``cron/``, as ``(source filename, spec)``.

    Raises ``CronDeclarationError`` on the first bad file, naming it — a sync
    that half-applied a batch would leave the store in a state nobody declared.
    """
    declarations: List[Tuple[str, Dict[str, Any]]] = []
    seen: Dict[str, str] = {}
    for path in _iter_declaration_files(declaration_dir(home)):
        for raw in _load_declaration_file(path):
            spec = _normalize_declaration(raw, path.name)
            previous = seen.get(spec["name"])
            if previous:
                raise CronDeclarationError(
                    f"{path.name}: job name '{spec['name']}' is already declared in {previous} — "
                    "names identify jobs, so they have to be unique"
                )
            seen[spec["name"]] = path.name
            declarations.append((path.name, spec))
    return declarations


def _create_kwargs(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt": spec["prompt"],
        "schedule": spec["schedule"],
        "name": spec["name"],
        "repeat": spec["repeat"],
        "deliver": spec["deliver"],
        "skills": spec["skills"],
        "model": spec["model"],
        "provider": spec["provider"],
        "base_url": spec["base_url"],
        "script": spec["script"],
        "context_from": spec["context_from"],
        "enabled_toolsets": spec["enabled_toolsets"],
        "workdir": spec["workdir"],
        "no_agent": spec["no_agent"],
        "attach_to_session": spec["attach_to_session"],
    }


def _update_payload(spec: Dict[str, Any], existing: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Field updates that bring ``existing`` in line with ``spec``.

    ``enabled``/``state``/``paused_at`` are conspicuously absent: whether a job
    runs is the operator's call, and re-asserting it here would undo a
    deliberate pause on the next redeploy.
    """
    completed = 0
    if isinstance(existing.get("repeat"), dict):
        completed = existing["repeat"].get("completed") or 0
    return {
        "name": spec["name"],
        "prompt": spec["prompt"] or "",
        "skills": spec["skills"],
        "deliver": spec["deliver"],
        "workdir": spec["workdir"],
        "enabled_toolsets": spec["enabled_toolsets"],
        "script": spec["script"],
        "no_agent": spec["no_agent"],
        "context_from": spec["context_from"],
        "model": spec["model"],
        "provider": spec["provider"],
        "base_url": spec["base_url"],
        "repeat": {"times": spec["repeat"], "completed": completed},
        "schedule": spec["schedule"],
        "distribution_source": source,
        "distribution_spec": spec,
    }


def sync(dry_run: bool = False, home: Optional[Path] = None) -> Dict[str, Any]:
    """Converge declared cron jobs into the store.

    Returns a report of what changed, so both the CLI and a caller driving this
    from outside the container can act on it without re-reading the store.
    """
    from cron.jobs import create_job, load_jobs, remove_job, update_job

    declarations = load_declarations(home)
    declared_names = {spec["name"] for _, spec in declarations}

    existing_by_name: Dict[str, Dict[str, Any]] = {}
    orphans: List[Dict[str, Any]] = []
    for job in load_jobs():
        if not job.get("distribution_source"):
            continue  # hand-made; never ours to touch
        name = (job.get("name") or "").strip()
        if name in declared_names:
            existing_by_name[name] = job
        else:
            orphans.append(job)

    report: Dict[str, Any] = {
        "created": [],
        "updated": [],
        "unchanged": [],
        "pruned": [],
        "dry_run": bool(dry_run),
    }

    for source, spec in declarations:
        name = spec["name"]
        existing = existing_by_name.get(name)
        if existing is None:
            report["created"].append({"name": name, "source": source, "schedule": spec["schedule"]})
            if dry_run:
                continue
            job = create_job(**_create_kwargs(spec))
            # Paused on arrival, and stamped so later syncs recognise it as ours.
            update_job(
                job["id"],
                {
                    "enabled": False,
                    "state": "paused",
                    "paused_reason": f"declared by {source}; resume to schedule it",
                    "distribution_source": source,
                    "distribution_spec": spec,
                },
            )
            continue

        if existing.get("distribution_spec") == spec and existing.get("distribution_source") == source:
            report["unchanged"].append({"name": name, "source": source, "id": existing["id"]})
            continue

        report["updated"].append({"name": name, "source": source, "id": existing["id"]})
        if not dry_run:
            update_job(existing["id"], _update_payload(spec, existing, source))

    for job in orphans:
        report["pruned"].append(
            {
                "name": job.get("name"),
                "source": job.get("distribution_source"),
                "id": job.get("id"),
            }
        )
        if not dry_run:
            remove_job(job["id"])

    return report
