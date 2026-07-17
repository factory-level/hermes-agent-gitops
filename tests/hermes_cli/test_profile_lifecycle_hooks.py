"""Tests for profile distribution lifecycle plugin hooks.

Verifies that ``hermes profile install`` / ``update`` fire the
``profile_install`` / ``profile_update`` / ``profile_install_failed``
plugin hooks AFTER durable install state (manifest written, payload
copied), with the documented kwargs; that a misbehaving hook callback
never breaks the install/update (kanban-hook parity); and the fail-loud
contract — a callback returning ``{"error": ..., "fatal": True}`` makes
the install/update command exit non-zero while the profile stays
installed on disk.

Mirrors tests/hermes_cli/test_kanban_lifecycle_hooks.py: patches the
plugin manager's ``_hooks`` dict directly (the registry ``invoke_hook``
reads) and monkeypatches ``discover_plugins`` to a no-op so the bundled
plugin scan never runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager
from hermes_cli.profile_distribution import (
    DistributionManifest,
    ProfileHookError,
    install_distribution,
    read_manifest,
    update_distribution,
    write_manifest,
)


# ---------------------------------------------------------------------------
# Isolated profile env (matches test_profile_distribution.py)
# ---------------------------------------------------------------------------


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


def _make_staging_dir(root: Path, name: str = "src", *, version: str = "0.1.0") -> Path:
    staged = root / f"staging_{name}"
    staged.mkdir(parents=True, exist_ok=True)
    (staged / "SOUL.md").write_text("I am Source.\n")
    write_manifest(staged, DistributionManifest(name=name, version=version))
    return staged


@pytest.fixture
def hook_registry(monkeypatch):
    """Patch the plugin manager's hook registry + no-op ``discover_plugins``.

    ``_fire_profile_lifecycle_hook`` calls ``discover_plugins()`` itself
    (the profile install path doesn't otherwise trigger discovery), so
    tests neutralize the real bundled-plugin scan the same way
    test_kanban_lifecycle_hooks.py neutralizes it implicitly by never
    calling it — here we must patch it explicitly.
    """
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda force=False: None)
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    try:
        yield mgr
    finally:
        mgr._hooks = saved


def _capture(mgr, hook_name, events):
    mgr._hooks.setdefault(hook_name, []).append(
        lambda _h=hook_name, **kw: events.append((_h, kw))
    )


# ---------------------------------------------------------------------------
# 1. Hook names registered
# ---------------------------------------------------------------------------


def test_hook_names_are_registered_as_valid():
    assert "profile_install" in VALID_HOOKS
    assert "profile_update" in VALID_HOOKS
    assert "profile_install_failed" in VALID_HOOKS


# ---------------------------------------------------------------------------
# 2. Local-dir install fires profile_install once, full payload, durable
# ---------------------------------------------------------------------------


def test_install_fires_profile_install_with_full_payload(profile_env, hook_registry):
    events = []
    _capture(hook_registry, "profile_install", events)
    staged = _make_staging_dir(profile_env, "src")

    plan = install_distribution(str(staged))

    fired = [e for e in events if e[0] == "profile_install"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["name"] == "src"
    assert kw["source_url"] == str(staged.resolve())
    assert kw["ref"] == ""
    assert kw["sha"] == ""
    assert kw["distribution_version"] == "0.1.0"
    assert kw["target_dir"] == str(plan.target_dir)
    assert kw["event"] == "install"
    # Durable-first: manifest already on disk with installed_at stamped
    # by the time the callback runs.
    on_disk = read_manifest(plan.target_dir)
    assert on_disk is not None
    assert on_disk.installed_at


def test_install_durability_observed_inside_callback(profile_env, hook_registry):
    """The callback itself can observe durable state (not just after return)."""
    seen = {}

    def _observer(**kw):
        mf = read_manifest(Path(kw["target_dir"]))
        seen["installed_at"] = mf.installed_at if mf else None

    hook_registry._hooks.setdefault("profile_install", []).append(_observer)
    staged = _make_staging_dir(profile_env, "src2")

    install_distribution(str(staged))

    assert seen.get("installed_at")


# ---------------------------------------------------------------------------
# 3. Update fires profile_update with previous_version/previous_sha
# ---------------------------------------------------------------------------


def test_update_fires_profile_update_with_previous_fields(profile_env, hook_registry):
    staged = _make_staging_dir(profile_env, "upd", version="1.0.0")
    install_distribution(str(staged))

    events = []
    _capture(hook_registry, "profile_update", events)

    # Bump the source version in place — update_distribution re-reads the
    # recorded source (the same staged dir) so this simulates a new release.
    write_manifest(staged, DistributionManifest(name="upd", version="2.0.0"))

    plan = update_distribution("upd")

    fired = [e for e in events if e[0] == "profile_update"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["previous_version"] == "1.0.0"
    assert kw["previous_sha"] == ""
    assert kw["distribution_version"] == "2.0.0"
    assert kw["event"] == "update"
    assert kw["target_dir"] == str(plan.target_dir)


# ---------------------------------------------------------------------------
# 4b. Failed update fires profile_install_failed with event="update_failed"
# ---------------------------------------------------------------------------


def test_failed_update_fires_failed_hook_with_update_failed_event(
    profile_env, hook_registry
):
    staged = _make_staging_dir(profile_env, "upd2", version="1.0.0")
    install_distribution(str(staged))

    # Corrupt the recorded source to point somewhere unreachable, simulating
    # a bogus/unreachable source on update — plan_install raises
    # DistributionError re-staging it.
    from hermes_cli.profiles import get_profile_dir

    target = get_profile_dir("upd2")
    manifest = read_manifest(target)
    manifest.source = str(profile_env / "does_not_exist_anymore")
    write_manifest(target, manifest)

    events = []
    _capture(hook_registry, "profile_update", events)
    _capture(hook_registry, "profile_install_failed", events)

    with pytest.raises(Exception):
        update_distribution("upd2")

    assert [e for e in events if e[0] == "profile_update"] == []
    failed = [e for e in events if e[0] == "profile_install_failed"]
    assert len(failed) == 1
    kw = failed[0][1]
    assert kw["error"]
    assert kw["event"] == "update_failed"


# ---------------------------------------------------------------------------
# 4. Failed install fires profile_install_failed, not profile_install
# ---------------------------------------------------------------------------


def test_failed_install_fires_failed_hook_only(profile_env, hook_registry):
    events = []
    _capture(hook_registry, "profile_install", events)
    _capture(hook_registry, "profile_install_failed", events)

    with pytest.raises(Exception):
        install_distribution(str(profile_env / "does_not_exist"))

    assert [e for e in events if e[0] == "profile_install"] == []
    failed = [e for e in events if e[0] == "profile_install_failed"]
    assert len(failed) == 1
    kw = failed[0][1]
    assert kw["name"] == ""
    assert kw["error"]
    assert kw["event"] == "install_failed"


def test_failed_install_from_reserved_name_fires_failed_hook(profile_env, hook_registry):
    """``plan_install`` raises a bare ``ValueError`` for reserved profile
    names (e.g. 'test') via ``validate_profile_name`` — not a
    ``DistributionError``. The failure wrapper must still fire
    ``profile_install_failed``, and the original ``ValueError`` must
    propagate unchanged.
    """
    events = []
    _capture(hook_registry, "profile_install", events)
    _capture(hook_registry, "profile_install_failed", events)
    staged = _make_staging_dir(profile_env, "test")

    with pytest.raises(ValueError, match="reserved"):
        install_distribution(str(staged))

    assert [e for e in events if e[0] == "profile_install"] == []
    failed = [e for e in events if e[0] == "profile_install_failed"]
    assert len(failed) == 1
    kw = failed[0][1]
    assert kw["error"]
    assert "reserved" in kw["error"]
    assert kw["event"] == "install_failed"


# ---------------------------------------------------------------------------
# 5. Raising callback does not break install (kanban parity)
# ---------------------------------------------------------------------------


def test_misbehaving_hook_does_not_break_install(profile_env, hook_registry):
    def _boom(**kw):
        raise RuntimeError("plugin exploded")

    hook_registry._hooks.setdefault("profile_install", []).append(_boom)
    staged = _make_staging_dir(profile_env, "src")

    plan = install_distribution(str(staged))

    assert plan.target_dir.is_dir()
    assert read_manifest(plan.target_dir) is not None


# ---------------------------------------------------------------------------
# 6/7. Fatal dict return -> ProfileHookError, profile stays installed,
#      and profile_install_failed is NOT additionally fired
# ---------------------------------------------------------------------------


def test_fatal_hook_return_raises_profile_hook_error_but_keeps_profile(
    profile_env, hook_registry
):
    def _fatal(**kw):
        return {"error": "gitops push failed", "fatal": True, "plugin": "gitops-emitter"}

    failed_events = []
    hook_registry._hooks.setdefault("profile_install", []).append(_fatal)
    _capture(hook_registry, "profile_install_failed", failed_events)
    staged = _make_staging_dir(profile_env, "src")

    with pytest.raises(ProfileHookError) as excinfo:
        install_distribution(str(staged))

    assert "gitops push failed" in str(excinfo.value)

    # Profile remains installed — the failure is post-install, not
    # install failure.
    from hermes_cli.profiles import get_profile_dir

    target = get_profile_dir("src")
    assert target.is_dir()
    assert read_manifest(target) is not None

    # profile_install_failed must NOT additionally fire — the install
    # itself succeeded; only the observer hook failed.
    assert failed_events == []


# ---------------------------------------------------------------------------
# 8. Non-fatal dict return is ignored
# ---------------------------------------------------------------------------


def test_non_fatal_dict_return_is_ignored(profile_env, hook_registry):
    def _warn(**kw):
        return {"error": "just a warning, no fatal key"}

    hook_registry._hooks.setdefault("profile_install", []).append(_warn)
    staged = _make_staging_dir(profile_env, "src")

    plan = install_distribution(str(staged))  # must not raise

    assert plan.target_dir.is_dir()


# ---------------------------------------------------------------------------
# 9. CLI-level: cmd_profile install with a fatal hook -> SystemExit(1)
# ---------------------------------------------------------------------------


def test_cli_install_with_fatal_hook_exits_nonzero(profile_env, hook_registry, capsys):
    from hermes_cli.main import cmd_profile
    from hermes_cli.subcommands.profile import build_profile_parser

    def _fatal(**kw):
        return {"error": "gitops push failed", "fatal": True, "plugin": "gitops-emitter"}

    hook_registry._hooks.setdefault("profile_install", []).append(_fatal)
    staged = _make_staging_dir(profile_env, "src")

    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_profile_parser(sub, cmd_profile=cmd_profile)
    ns = parser.parse_args(["profile", "install", str(staged), "--yes"])

    with pytest.raises(SystemExit) as excinfo:
        ns.func(ns)

    assert excinfo.value.code == 1

    from hermes_cli.profiles import get_profile_dir

    assert get_profile_dir("src").is_dir()


# ---------------------------------------------------------------------------
# HERMESVISOR_REQUIRE_EMITTER
# ---------------------------------------------------------------------------


def test_require_emitter_env_var_with_no_subscribers_raises(
    profile_env, hook_registry, monkeypatch
):
    monkeypatch.setenv("HERMESVISOR_REQUIRE_EMITTER", "1")
    staged = _make_staging_dir(profile_env, "src")

    with pytest.raises(ProfileHookError, match="HERMESVISOR_REQUIRE_EMITTER"):
        install_distribution(str(staged))


def test_no_env_var_and_no_subscribers_is_vanilla_success(profile_env, hook_registry):
    staged = _make_staging_dir(profile_env, "src")

    plan = install_distribution(str(staged))  # must not raise

    assert plan.target_dir.is_dir()
