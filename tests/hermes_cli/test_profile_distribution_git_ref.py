"""Tests for git ``#<ref>`` pin plumbing in hermes_cli.profile_distribution.

Covers: splitting ``url#ref`` sources, ``_git_clone`` resolving a tag /
branch / full commit SHA to the right commit, and ``installed_ref`` /
``installed_sha`` being threaded through ``plan_install`` /
``install_distribution`` / ``update_distribution`` and persisted in the
on-disk manifest.

test_profile_distribution.py carries a policy comment that transport-layer
tests (git clone, URL handling) are E2E-only, not unit tests — the point
being "don't mock git and assert on the mock." This module doesn't: every
test here drives the real ``git`` binary against a real (local-filesystem)
bare repository, so cloning, ref resolution and sha capture are genuinely
exercised end-to-end. It's hermetic (no network, no external service) and
fast, so it belongs in the regular unit-test run rather than a live E2E
suite.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hermes_cli.profile_distribution import (
    DistributionError,
    DistributionManifest,
    MANIFEST_FILENAME,
    _git_clone,
    _looks_like_git_url,
    _split_source_ref,
    _stage_source,
    install_distribution,
    read_manifest,
    update_distribution,
    write_manifest,
    _parse_fragment,
    _normalize_subdir,
    _canonical_provenance,
    plan_install,
)


# ---------------------------------------------------------------------------
# Isolated profile env (matches test_profile_distribution.py / AGENTS.md)
# ---------------------------------------------------------------------------


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


# ---------------------------------------------------------------------------
# Bare-repo fixture
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _make_bare_repo(
    tmp_path: Path, name: str = "origin", subdir: str = ""
) -> SimpleNamespace:
    """Build a local bare git repo with two commits, a branch, and a tag.

    Layout:
      * commit 1 — a minimal distribution (SOUL.md + distribution.yaml),
        tagged ``v1``.
      * commit 2 — bumps SOUL.md; the ``main`` branch tip moves past the
        tag, so a tag pin, a branch pin, and a full-sha pin each resolve
        to a genuinely different commit.

    Returns a namespace with ``bare`` (the bare repo path used as the
    install source), ``work`` (a live worktree clone origin tests can push
    further commits from, e.g. to simulate a moved branch tip), and the
    resolved shas for each commit.
    """
    bare = tmp_path / f"{name}.git"
    subprocess.run(
        ["git", "init", "--bare", "-b", "main", str(bare)],
        check=True, capture_output=True,
    )

    work = tmp_path / f"{name}_work"
    subprocess.run(["git", "clone", str(bare), str(work)], check=True, capture_output=True)
    _git("config", "user.email", "test@example.com", cwd=work)
    _git("config", "user.name", "Test", cwd=work)

    payload_root = work / subdir if subdir else work
    payload_root.mkdir(parents=True, exist_ok=True)
    (payload_root / "SOUL.md").write_text("I am v1.\n")
    write_manifest(payload_root, DistributionManifest(name="gitdist", version="1.0.0"))
    _git("add", "-A", cwd=work)
    _git("commit", "-m", "v1", cwd=work)
    _git("push", "origin", "main", cwd=work)
    v1_sha = _git("rev-parse", "HEAD", cwd=work)
    _git("tag", "v1", cwd=work)
    _git("push", "origin", "v1", cwd=work)

    (payload_root / "SOUL.md").write_text("I am v2.\n")
    _git("add", "-A", cwd=work)
    _git("commit", "-m", "v2", cwd=work)
    _git("push", "origin", "main", cwd=work)
    tip_sha = _git("rev-parse", "HEAD", cwd=work)

    assert v1_sha != tip_sha, "fixture bug: tag and branch tip must differ"

    return SimpleNamespace(bare=bare, work=work, v1_sha=v1_sha, tip_sha=tip_sha)


# ===========================================================================
# _split_source_ref
# ===========================================================================


class TestSplitSourceRef:

    @pytest.mark.parametrize("source,expected", [
        ("github.com/user/repo", ("github.com/user/repo", "")),
        ("https://github.com/user/repo.git", ("https://github.com/user/repo.git", "")),
        ("https://github.com/user/repo.git#v1", ("https://github.com/user/repo.git", "v1")),
        ("git@github.com:user/repo.git#main", ("git@github.com:user/repo.git", "main")),
        (
            "https://github.com/user/repo.git#abc123def4567890abc123def4567890abc123d",
            ("https://github.com/user/repo.git", "abc123def4567890abc123def4567890abc123d"),
        ),
        ("/local/path/without/fragment", ("/local/path/without/fragment", "")),
    ])
    def test_split_variants(self, source, expected):
        assert _split_source_ref(source) == expected

    def test_only_trailing_fragment_is_split(self):
        # rsplit("#", 1): only the LAST '#' separates the ref; anything
        # earlier stays part of the url.
        assert _split_source_ref("https://x/y#a#b") == ("https://x/y#a", "b")


class TestLooksLikeGitUrlStaysFragmentFree:
    """`_looks_like_git_url` itself must not understand '#<ref>' — splitting
    is `_stage_source`'s job, applied before this check runs.

    Only the shorthand-regex and `.git`-suffix branches are fragment-
    sensitive (a bare fragment breaks the regex / suffix match); the
    `http(s)://` branch matches on prefix alone, so it's indifferent to a
    trailing fragment either way — not exercised here since it isn't a
    case that could regress by forgetting to split first.
    """

    @pytest.mark.parametrize("src", [
        "github.com/user/repo#v1",
        "some/local/path.git#deadbeef",
    ])
    def test_fragment_bearing_sources_rejected(self, src):
        assert not _looks_like_git_url(src)


# ===========================================================================
# _git_clone — ref resolution
# ===========================================================================


class TestGitCloneRefResolution:

    def test_no_ref_resolves_default_branch_head(self, tmp_path):
        info = _make_bare_repo(tmp_path)
        dest = tmp_path / "dest_noref"
        sha = _git_clone(str(info.bare), dest)
        assert sha == info.tip_sha
        assert (dest / "SOUL.md").read_text() == "I am v2.\n"

    def test_tag_ref_resolves_tag_commit(self, tmp_path):
        info = _make_bare_repo(tmp_path)
        dest = tmp_path / "dest_tag"
        sha = _git_clone(str(info.bare), dest, ref="v1")
        assert sha == info.v1_sha
        assert (dest / "SOUL.md").read_text() == "I am v1.\n"

    def test_branch_ref_resolves_branch_tip(self, tmp_path):
        info = _make_bare_repo(tmp_path)
        dest = tmp_path / "dest_branch"
        sha = _git_clone(str(info.bare), dest, ref="main")
        assert sha == info.tip_sha
        assert (dest / "SOUL.md").read_text() == "I am v2.\n"

    def test_full_sha_ref_resolves_via_fallback_checkout(self, tmp_path):
        # A shallow `--branch <sha>` clone fails for an arbitrary commit
        # SHA (not a branch/tag name) — this must fall back to a full
        # clone + `checkout --detach`.
        info = _make_bare_repo(tmp_path)
        dest = tmp_path / "dest_sha"
        sha = _git_clone(str(info.bare), dest, ref=info.v1_sha)
        assert sha == info.v1_sha
        assert (dest / "SOUL.md").read_text() == "I am v1.\n"

    def test_unknown_ref_raises_distribution_error(self, tmp_path):
        info = _make_bare_repo(tmp_path)
        dest = tmp_path / "dest_bad"
        with pytest.raises(DistributionError, match="git clone failed"):
            _git_clone(str(info.bare), dest, ref="does-not-exist-ref")


# ===========================================================================
# _stage_source — ref/sha threading
# ===========================================================================


class TestStageSourceGitRef:

    def test_no_ref_records_head_sha_and_empty_ref(self, tmp_path):
        info = _make_bare_repo(tmp_path)
        staged, provenance, ref, sha, _subdir = _stage_source(str(info.bare), tmp_path / "work1")
        assert provenance == str(info.bare)
        assert ref == ""
        assert sha == info.tip_sha
        assert (staged / MANIFEST_FILENAME).is_file()

    def test_tag_ref_keeps_full_url_hash_ref_provenance(self, tmp_path):
        info = _make_bare_repo(tmp_path)
        source = f"{info.bare}#v1"
        staged, provenance, ref, sha, _subdir = _stage_source(source, tmp_path / "work2")
        assert provenance == source  # url#ref preserved verbatim for `update`
        assert ref == "v1"
        assert sha == info.v1_sha

    def test_local_dir_source_has_empty_ref_and_sha(self, tmp_path):
        staged_src = tmp_path / "localsrc"
        staged_src.mkdir()
        write_manifest(staged_src, DistributionManifest(name="local"))
        staged, provenance, ref, sha, _subdir = _stage_source(str(staged_src), tmp_path / "work3")
        assert ref == ""
        assert sha == ""
        assert provenance == str(staged_src.resolve())

    def test_local_dir_source_with_no_fragment_untouched(self, tmp_path):
        # Sanity check the "local path with no fragment" case explicitly:
        # a plain local path must resolve exactly as before.
        staged_src = tmp_path / "plainlocal"
        staged_src.mkdir()
        write_manifest(staged_src, DistributionManifest(name="plain"))
        staged, provenance, ref, sha, _subdir = _stage_source(str(staged_src), tmp_path / "work4")
        assert staged == staged_src.resolve()
        assert ref == "" and sha == ""


# ===========================================================================
# install_distribution — installed_ref / installed_sha persistence
# ===========================================================================


class TestInstallRecordsRefSha:

    def test_no_ref_install_records_head_sha(self, profile_env, tmp_path):
        info = _make_bare_repo(tmp_path)
        plan = install_distribution(str(info.bare), name="gitdist")
        assert plan.ref == ""
        assert plan.sha == info.tip_sha

        mf = read_manifest(plan.target_dir)
        assert mf.installed_ref == ""
        assert mf.installed_sha == info.tip_sha

    def test_tag_pin_install_persists_ref_and_sha(self, profile_env, tmp_path):
        info = _make_bare_repo(tmp_path)
        source = f"{info.bare}#v1"
        plan = install_distribution(source, name="gitdist")
        assert plan.ref == "v1"
        assert plan.sha == info.v1_sha

        mf = read_manifest(plan.target_dir)
        assert mf.installed_ref == "v1"
        assert mf.installed_sha == info.v1_sha
        assert mf.source == source  # provenance keeps url#ref for update

    def test_branch_pin_install_persists_ref_and_sha(self, profile_env, tmp_path):
        info = _make_bare_repo(tmp_path)
        source = f"{info.bare}#main"
        plan = install_distribution(source, name="gitdist")
        mf = read_manifest(plan.target_dir)
        assert mf.installed_ref == "main"
        assert mf.installed_sha == info.tip_sha

    def test_full_sha_pin_install_persists_ref_and_sha(self, profile_env, tmp_path):
        info = _make_bare_repo(tmp_path)
        source = f"{info.bare}#{info.v1_sha}"
        plan = install_distribution(source, name="gitdist")
        mf = read_manifest(plan.target_dir)
        assert mf.installed_ref == info.v1_sha
        assert mf.installed_sha == info.v1_sha

    def test_local_dir_install_manifest_omits_ref_sha_keys(self, profile_env, tmp_path):
        """Vanilla (local-dir) installs must be byte-unchanged: no
        installed_ref/installed_sha keys at all, not even empty ones."""
        staged_src = tmp_path / "localsrc2"
        staged_src.mkdir()
        (staged_src / "SOUL.md").write_text("local\n")
        write_manifest(staged_src, DistributionManifest(name="localdist"))

        plan = install_distribution(str(staged_src), name="localdist")
        assert plan.ref == "" and plan.sha == ""

        raw = yaml.safe_load((plan.target_dir / MANIFEST_FILENAME).read_text())
        assert "installed_ref" not in raw
        assert "installed_sha" not in raw

        mf = read_manifest(plan.target_dir)
        assert mf.installed_ref == ""
        assert mf.installed_sha == ""


# ===========================================================================
# update_distribution — branch pins re-resolve to the moved tip
# ===========================================================================


class TestUpdateRePinsRef:

    def test_update_on_branch_pin_resolves_to_moved_tip(self, profile_env, tmp_path):
        info = _make_bare_repo(tmp_path)
        source = f"{info.bare}#main"
        plan = install_distribution(source, name="gitdist")
        mf = read_manifest(plan.target_dir)
        assert mf.installed_ref == "main"
        assert mf.installed_sha == info.tip_sha

        # Advance the branch tip past what was installed.
        _git("checkout", "main", cwd=info.work)
        (info.work / "SOUL.md").write_text("I am v3.\n")
        _git("add", "-A", cwd=info.work)
        _git("commit", "-m", "v3", cwd=info.work)
        _git("push", "origin", "main", cwd=info.work)
        new_tip = _git("rev-parse", "HEAD", cwd=info.work)
        assert new_tip != info.tip_sha

        update_distribution("gitdist")
        mf2 = read_manifest(plan.target_dir)
        assert mf2.installed_ref == "main"
        assert mf2.installed_sha == new_tip
        assert (plan.target_dir / "SOUL.md").read_text() == "I am v3.\n"

    def test_update_on_tag_pin_stays_fixed(self, profile_env, tmp_path):
        info = _make_bare_repo(tmp_path)
        source = f"{info.bare}#v1"
        plan = install_distribution(source, name="gitdist")

        # Advance the branch tip; the tag pin must NOT move.
        _git("checkout", "main", cwd=info.work)
        (info.work / "SOUL.md").write_text("I am v3.\n")
        _git("add", "-A", cwd=info.work)
        _git("commit", "-m", "v3", cwd=info.work)
        _git("push", "origin", "main", cwd=info.work)

        update_distribution("gitdist")
        mf2 = read_manifest(plan.target_dir)
        assert mf2.installed_ref == "v1"
        assert mf2.installed_sha == info.v1_sha
        assert (plan.target_dir / "SOUL.md").read_text() == "I am v1.\n"


# ===========================================================================
# DistributionManifest — installed_ref / installed_sha round-trip
# ===========================================================================


class TestManifestRefShaRoundtrip:

    def test_roundtrip_with_ref_sha(self, tmp_path):
        original = DistributionManifest(
            name="rt", installed_ref="v1", installed_sha="a" * 40,
        )
        write_manifest(tmp_path, original)
        raw = yaml.safe_load((tmp_path / MANIFEST_FILENAME).read_text())
        assert raw["installed_ref"] == "v1"
        assert raw["installed_sha"] == "a" * 40

        parsed = read_manifest(tmp_path)
        assert parsed.installed_ref == "v1"
        assert parsed.installed_sha == "a" * 40

    def test_roundtrip_without_ref_sha_omits_keys(self, tmp_path):
        original = DistributionManifest(name="rt2")
        write_manifest(tmp_path, original)
        raw = yaml.safe_load((tmp_path / MANIFEST_FILENAME).read_text())
        assert "installed_ref" not in raw
        assert "installed_sha" not in raw

        parsed = read_manifest(tmp_path)
        assert parsed.installed_ref == ""
        assert parsed.installed_sha == ""

    def test_from_dict_defaults_missing_keys_to_empty(self):
        m = DistributionManifest.from_dict({"name": "x"})
        assert m.installed_ref == ""
        assert m.installed_sha == ""


# ===========================================================================
# Subdirectory distributions (source fragment + --subdir flag)
# ===========================================================================


class TestParseFragment:
    @pytest.mark.parametrize(
        ("fragment", "expected"),
        [
            ("", ("", "")),
            ("v1", ("v1", "")),
            ("subdirectory=.hermes-dist", ("", ".hermes-dist")),
            ("v1&subdirectory=agents/bot", ("v1", "agents/bot")),
            ("a&b", ("a&b", "")),  # ref with a literal '&' survives
            ("a&b&subdirectory=x", ("a&b", "x")),
        ],
    )
    def test_shapes(self, fragment, expected):
        assert _parse_fragment(fragment) == expected

    def test_empty_subdirectory_value_rejected(self):
        with pytest.raises(DistributionError, match="Empty subdirectory="):
            _parse_fragment("v1&subdirectory=")

    def test_duplicate_subdirectory_params_rejected(self):
        with pytest.raises(DistributionError, match="Multiple subdirectory="):
            _parse_fragment("v1&subdirectory=a&subdirectory=b")


class TestNormalizeSubdir:
    @pytest.mark.parametrize(
        ("raw", "clean"),
        [
            ("agents/bot", "agents/bot"),
            ("./x/", "x"),
            (".hermes-dist", ".hermes-dist"),
            ("a/./b", "a/b"),
        ],
    )
    def test_accepts_and_cleans(self, raw, clean):
        assert _normalize_subdir(raw) == clean

    @pytest.mark.parametrize("raw", ["/abs", "../x", "a/../b", "a\\b", "", "."])
    def test_rejects(self, raw):
        with pytest.raises(DistributionError, match="Invalid distribution subdirectory"):
            _normalize_subdir(raw)


class TestStageSourceSubdir:
    SUB = "agents/bot"

    def test_fragment_subdir_at_tag(self, tmp_path):
        info = _make_bare_repo(tmp_path, subdir=self.SUB)
        source = f"{info.bare}#v1&subdirectory={self.SUB}"
        staged, provenance, ref, sha, subdir = _stage_source(source, tmp_path / "w1")
        assert (staged / MANIFEST_FILENAME).is_file()
        assert staged.name == "bot"
        assert (ref, sha, subdir) == ("v1", info.v1_sha, self.SUB)
        assert provenance == source

    def test_fragment_subdir_no_ref(self, tmp_path):
        info = _make_bare_repo(tmp_path, subdir=self.SUB)
        source = f"{info.bare}#subdirectory={self.SUB}"
        staged, provenance, ref, sha, subdir = _stage_source(source, tmp_path / "w2")
        assert (staged / MANIFEST_FILENAME).is_file()
        assert (ref, sha, subdir) == ("", info.tip_sha, self.SUB)
        assert provenance == source

    def test_flag_subdir_canonicalizes_provenance(self, tmp_path):
        info = _make_bare_repo(tmp_path, subdir=self.SUB)
        staged, provenance, ref, sha, subdir = _stage_source(
            str(info.bare), tmp_path / "w3", subdir=self.SUB
        )
        assert (staged / MANIFEST_FILENAME).is_file()
        assert provenance == f"{info.bare}#subdirectory={self.SUB}"

    def test_flag_plus_tag_canonicalizes_provenance(self, tmp_path):
        info = _make_bare_repo(tmp_path, subdir=self.SUB)
        staged, provenance, ref, sha, subdir = _stage_source(
            f"{info.bare}#v1", tmp_path / "w4", subdir=self.SUB
        )
        assert provenance == f"{info.bare}#v1&subdirectory={self.SUB}"
        assert sha == info.v1_sha

    def test_flag_vs_fragment_conflict_fails_naming_both(self, tmp_path):
        info = _make_bare_repo(tmp_path, subdir=self.SUB)
        with pytest.raises(DistributionError, match="Conflicting subdirectories"):
            _stage_source(
                f"{info.bare}#subdirectory={self.SUB}", tmp_path / "w5", subdir="other"
            )

    def test_flag_equal_to_fragment_ok(self, tmp_path):
        info = _make_bare_repo(tmp_path, subdir=self.SUB)
        _, provenance, _, _, subdir = _stage_source(
            f"{info.bare}#subdirectory={self.SUB}", tmp_path / "w6", subdir=self.SUB
        )
        assert subdir == self.SUB

    def test_missing_subdir_names_it(self, tmp_path):
        info = _make_bare_repo(tmp_path)
        with pytest.raises(DistributionError, match="does not exist"):
            _stage_source(
                f"{info.bare}#subdirectory=nope/where", tmp_path / "w7"
            )

    def test_subdir_without_manifest_names_exact_path(self, tmp_path):
        info = _make_bare_repo(tmp_path)  # manifest at ROOT, not in sub
        # Create a real (manifest-less) subdir in a new commit
        (info.work / "empty-sub").mkdir()
        (info.work / "empty-sub" / "keep.txt").write_text("x\n")
        _git("add", "-A", cwd=info.work)
        _git("commit", "-m", "sub", cwd=info.work)
        _git("push", "origin", "main", cwd=info.work)
        with pytest.raises(
            DistributionError, match=r"empty-sub/distribution\.yaml"
        ):
            _stage_source(f"{info.bare}#subdirectory=empty-sub", tmp_path / "w8")

    @pytest.mark.parametrize("bad", ["../x", "/abs", "a\\b"])
    def test_escape_rejected_before_clone(self, tmp_path, bad):
        workdir = tmp_path / "w9"
        with pytest.raises(DistributionError, match="Invalid distribution subdirectory"):
            _stage_source(
                f"https://example.invalid/repo.git#subdirectory={bad}", workdir
            )
        assert not (workdir / "clone").exists(), "validation must precede the clone"


class TestUpdateRePinsSubdir:
    SUB = "agents/bot"

    def test_branch_pinned_subdir_follows_moved_tip(self, tmp_path, profile_env):
        info = _make_bare_repo(tmp_path, subdir=self.SUB)
        source = f"{info.bare}#main&subdirectory={self.SUB}"
        plan = install_distribution(source, name="subdist")
        assert plan.manifest.source == source
        assert plan.subdir == self.SUB

        # Move the branch tip: bump the nested manifest version.
        payload = info.work / self.SUB
        write_manifest(
            payload, DistributionManifest(name="gitdist", version="2.0.0")
        )
        _git("add", "-A", cwd=info.work)
        _git("commit", "-m", "v2 manifest", cwd=info.work)
        _git("push", "origin", "main", cwd=info.work)
        moved_sha = _git("rev-parse", "HEAD", cwd=info.work)

        updated = update_distribution("subdist")
        assert updated.manifest.version == "2.0.0"
        assert updated.sha == moved_sha
        assert updated.manifest.source == source  # provenance unchanged
        assert updated.subdir == self.SUB

    def test_tag_pinned_subdir_stays_fixed(self, tmp_path, profile_env):
        info = _make_bare_repo(tmp_path, subdir=self.SUB)
        source = f"{info.bare}#v1&subdirectory={self.SUB}"
        plan = install_distribution(source, name="subdist2")
        assert plan.sha == info.v1_sha
        updated = update_distribution("subdist2")
        assert updated.sha == info.v1_sha
        assert updated.manifest.source == source
