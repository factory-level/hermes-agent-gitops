"""Profile distributions — shareable, packaged Hermes profiles via git.

A distribution is a Hermes profile published as a git repository (or
installed from a local directory for development). Install with one command
from a git URL, update in place, and keep your local memories / sessions /
credentials untouched.

Where this fits relative to the existing pieces:

* ``hermes profile export/import`` — local backup / restore for a profile
  on your own machine. NOT a distribution format. Stays as-is.
* ``hermes skills install <url>`` — the URL install pattern we're mirroring,
  but at the profile granularity.

Subcommands (all live under ``hermes profile``, not a parallel tree):

    hermes profile install <source> [--name N] [--alias] [--force] [--yes]
    hermes profile update  <name>  [--force-config] [--yes]
    hermes profile info    <name>

``<source>`` is one of:

* A git URL (``github.com/user/repo``, ``https://github.com/...``, ``git@...``,
  ``ssh://``, ``git://``), optionally with ``#<ref>`` to pin a tag / branch /
  commit SHA. The resolved ref and commit SHA are recorded in the installed
  manifest as ``installed_ref`` / ``installed_sha`` (see below); ``hermes
  profile update`` re-pulls that same ``#<ref>`` — a branch pin re-resolves
  to the branch's current tip, while a tag or commit-SHA pin stays fixed.
* A local directory that already contains ``distribution.yaml`` — used
  during profile development before the first push. Local-dir installs
  have no ref/sha to record; ``installed_ref`` / ``installed_sha`` are
  left empty (and omitted from the written manifest entirely).

Manifest format (``distribution.yaml`` at the profile root)::

    name: telemetry
    version: 0.1.0
    description: "Compliance monitoring harness"
    hermes_requires: ">=0.12.0"
    author: "..."
    license: "..."
    env_requires:
      - name: OPENAI_API_KEY
        description: "OpenAI API key"
        required: true
      - name: GRAPHITI_MCP_URL
        description: "Memory graph URL"
        required: false
        default: "http://127.0.0.1:8000/sse"
    distribution_owned:      # optional; sensible defaults apply
      - SOUL.md
      - skills/
      - cron/
      - mcp.json

Any other top-level key is a vendor extension block (e.g. a downstream
tool's ``deployment:`` / ``reach:`` config) — unrecognized by this
dataclass, but preserved verbatim through install / update rather than
being dropped. See ``DistributionManifest.extras``.

**Reserved names**: The top-level key ``extras`` is reserved for internal
use and may not appear literally in a source distribution.yaml; attempting
to use it will raise a ``DistributionError``.

Update semantics:

* Distribution-owned paths (SOUL.md, mcp.json, skills/, cron/,
  distribution.yaml) are replaced from the new source.
* ``config.yaml`` is distribution-owned but preserved on update unless
  ``--force-config`` is passed (user overrides typically live here).
* User-owned paths (memories/, sessions/, state.db, auth.json, .env,
  logs/, workspace/, home/, plans/, *_cache/, and anything under
  ``local/``) are never touched.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict, FrozenSet, List, Optional, Tuple

from agent.skill_utils import is_excluded_skill_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_FILENAME = "distribution.yaml"
ENV_TEMPLATE_FILENAME = ".env.template"
ENV_EXAMPLE_FILENAME = ".env.EXAMPLE"

# Default distribution-owned paths (relative to profile root).  Authors may
# override via ``distribution_owned:`` in the manifest.  config.yaml is
# distribution-owned but treated specially on update (see _is_config_like).
DEFAULT_DIST_OWNED: Tuple[str, ...] = (
    "SOUL.md",
    "config.yaml",
    "mcp.json",
    "skills",
    "cron",
    MANIFEST_FILENAME,
)

# Paths that are NEVER part of a distribution. These are user-owned and are
# protected on update. Must stay consistent with
# ``profiles.py::_DEFAULT_EXPORT_EXCLUDE_ROOT`` plus the ``local/``
# convention for user customizations.
USER_OWNED_EXCLUDE: frozenset = frozenset({
    # Credentials & runtime secrets
    "auth.json", ".env",
    # Databases & runtime state
    "state.db", "state.db-shm", "state.db-wal",
    "hermes_state.db", "response_store.db",
    "response_store.db-shm", "response_store.db-wal",
    "gateway.pid", "gateway_state.json", "processes.json",
    "auth.lock", "active_profile", ".update_check",
    "errors.log", ".hermes_history",
    # User data
    "memories", "sessions", "logs", "plans", "workspace", "home",
    "image_cache", "audio_cache", "document_cache",
    "browser_screenshots", "checkpoints", "sandboxes",
    "backups", "cache",
    # Infrastructure
    "hermes-agent", ".worktrees", "profiles", "bin", "node_modules",
    # User customization namespace
    "local",
})


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DistributionError(Exception):
    """Raised for distribution install/update failures."""


class ProfileHookError(DistributionError):
    """Raised when a strict profile lifecycle hook demands fail-loud exit.

    Fired either by a callback returning ``{"error": ..., "fatal": True}``
    (a subscriber, e.g. the gitops-emitter plugin, hit an unrecoverable
    error pushing the install record) or by the ``HERMESVISOR_REQUIRE_EMITTER``
    guard finding no subscriber at all. In both cases the profile itself is
    already durably installed on disk — only the CLI's exit code reflects
    the hook failure (see ``cmd_profile``'s existing
    ``except (DistributionError, ValueError)`` handler, which this subclass
    routes through unchanged).
    """


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class EnvRequirement:
    name: str
    description: str = ""
    required: bool = True
    default: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Any) -> "EnvRequirement":
        if not isinstance(data, dict):
            raise DistributionError(
                f"env_requires entry must be a mapping, got {type(data).__name__}"
            )
        name = str(data.get("name") or "").strip()
        if not name:
            raise DistributionError("env_requires entry missing 'name'")
        return cls(
            name=name,
            description=str(data.get("description") or ""),
            required=bool(data.get("required", True)),
            default=data.get("default"),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "description": self.description}
        if not self.required:
            out["required"] = False
        if self.default is not None:
            out["default"] = self.default
        return out


@dataclass
class DistributionManifest:
    name: str
    version: str = "0.1.0"
    description: str = ""
    hermes_requires: str = ""
    author: str = ""
    license: str = ""
    env_requires: List[EnvRequirement] = field(default_factory=list)
    distribution_owned: List[str] = field(default_factory=list)
    # Tracked after install — where we pulled from, so ``update`` can re-pull.
    source: str = ""
    # ISO-8601 UTC timestamp written on install / update, so ``info`` and
    # ``list`` can show when a distribution landed on disk.  Empty for
    # manifests that ship in a repo (authors don't populate this).
    installed_at: str = ""
    # The ``#<ref>`` pin (tag / branch / commit SHA) and the exact commit
    # SHA it resolved to, both captured at install/update time.  Empty for
    # a git source with no ``#<ref>`` (installed_sha is still populated —
    # it's HEAD's sha) and for local-directory sources (both empty).
    installed_ref: str = ""
    installed_sha: str = ""
    # Unknown top-level keys from the source distribution.yaml — vendor
    # extension blocks (e.g. HermesVisor's ``deployment:`` / ``reach:`` /
    # ``workloads:`` / ``expose:``, but generically ANY key an author adds
    # that this dataclass doesn't know about). Captured verbatim in
    # from_dict() and re-emitted as-is in to_dict() so install/update never
    # silently drops them. Never populated by hand — always derived from
    # the parsed manifest.
    extras: Dict[str, Any] = field(default_factory=dict)

    # Every field name this dataclass understands — anything else in a
    # parsed manifest's top-level mapping falls through to ``extras``.
    # ClassVar so dataclass doesn't treat this as an instance field.
    _KNOWN_KEYS: ClassVar[FrozenSet[str]] = frozenset({
        "name", "version", "description", "hermes_requires", "author",
        "license", "env_requires", "distribution_owned", "source",
        "installed_at", "installed_ref", "installed_sha", "extras",
    })

    @classmethod
    def from_dict(cls, data: Any) -> "DistributionManifest":
        if not isinstance(data, dict):
            raise DistributionError(
                f"{MANIFEST_FILENAME} must be a mapping, got {type(data).__name__}"
            )
        # Check for reserved top-level key that would silently drop data
        if "extras" in data:
            raise DistributionError(
                "distribution.yaml may not use the reserved top-level key 'extras'; "
                "rename it to avoid data loss"
            )
        name = str(data.get("name") or "").strip()
        if not name:
            raise DistributionError(f"{MANIFEST_FILENAME} missing 'name'")
        env_raw = data.get("env_requires") or []
        if not isinstance(env_raw, list):
            raise DistributionError("env_requires must be a list")
        env_requires = [EnvRequirement.from_dict(e) for e in env_raw]
        dist_owned_raw = data.get("distribution_owned") or []
        if dist_owned_raw and not isinstance(dist_owned_raw, list):
            raise DistributionError("distribution_owned must be a list")
        distribution_owned = [str(p).strip().strip("/") for p in dist_owned_raw if str(p).strip()]
        extras = {k: v for k, v in data.items() if k not in cls._KNOWN_KEYS}
        return cls(
            name=name,
            version=str(data.get("version") or "0.1.0"),
            description=str(data.get("description") or ""),
            hermes_requires=str(data.get("hermes_requires") or ""),
            author=str(data.get("author") or ""),
            license=str(data.get("license") or ""),
            env_requires=env_requires,
            distribution_owned=distribution_owned,
            source=str(data.get("source") or ""),
            installed_at=str(data.get("installed_at") or ""),
            installed_ref=str(data.get("installed_ref") or ""),
            installed_sha=str(data.get("installed_sha") or ""),
            extras=extras,
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name,
            "version": self.version,
        }
        if self.description:
            out["description"] = self.description
        if self.hermes_requires:
            out["hermes_requires"] = self.hermes_requires
        if self.author:
            out["author"] = self.author
        if self.license:
            out["license"] = self.license
        if self.env_requires:
            out["env_requires"] = [e.to_dict() for e in self.env_requires]
        if self.distribution_owned:
            out["distribution_owned"] = self.distribution_owned
        if self.source:
            out["source"] = self.source
        if self.installed_at:
            out["installed_at"] = self.installed_at
        if self.installed_ref:
            out["installed_ref"] = self.installed_ref
        if self.installed_sha:
            out["installed_sha"] = self.installed_sha
        for key, value in self.extras.items():
            out[key] = value
        return out

    def owned_paths(self) -> List[str]:
        """Resolve which paths count as distribution-owned."""
        if self.distribution_owned:
            return list(self.distribution_owned)
        return list(DEFAULT_DIST_OWNED)


def _load_yaml(text: str) -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover — pyyaml is a hard dep
        raise DistributionError("PyYAML is required for distribution manifests") from exc
    return yaml.safe_load(text)


def _dump_yaml(data: Any) -> str:
    import yaml

    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def read_manifest(profile_dir: Path) -> Optional[DistributionManifest]:
    """Return the manifest for *profile_dir*, or None if it isn't a distribution."""
    mf_path = profile_dir / MANIFEST_FILENAME
    if not mf_path.is_file():
        return None
    try:
        data = _load_yaml(mf_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DistributionError(f"Failed to parse {mf_path}: {exc}") from exc
    return DistributionManifest.from_dict(data or {})


def write_manifest(profile_dir: Path, manifest: DistributionManifest) -> Path:
    mf_path = profile_dir / MANIFEST_FILENAME
    mf_path.write_text(_dump_yaml(manifest.to_dict()), encoding="utf-8")
    return mf_path


# ---------------------------------------------------------------------------
# Version check
# ---------------------------------------------------------------------------


_VERSION_OP_RE = re.compile(r"^\s*(>=|<=|==|!=|>|<)\s*(.+?)\s*$")


def _parse_semver(v: str) -> Tuple[int, int, int]:
    """Very small semver parser — major.minor.patch only.  Extra labels stripped."""
    s = str(v).strip().lstrip("v")
    # Strip any pre-release / build metadata (e.g. "0.12.0-rc1+abc")
    s = re.split(r"[-+]", s, 1)[0]
    parts = s.split(".")
    while len(parts) < 3:
        parts.append("0")
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as exc:
        raise DistributionError(f"Unparseable version: {v!r}") from exc


def check_hermes_requires(spec: str, current_version: str) -> None:
    """Raise DistributionError if ``current_version`` does not satisfy ``spec``.

    ``spec`` accepts a single comparator (``>=0.12.0``, ``==0.12.0``, etc.).
    Empty or blank spec is a no-op — no requirement.
    """
    if not spec or not spec.strip():
        return
    m = _VERSION_OP_RE.match(spec)
    if not m:
        # Bare version → treat as ``>=``
        op, target = ">=", spec.strip()
    else:
        op, target = m.group(1), m.group(2)
    cur = _parse_semver(current_version)
    tgt = _parse_semver(target)
    ok = {
        ">=": cur >= tgt,
        "<=": cur <= tgt,
        "==": cur == tgt,
        "!=": cur != tgt,
        ">":  cur > tgt,
        "<":  cur < tgt,
    }[op]
    if not ok:
        raise DistributionError(
            f"This distribution requires Hermes {op}{target}, "
            f"but you have {current_version}."
        )


# ---------------------------------------------------------------------------
# Env var template helper
# ---------------------------------------------------------------------------


def _env_template_from_manifest(manifest: DistributionManifest) -> str:
    """Generate a ``.env.template`` body from env_requires."""
    lines = [
        "# Environment variables required by this Hermes distribution.",
        "# Copy to `.env` and fill in your own values before running.",
        "",
    ]
    for req in manifest.env_requires:
        if req.description:
            lines.append(f"# {req.description}")
        status = "required" if req.required else "optional"
        lines.append(f"# ({status})")
        default_val = req.default if req.default is not None else ""
        prefix = "" if req.required else "# "
        lines.append(f"{prefix}{req.name}={default_val}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Source staging — git clone or local directory
# ---------------------------------------------------------------------------


def _looks_like_git_url(s: str) -> bool:
    s = s.strip()
    if s.endswith(".git"):
        return True
    if s.startswith(("git@", "ssh://", "git://")):
        return True
    if s.startswith(("http://", "https://")):
        # Any http(s) URL is treated as a git repo.  We no longer accept
        # tar.gz URLs — git is the only remote transport.
        return True
    # Bare github.com/user/repo shorthand
    if re.match(r"^github\.com/[\w.-]+/[\w.-]+/?$", s):
        return True
    return False


def _split_source_ref(source: str) -> Tuple[str, str]:
    """Split a trailing ``#<ref>`` off *source*.

    Returns ``(url, ref)`` — ``ref`` is ``""`` when *source* has no
    fragment.  Uses ``rsplit`` with ``maxsplit=1`` so only the final ``#``
    is treated as the ref separator; an earlier literal ``#`` in the URL
    itself is left alone.

    Must run BEFORE ``_looks_like_git_url`` — that check knows nothing
    about fragments (``github.com/u/r#v1`` fails its shorthand regex,
    ``path.git#ref`` fails the ``.git``-suffix check) and is meant to stay
    that way; ref-splitting happens one layer up, in ``_stage_source``.
    """
    if "#" not in source:
        return source, ""
    url, ref = source.rsplit("#", 1)
    return url, ref


def _git_clone(url: str, dest: Path, ref: str = "") -> str:
    """Clone *url* into *dest*, optionally pinned to *ref*.

    Returns the resolved commit SHA (captured before the caller strips
    ``.git``).

    * No ``ref`` — plain shallow clone of the default branch.
    * ``ref`` given — try a shallow clone of that ref first (works for
      both branches and tags). If that fails (``ref`` is a commit SHA,
      which shallow clone can't target directly over most transports),
      fall back to a full clone followed by a detached checkout of
      ``ref`` — this covers arbitrary commit SHAs portably.
    """
    # Normalize github.com/user/repo shorthand
    if re.match(r"^github\.com/[\w.-]+/[\w.-]+/?$", url):
        url = f"https://{url.rstrip('/')}"
    try:
        if not ref:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(dest)],
                check=True,
                capture_output=True,
            )
        else:
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", "--branch", ref, url, str(dest)],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError:
                shutil.rmtree(dest, ignore_errors=True)
                subprocess.run(
                    ["git", "clone", url, str(dest)],
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    ["git", "-C", str(dest), "checkout", "--detach", ref],
                    check=True,
                    capture_output=True,
                )
        rev = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return rev.stdout.strip()
    except FileNotFoundError as exc:
        raise DistributionError("git is required for git-URL installs") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise DistributionError(f"git clone failed: {stderr.strip()}") from exc


def _stage_source(source: str, workdir: Path) -> Tuple[Path, str, str, str]:
    """Resolve *source* to a local directory containing distribution.yaml.

    Returns ``(staged_dir, provenance, ref, sha)``:

    * ``provenance`` is stored verbatim (including any ``#<ref>``) in the
      installed manifest's ``source:`` field so ``hermes profile update``
      re-pulls from the same place — and re-resolves the same pin (a
      branch pin follows the branch; a tag / commit-SHA pin stays fixed).
    * ``ref`` / ``sha`` are the resolved git ref and commit SHA. Both are
      ``""`` for local-directory sources, which have no ref pin at all.

    Accepts:
      * A git URL (https / ssh / git@ / bare github.com shorthand),
        optionally suffixed with ``#<ref>`` — cloned into a temp
        directory; ``.git`` removed after clone.
      * A local directory already containing ``distribution.yaml``.
    """
    src_str = source.strip()
    url_part, ref = _split_source_ref(src_str)

    # Git URL — ref-split url_part is what _looks_like_git_url evaluates,
    # per the fragment-free contract documented on _split_source_ref.
    if _looks_like_git_url(url_part):
        cloned = workdir / "clone"
        sha = _git_clone(url_part, cloned, ref=ref)
        # Remove .git to keep the staged tree clean
        shutil.rmtree(cloned / ".git", ignore_errors=True)
        if not (cloned / MANIFEST_FILENAME).is_file():
            raise DistributionError(
                f"No {MANIFEST_FILENAME} at the root of {src_str!r}. "
                "This repository is not a Hermes profile distribution."
            )
        return cloned, src_str, ref, sha

    # Local directory — use the ORIGINAL (unsplit) string. Local sources
    # don't support ref pins; a literal '#' in a local path is just part
    # of the path, not a fragment to strip.
    path_guess = Path(src_str).expanduser()
    if path_guess.is_dir():
        if not (path_guess / MANIFEST_FILENAME).is_file():
            raise DistributionError(
                f"No {MANIFEST_FILENAME} in {path_guess}. "
                "A local-directory source must contain a distribution.yaml at its root."
            )
        return path_guess.resolve(), str(path_guess.resolve()), "", ""

    raise DistributionError(
        f"Cannot resolve distribution source: {source!r}. "
        "Expected a git URL (e.g. github.com/user/repo) or a local directory."
    )


def _reject_distribution_symlinks(staged: Path) -> None:
    """Reject symlinks before reading or copying distribution files."""
    for entry in staged.rglob("*"):
        if not entry.is_symlink():
            continue
        try:
            rel = entry.relative_to(staged)
        except ValueError:
            rel = entry
        raise DistributionError(
            f"Profile distributions cannot contain symlinks: {rel}"
        )


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


@dataclass
class InstallPlan:
    """Summary of what an install will do, surfaced for user confirmation."""
    manifest: DistributionManifest
    staged_dir: Path
    provenance: str
    target_dir: Path
    existing: bool  # True if target profile already exists (update path)
    preserves_config: bool = True
    has_cron: bool = False
    has_skills: bool = False
    ref: str = ""
    sha: str = ""


def _has_cron_jobs(staged: Path) -> bool:
    cron_dir = staged / "cron"
    if not cron_dir.is_dir():
        return False
    for _ in cron_dir.rglob("*.json"):
        return True
    for _ in cron_dir.rglob("*.yaml"):
        return True
    return False


def _count_skills(staged: Path) -> int:
    skills_dir = staged / "skills"
    if not skills_dir.is_dir():
        return 0
    return sum(
        1 for p in skills_dir.rglob("SKILL.md") if not is_excluded_skill_path(p)
    )


def plan_install(
    source: str,
    workdir: Path,
    override_name: Optional[str] = None,
) -> InstallPlan:
    """Stage *source* and produce a plan describing what install would do."""
    from hermes_cli.profiles import (
        get_profile_dir,
        normalize_profile_name,
        validate_profile_name,
    )
    from hermes_cli import __version__ as hermes_version

    staged, provenance, ref, sha = _stage_source(source, workdir)
    _reject_distribution_symlinks(staged)
    manifest = read_manifest(staged)
    if manifest is None:
        raise DistributionError(
            f"No {MANIFEST_FILENAME} found at the distribution root — "
            "this source is not a Hermes distribution."
        )

    # Version check up-front so we fail fast
    check_hermes_requires(manifest.hermes_requires, hermes_version)

    # Resolve target profile name
    target_name = override_name or manifest.name
    canon = normalize_profile_name(target_name)
    validate_profile_name(canon)
    if canon == "default":
        raise DistributionError(
            "Cannot install a distribution as 'default' — that is the built-in "
            "root profile (~/.hermes).  Pass --name <name> to install under a "
            "new profile."
        )
    manifest.name = canon
    manifest.source = provenance
    manifest.installed_ref = ref
    manifest.installed_sha = sha
    # Stamped once here so plan_install() callers (both fresh install and
    # update) propagate a freshly-minted timestamp through _copy_dist_payload.
    manifest.installed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    target_dir = get_profile_dir(canon)
    existing = target_dir.is_dir()
    has_cron = _has_cron_jobs(staged)
    skill_count = _count_skills(staged)

    return InstallPlan(
        manifest=manifest,
        staged_dir=staged,
        provenance=provenance,
        target_dir=target_dir,
        existing=existing,
        preserves_config=existing,
        has_cron=has_cron,
        has_skills=skill_count > 0,
        ref=ref,
        sha=sha,
    )


def _copy_dist_payload(
    staged: Path,
    target: Path,
    manifest: DistributionManifest,
    preserve_config: bool,
) -> None:
    """Copy distribution-owned files from *staged* into *target*.

    User-owned paths are never touched.  ``config.yaml`` is replaced only when
    ``preserve_config`` is False (fresh install or ``--force-config`` update).
    ``.env.template`` is renamed to ``.env.EXAMPLE`` in the target to avoid
    shadowing a real ``.env``.
    """
    target.mkdir(parents=True, exist_ok=True)

    for entry in staged.iterdir():
        name = entry.name

        if name in USER_OWNED_EXCLUDE:
            continue
        if name == ENV_TEMPLATE_FILENAME:
            shutil.copy2(entry, target / ENV_EXAMPLE_FILENAME)
            continue
        if name == "config.yaml" and preserve_config and (target / "config.yaml").exists():
            # Leave user's config.yaml alone on update
            continue

        dest = target / name
        if entry.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            staged_resolved = staged.resolve()
            shutil.copytree(
                entry,
                dest,
                ignore=lambda d, names: (
                    [n for n in names if n in USER_OWNED_EXCLUDE]
                    if Path(d).resolve() == staged_resolved
                    else []
                ),
            )
        else:
            shutil.copy2(entry, dest)

    # Emit .env.EXAMPLE from manifest if the staged tree didn't ship one
    if manifest.env_requires and not (target / ENV_EXAMPLE_FILENAME).exists():
        (target / ENV_EXAMPLE_FILENAME).write_text(
            _env_template_from_manifest(manifest), encoding="utf-8"
        )

    # Make sure the manifest on disk reflects resolved name + source
    write_manifest(target, manifest)


def _bootstrap_user_dirs(target: Path) -> None:
    """Create the bootstrap dirs a fresh profile expects."""
    for d in ("memories", "sessions", "skills", "skins", "logs",
              "plans", "workspace", "cron", "home"):
        (target / d).mkdir(parents=True, exist_ok=True)


def _fragment_free_source_url(provenance: str, ref: str) -> str:
    """Strip a trailing ``#<ref>`` from *provenance* for hook payloads.

    ``provenance`` (``InstallPlan.provenance`` / the manifest's ``source:``
    field) is stored verbatim, fragment included, so ``update`` re-resolves
    the same pin. Hook payloads want the two separated (``source_url`` +
    ``ref``). Only strips when *ref* is non-empty — local-directory sources
    never have a ref (``ref`` is always ``""`` for them), so a literal
    ``#`` in a local path is never touched.
    """
    if ref and provenance.endswith(f"#{ref}"):
        return provenance[: -(len(ref) + 1)]
    return provenance


def _fire_profile_lifecycle_hook(hook_name: str, *, strict: bool, **fields: Any) -> None:
    """Fire a profile lifecycle plugin hook, best-effort at the infra layer.

    Called by install/update AFTER durable install state (manifest written,
    payload copied to the target profile dir) so a plugin callback always
    observes a profile that's already on disk. Plugin discovery does not
    run on the profile install/update path otherwise, so this fires it
    itself (idempotent — a no-op once already discovered).

    Any infra failure (plugins unavailable, discovery error) is swallowed —
    a misbehaving or absent plugin subsystem must never block an install.

    When *strict* is True (``profile_install`` / ``profile_update``), a
    callback may opt the CLI into fail-loud behavior by returning
    ``{"error": str, "fatal": True, "plugin": str}`` — see the
    ``profile_install`` entry in ``VALID_HOOKS`` for the full contract.
    ``profile_install_failed`` is always fired with ``strict=False``: it's
    observer-only, so a failing push there can't also fail the failure
    report.
    """
    try:
        from hermes_cli.plugins import discover_plugins, has_hook, invoke_hook

        discover_plugins()
        results = invoke_hook(hook_name, **fields)
    except Exception as exc:  # infra failure = best-effort, never blocks the install
        logger.debug("profile lifecycle hook %s failed: %s", hook_name, exc)
        return

    if not strict:
        return

    if (
        os.environ.get("HERMESVISOR_REQUIRE_EMITTER") == "1"
        and not results
        and not has_hook(hook_name)
    ):
        raise ProfileHookError(
            f"HERMESVISOR_REQUIRE_EMITTER=1 but no plugin is subscribed to {hook_name}"
        )

    fatal = [r for r in results if isinstance(r, dict) and r.get("fatal") and r.get("error")]
    if fatal:
        raise ProfileHookError(
            "Profile was installed, but a post-install hook failed: "
            + "; ".join(f"[{f.get('plugin', '?')}] {f['error']}" for f in fatal)
        )


def install_distribution(
    source: str,
    name: Optional[str] = None,
    force: bool = False,
    create_alias: bool = False,
) -> InstallPlan:
    """Install a distribution from *source* into a new profile.

    Returns the resolved :class:`InstallPlan`.  Use :func:`plan_install`
    first if you want to preview + prompt the user before calling this.

    Fires the ``profile_install`` lifecycle hook after the install is
    durable on disk. A subscriber (e.g. HermesVisor's gitops-emitter
    plugin) may return a fatal error dict to make this raise
    :class:`ProfileHookError` — the profile stays installed either way.
    On any other :class:`DistributionError` (install itself failed) — or a
    bare :class:`ValueError` from ``plan_install``'s profile-name validation
    (e.g. a reserved name) — fires ``profile_install_failed`` instead.
    """
    from hermes_cli.profiles import (
        check_alias_collision,
        create_wrapper_script,
    )

    plan: Optional[InstallPlan] = None
    try:
        with tempfile.TemporaryDirectory(prefix="hermes_dist_install_") as tmp:
            plan = plan_install(source, Path(tmp), override_name=name)

            if plan.existing and not force:
                raise DistributionError(
                    f"Profile '{plan.manifest.name}' already exists at {plan.target_dir}. "
                    "Use `hermes profile update` to upgrade in place, "
                    "or pass --force to overwrite."
                )

            # Fresh install: config.yaml comes from the distribution.
            _bootstrap_user_dirs(plan.target_dir)
            _copy_dist_payload(
                plan.staged_dir,
                plan.target_dir,
                plan.manifest,
                preserve_config=False,
            )

            if create_alias:
                collision = check_alias_collision(plan.manifest.name)
                if collision is None:
                    create_wrapper_script(plan.manifest.name)

            _fire_profile_lifecycle_hook(
                "profile_install",
                strict=True,
                name=plan.manifest.name,
                source_url=_fragment_free_source_url(plan.provenance, plan.ref),
                ref=plan.ref,
                sha=plan.sha,
                distribution_version=plan.manifest.version,
                target_dir=str(plan.target_dir),
                event="install",
            )
            return plan
    except (DistributionError, ValueError) as e:
        if not isinstance(e, ProfileHookError):
            _fire_profile_lifecycle_hook(
                "profile_install_failed",
                strict=False,
                name=plan.manifest.name if plan is not None else "",
                source_url=(
                    _fragment_free_source_url(plan.provenance, plan.ref)
                    if plan is not None
                    else source
                ),
                ref=plan.ref if plan is not None else "",
                error=str(e),
                event="install_failed",
            )
        raise


def update_distribution(
    profile_name: str,
    force_config: bool = False,
) -> InstallPlan:
    """Re-pull the distribution for an existing profile and apply updates.

    The source is read from the installed profile's ``distribution.yaml``
    ``source:`` field.  Distribution-owned files are overwritten; user-owned
    data (memories, sessions, auth) is never touched.  ``config.yaml`` is
    preserved unless ``force_config`` is True.

    Fires the ``profile_update`` lifecycle hook after the update is durable
    on disk (see :func:`install_distribution` for the fail-loud contract).
    On any other :class:`DistributionError` — or a bare :class:`ValueError`
    from ``plan_install``'s profile-name validation (e.g. a reserved name) —
    fires ``profile_install_failed`` (event ``"update_failed"``) instead.
    """
    from hermes_cli.profiles import (
        get_profile_dir,
        normalize_profile_name,
        validate_profile_name,
    )

    canon = normalize_profile_name(profile_name)
    validate_profile_name(canon)
    target = get_profile_dir(canon)
    if not target.is_dir():
        raise DistributionError(f"Profile '{canon}' does not exist.")

    existing_manifest = read_manifest(target)
    if existing_manifest is None:
        raise DistributionError(
            f"Profile '{canon}' is not a distribution (no {MANIFEST_FILENAME}). "
            "Only profiles installed via `hermes profile install` can be updated."
        )
    if not existing_manifest.source:
        raise DistributionError(
            f"Profile '{canon}' has no recorded source.  Re-install with "
            "`hermes profile install <source> --name {canon} --force`."
        )

    # Captured BEFORE re-staging — plan_install below overwrites plan.manifest
    # with the newly-staged version's fields.
    previous_version = existing_manifest.version
    previous_sha = existing_manifest.installed_sha

    plan: Optional[InstallPlan] = None
    try:
        with tempfile.TemporaryDirectory(prefix="hermes_dist_update_") as tmp:
            plan = plan_install(
                existing_manifest.source,
                Path(tmp),
                override_name=canon,
            )
            plan.preserves_config = not force_config

            _copy_dist_payload(
                plan.staged_dir,
                plan.target_dir,
                plan.manifest,
                preserve_config=plan.preserves_config,
            )

            _fire_profile_lifecycle_hook(
                "profile_update",
                strict=True,
                name=plan.manifest.name,
                source_url=_fragment_free_source_url(plan.provenance, plan.ref),
                ref=plan.ref,
                sha=plan.sha,
                distribution_version=plan.manifest.version,
                target_dir=str(plan.target_dir),
                event="update",
                previous_version=previous_version,
                previous_sha=previous_sha,
            )
            return plan
    except (DistributionError, ValueError) as e:
        if not isinstance(e, ProfileHookError):
            _fire_profile_lifecycle_hook(
                "profile_install_failed",
                strict=False,
                name=plan.manifest.name if plan is not None else canon,
                source_url=(
                    _fragment_free_source_url(plan.provenance, plan.ref)
                    if plan is not None
                    else existing_manifest.source
                ),
                ref=plan.ref if plan is not None else "",
                error=str(e),
                event="update_failed",
            )
        raise


# ---------------------------------------------------------------------------
# Info — render a manifest summary
# ---------------------------------------------------------------------------


def describe_distribution(profile_name: str) -> Dict[str, Any]:
    """Return a structured view of a profile's distribution metadata.

    Returns an empty dict if the profile exists but has no manifest.
    Raises DistributionError if the profile itself doesn't exist.
    """
    from hermes_cli.profiles import (
        get_profile_dir,
        normalize_profile_name,
        validate_profile_name,
    )

    canon = normalize_profile_name(profile_name)
    validate_profile_name(canon)
    target = get_profile_dir(canon)
    if not target.is_dir():
        raise DistributionError(f"Profile '{canon}' does not exist.")
    manifest = read_manifest(target)
    if manifest is None:
        return {}
    return manifest.to_dict()
