"""Projects: a real folder tree the agent's output lives in.

Stage 3 answered a question and dropped a chart in ``plots/``. That is enough to
answer one question and not enough to hand anything to anybody else: the files are
unrelated to each other, nothing records which download produced them, and there
is no way to send a colleague "the three days you asked about" as one artefact.

A project is a directory laid out by what the data *is*::

    /projects/<name>/
        manifest.json
        <station>/
            <date>/
                manifest.json
                raw.csv
                derived.csv
                *.html

The layout is the point. Because the station and the date are directories, a
comparison is a glob rather than a naming convention, two runs cannot collide by
accident, and the ZIP is meaningful to whoever opens it -- they see the same tree
the agent built.

Every leaf carries a ``manifest.json`` because a CSV on its own is not evidence of
anything: it does not say which observatory it came from, whether the values were
definitive or quasi-definite, when they were downloaded, or which commit of the
agent wrote them. Those are exactly the four questions asked of a result a year
later, so they are written next to the data rather than remembered.
"""

from __future__ import annotations

import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "PROJECTS_ROOT",
    "EXPORTS_ROOT",
    "Workspace",
    "agent_commit",
    "utc_now",
    "create_project",
    "list_projects",
    "parse_slot",
    "project_dir",
    "project_manifest",
    "read_project_manifest",
    "artifact_dir",
    "save_artifact",
    "write_manifest",
    "export_project",
    "project_tree",
    "tree_markdown",
]

BASE_DIR = Path(__file__).resolve().parent
PROJECTS_ROOT = BASE_DIR / "projects"
EXPORTS_ROOT = BASE_DIR / "exports"

#: Files a manifest may name but the tree should never show as browsable data.
_MANIFEST = "manifest.json"

_COMMIT_CACHE: str | None = None


def utc_now() -> str:
    """ISO-8601 UTC, second resolution, always suffixed so it sorts as UTC."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def agent_commit() -> str | None:
    """The commit this checkout is at, or ``None`` outside a git working tree.

    Recorded in every manifest because the numbers are only reproducible with the
    code that made them: the same query answered before and after a change to the
    analyser is two different results. Cached because a manifest write should not
    pay for a subprocess, and the answer cannot change while the process lives.
    """
    global _COMMIT_CACHE
    if _COMMIT_CACHE is not None:
        return _COMMIT_CACHE or None
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        _COMMIT_CACHE = ""
        return None
    _COMMIT_CACHE = done.stdout.strip() if done.returncode == 0 else ""
    return _COMMIT_CACHE or None


# --------------------------------------------------------------------------- #
# names
# --------------------------------------------------------------------------- #
def _slug(name: Any) -> str:
    """A folder name that cannot escape the projects root.

    The name reaches here from a language model, so it is treated as untrusted
    input rather than a label: ``../..`` and absolute paths are rejected outright
    instead of being sanitised into something that looks safe. Restricting the
    alphabet also keeps the tree readable and the ZIP safe to extract on Windows.
    """
    text = str(name or "").strip().lower()
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in text)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned.strip("-")


def project_dir(name: str, root: str | Path | None = None) -> Path | None:
    """The absolute directory for ``name``, or ``None`` if the name is unusable.

    ``None`` rather than a sentinel path: ``Path("")`` stringifies to ``"."``,
    which is truthy, so an empty name would slip past an ``if not str(...)`` test
    and end up writing a manifest into the current working directory.
    """
    base = Path(root) if root is not None else PROJECTS_ROOT
    slug = _slug(name)
    if not slug:
        return None
    target = (base / slug).resolve()
    # Belt and braces: even after the alphabet filter, confirm the result is a
    # direct child of the root. Cheap, and it makes the invariant local.
    if target.parent != base.resolve():
        return None
    return target


def list_projects(root: str | Path | None = None) -> list[str]:
    base = Path(root) if root is not None else PROJECTS_ROOT
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def _error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    payload = {"ok": False, "error": code, "message": message}
    payload.update(extra)
    return payload


def is_error(value: Any) -> bool:
    return isinstance(value, dict) and value.get("ok") is False


# --------------------------------------------------------------------------- #
# manifests
# --------------------------------------------------------------------------- #
def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write ``payload`` as pretty UTF-8 JSON, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


def read_project_manifest(name: str, root: str | Path | None = None) -> dict[str, Any]:
    """The project's own manifest, or an error payload if it has none."""
    target = project_dir(name, root)
    if target is None:
        return _error(
            "invalid_project_name",
            "A project name must contain letters, digits, '-' or '_'.",
            requested=name,
        )
    manifest = target / _MANIFEST
    if not manifest.is_file():
        return _error(
            "project_not_found",
            f"No project {target.name!r}. Known projects: {', '.join(list_projects(root)) or '(none)'}",
            available=list_projects(root),
        )
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _error("unreadable_manifest", f"{type(exc).__name__}: {exc}", path=str(manifest))


def project_manifest(
    name: str,
    stations: Sequence[str],
    dates: Sequence[str],
    artifacts: Sequence[dict[str, Any]] | None = None,
    root: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The project-level manifest.

    One place that answers "what is in here and when was it collected". The
    station/date pairs are stored sorted and de-duplicated because the same window
    fetched twice is one window of this project, and a manifest that listed it
    twice would misstate the size of the job.
    """
    manifest = {
        "project": _slug(name),
        "stations": sorted({_slug(s) for s in stations if _slug(s)}),
        "dates": sorted({str(d) for d in dates if d}),
        "artifacts": list(artifacts or []),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "agent_commit": agent_commit(),
    }
    if extra:
        manifest.update(extra)
    return manifest


# --------------------------------------------------------------------------- #
# creating and filling
# --------------------------------------------------------------------------- #
def create_project(
    name: str,
    stations: Sequence[str] | None = None,
    dates: Sequence[str] | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Create ``/projects/<name>/`` and its manifest.

    Idempotent on purpose. A model that calls create_project twice -- which it
    will, having no memory of its own calls -- must not be told the second call
    failed, because "already exists" is not a problem the model can solve and
    being told so just makes it try a third name and lose the earlier one.
    """
    target = project_dir(name, root)
    if target is None:
        return _error(
            "invalid_project_name",
            "A project name must be non-empty and use only letters, digits, '-' or '_'.",
            requested=name,
        )
    existed = (target / _MANIFEST).is_file()
    target.mkdir(parents=True, exist_ok=True)
    if not existed:
        write_manifest(
            target / _MANIFEST,
            project_manifest(target.name, stations or [], dates or [], root=root),
        )
    return {
        "ok": True,
        "project": target.name,
        "path": str(target),
        "created": not existed,
        "layout": "<station>/<date>/",
        "note": (
            f"Project {target.name!r} already exists; reusing it."
            if existed
            else f"Created project {target.name!r}."
        ),
    }


def artifact_dir(
    project: str,
    station: str,
    day: str,
    root: str | Path | None = None,
) -> Path | None:
    """``/projects/<project>/<station>/<date>/``, or ``None`` if a name is unusable."""
    project_path = project_dir(project, root)
    station_slug = _slug(station)
    day_text = str(day or "").strip()
    if not project_path or not station_slug or not day_text:
        return None
    # The date is a directory name too, so it gets the same treatment as a
    # station: a model must not be able to write outside its own project.
    day_slug = _slug(day_text)
    if not day_slug:
        return None
    return project_path / station_slug / day_slug


def save_artifact(
    project: str,
    station: str,
    day: str,
    frame: Any,
    kind: str = "raw",
    data_version: str | None = None,
    downloaded_at: str | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Write one DataFrame as CSV plus a manifest, and return what was written.

    The manifest names the *kind* of artifact separately from the data version,
    because those answer different questions: ``kind`` says what this file is
    (raw, or H/D/I computed from it), while ``data_version`` says what the
    observatory published. Collapsing them loses the ability to tell a re-derivation
    from a re-download.
    """
    target = artifact_dir(project, station, day, root)
    if target is None:
        return _error(
            "invalid_artifact_path",
            "A project, station and date are all required to save an artifact.",
            project=project, station=station, date=day,
        )
    target.mkdir(parents=True, exist_ok=True)
    filename = f"{_slug(kind) or 'data'}.csv"
    csv_path = target / filename
    try:
        frame.to_csv(csv_path, index=False)
    except Exception as exc:  # pragma: no cover - defensive, mirrors the loop
        return _error("artifact_write_failed", f"{type(exc).__name__}: {exc}",
                      path=str(csv_path))

    attrs = getattr(frame, "attrs", None) or {}
    payload = {
        "project": _slug(project),
        "station": _slug(station),
        "date": str(day),
        "kind": _slug(kind),
        "file": filename,
        "rows": int(len(frame)),
        "columns": [str(c) for c in getattr(frame, "columns", [])],
        # What the observatory published, and where it came from. Both are the
        # difference between a number and a traceable number.
        "data_version": data_version or attrs.get("publication_state"),
        "source": attrs.get("source"),
        # When it was written, and by which commit -- the reproduction pair.
        "downloaded_at": downloaded_at or attrs.get("downloaded_at") or utc_now(),
        "agent_commit": agent_commit(),
        "written_at": utc_now(),
    }
    write_manifest(target / _MANIFEST, payload)
    return {"ok": True, **payload, "path": str(csv_path), "dir": str(target)}


# --------------------------------------------------------------------------- #
# reading and exporting
# --------------------------------------------------------------------------- #
def _human(size: int) -> str:
    step = 1024.0
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < step or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= step
    return f"{size} B"


def project_tree(name: str, root: str | Path | None = None) -> list[dict[str, Any]]:
    """A read-only listing of a project: every file with its size.

    Sorted so the folder order is stable between calls, which matters for a UI
    that re-renders this on every selection change -- an unstable order reads as
    the app shuffling files under the user.
    """
    target = project_dir(name, root)
    if target is None or not target.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(target.rglob("*")):
        relative = path.relative_to(target).as_posix()
        if path.is_dir():
            rows.append({"path": relative + "/", "kind": "dir", "size": "", "bytes": 0})
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        rows.append(
            {
                "path": relative,
                "kind": "manifest" if path.name == _MANIFEST else path.suffix.lstrip(".") or "file",
                "size": _human(size),
                "bytes": size,
            }
        )
    return rows


def tree_markdown(name: str, root: str | Path | None = None) -> str:
    """``project_tree`` as a markdown listing, for the chat panel."""
    rows = project_tree(name, root)
    if not rows:
        return ""
    lines = [f"**{name}**", "```"]
    for row in rows:
        lines.append(f"{row['path']:<52} {row['size']:>9}")
    lines.append("```")
    return "\n".join(lines)


def export_project(
    name: str,
    root: str | Path | None = None,
    exports: str | Path | None = None,
) -> dict[str, Any]:
    """Zip a whole project and return where the archive is.

    Stored rather than streamed so the manifest can be written *inside* the tree
    first and then be picked up by the walk -- the archive's inventory and the
    project's own description are the same list, which cannot drift apart.
    """
    target = project_dir(name, root)
    if target is None or not target.is_dir():
        return _error(
            "project_not_found",
            f"No project {name!r} to export.",
            available=list_projects(root),
        )

    # Refresh the project manifest from what is actually on disk, so the archive
    # describes the files it contains rather than what was planned at creation.
    stations: list[str] = []
    dates: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    for station_dir in sorted(p for p in target.iterdir() if p.is_dir()):
        stations.append(station_dir.name)
        for day_dir in sorted(p for p in station_dir.iterdir() if p.is_dir()):
            dates.add(day_dir.name)
            for item in sorted(day_dir.iterdir()):
                if item.is_file():
                    artifacts.append(
                        {
                            "station": station_dir.name,
                            "date": day_dir.name,
                            "file": item.name,
                            "bytes": item.stat().st_size,
                        }
                    )
    write_manifest(
        target / _MANIFEST,
        project_manifest(
            target.name, stations, sorted(dates), artifacts, root=root,
            extra={"exported_at": utc_now()},
        ),
    )

    out_dir = Path(exports) if exports is not None else EXPORTS_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{target.name}.zip"
    # Replace rather than merge: a stale member from a previous export would
    # otherwise survive in the archive and describe a file that no longer exists.
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(target).as_posix())

    return {
        "ok": True,
        "project": target.name,
        "path": str(zip_path),
        "bytes": zip_path.stat().st_size,
        "size": _human(zip_path.stat().st_size),
        "files": len(artifacts) + 1,
        "stations": sorted(stations),
        "dates": sorted(dates),
        "manifest": str(target / _MANIFEST),
    }


def parse_slot(handle: Any) -> tuple[str, str] | None:
    """``"raw:irt:2024-09-10"`` -> ``("irt", "2024-09-10")``, else ``None``.

    This is how a frame handle becomes an artifact location. Keying the project
    tree off the handle the tools already advertise means the folder is named by
    the same string the model was told to copy -- there is no second naming scheme
    to keep in step, and no way for a chart and its data to land in different
    places because they were labelled differently.
    """
    if not isinstance(handle, str):
        return None
    parts = [p.strip().lower() for p in handle.strip().split(":")]
    if len(parts) != 3 or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


# --------------------------------------------------------------------------- #
# the workspace a run carries
# --------------------------------------------------------------------------- #
class Workspace:
    """The project the current agent run is filling in, if any.

    Held on the :class:`~agent_core.FrameStore` so the three project tools can
    reach it through the same ``(args, store)`` handler signature as every other
    tool, and so a run never leaks its project into the next one.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else PROJECTS_ROOT
        self.project: str | None = None

    # -- which project ----------------------------------------------------- #
    def resolve(self, name: Any = None) -> str | None:
        """The project to act on: the explicit ``name``, else the current one."""
        if name is not None and str(name).strip():
            target = project_dir(name, self.root)
            return target.name if target is not None else None
        return self.project

    def create(self, name: Any, **kwargs: Any) -> dict[str, Any]:
        result = create_project(name, root=self.root, **kwargs)
        if result.get("ok"):
            self.project = result["project"]
        return result

    # -- writing ----------------------------------------------------------- #
    def save(
        self,
        station: Any,
        day: Any,
        frame: Any,
        kind: str = "raw",
        project: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Write an artifact into the current project.

        ``project`` overrides the open one, which is what a tool taking an explicit
        name needs: without it ``fetch_many(project="Elsewhere")`` would *report*
        Elsewhere and quietly file the data nowhere.

        A no-op success when no project is open: plotting must keep working
        exactly as before for every flow that never mentions projects, and the
        plotter's own return value is what those flows report.
        """
        target = project or self.project
        if not target:
            return {"ok": True, "skipped": "no project is open"}
        return save_artifact(target, station, day, frame, kind=kind,
                             root=self.root, **kwargs)

    def save_chart(
        self,
        source: str | Path,
        handles: Sequence[Any],
        kind: str = "chart",
    ) -> list[str]:
        """Copy a chart into every project day its handles point at.

        Returns the paths written. A chart built from two days is deliberately
        written into both, because after export the two days' folders are separate
        and a reader looking at one of them should find the comparison in it.
        """
        if not self.project:
            return []
        written: list[str] = []
        seen: set[tuple[str, str]] = set()
        for handle in handles:
            parsed = parse_slot(handle)
            if parsed is None or parsed in seen:
                continue
            seen.add(parsed)
            station, day = parsed
            target = artifact_dir(self.project, station, day, self.root)
            if target is None or not target.is_dir():
                # Only write into a day that actually exists in the project.
                # Creating it here would invent a folder that no fetch produced.
                continue
            destination = target / f"{kind}.html"
            try:
                target.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(Path(source).read_bytes())
            except OSError:
                continue
            written.append(str(destination))
        return written

    # -- reading ----------------------------------------------------------- #
    def tree_markdown(self, name: Any = None) -> str:
        target = self.resolve(name)
        return tree_markdown(target, self.root) if target else ""

    def known(self) -> list[str]:
        return list_projects(self.root)


def iter_manifests(name: str, root: str | Path | None = None) -> Iterable[dict[str, Any]]:
    """Every leaf manifest in a project, skipping unreadable ones."""
    target = project_dir(name, root)
    if target is None or not target.is_dir():
        return []
    out = []
    for manifest in sorted(target.rglob(_MANIFEST)):
        try:
            out.append(json.loads(manifest.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out