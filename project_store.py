"""Projects: a real folder tree the agent's output lives in.

Stage 3 answered a question and dropped a chart in ``plots/``. That is enough to
answer one question and not enough to hand anything to anybody else: the files are
unrelated to each other, nothing records which download produced them, and there
is no way to send a colleague "the three days you asked about" as one artefact.

A project is a directory laid out by what the data *is*::

    /projects/<name>/
        manifest.json
        charts.json
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

import csv
import hashlib
import html
import io
import json
import re
import geomag_coords as gcoords
import intermagnet_loader as iloader
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote

import chart_naming

__all__ = [
    "PROJECTS_ROOT",
    "EXPORTS_ROOT",
    "CHARTS_INDEX",
    "Workspace",
    "agent_commit",
    "utc_now",
    "create_project",
    "list_projects",
    "parse_slot",
    "project_dir",
    "project_manifest",
    "read_project_manifest",
    "read_chart_index",
    "sort_charts",
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

#: The machine-readable chart index. Kept beside the project manifest rather
#: than inside it because ``export_project`` rewrites the project manifest from
#: the files on disk; a chart index must survive that refresh.
CHARTS_INDEX = "charts.json"

_EXPORT_INDEX_CSV = "index.csv"
_EXPORT_INDEX_HTML = "index.html"
_EXPORT_MANIFEST = "export_manifest.json"
_ROOT_METADATA_NAMES = {
    _MANIFEST,
    CHARTS_INDEX,
    _EXPORT_INDEX_CSV,
    _EXPORT_INDEX_HTML,
    _EXPORT_MANIFEST,
}
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

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
MAX_SLUG_LENGTH = 80


def _slug(name: Any) -> str:
    """A folder name that cannot escape the projects root.

    The name reaches here from a language model, so it is treated as untrusted
    input rather than a label: ``../..`` and absolute paths are rejected outright
    instead of being sanitised into something that looks safe. Restricting the
    alphabet also keeps the tree readable and the ZIP safe to extract on Windows.

    Windows device names (``CON``, ``NUL``, ``COM1`` ...) and overlong names are
    rejected as well.  Creating a directory called ``NUL`` can silently write to
    a device or raise halfway through, so treating it as an ordinary slug is not
    safe even though it looks harmless.
    """
    text = str(name or "").strip().lower()
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in text)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    cleaned = cleaned.strip("-")
    if not cleaned or len(cleaned) > MAX_SLUG_LENGTH:
        return ""
    if cleaned.casefold() in _WINDOWS_RESERVED:
        return ""
    return cleaned


def _as_bool(value: Any, default: bool = False) -> bool:
    """Accept the boolean spellings a JSON/LLM caller actually sends.

    ``"false"`` is a truthy Python string.  Tool arguments arrive as JSON, but a
    model that serialises them itself may send ``"false"`` or ``0``; treating
    those as true silently produces an index the user asked not to have.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def _is_link_like(path: Path) -> bool:
    """True for a symlink or Windows junction (a reparse-point directory).

    Project trees are written by us, but a project can be copied from somewhere
    else or edited by hand.  Following a link while listing or zipping would
    turn ``export_project`` into a way to read files outside the project.
    """
    try:
        if path.is_symlink():
            return True
    except OSError:
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction):
        try:
            if is_junction():
                return True
        except OSError:
            return True
    return False


def _is_within(target: Path, path: Path) -> bool:
    """True when *path* resolves inside *target* (links included)."""
    try:
        path.resolve().relative_to(target.resolve())
        return True
    except (OSError, ValueError):
        return False


def _iter_safe_children(target: Path, directory: Path) -> Iterable[Path]:
    """Yield immediate children of *directory* that stay inside *target*."""
    try:
        entries = sorted(directory.iterdir(), key=lambda item: item.name.casefold())
    except OSError:
        return
    for path in entries:
        if _is_link_like(path) or not _is_within(target, path):
            continue
        yield path


def _walk_project(target: Path, directory: Path | None = None) -> Iterable[Path]:
    """Walk a project without following symlinks or junctions.

    ``Path.rglob`` follows directory links on some platforms.  A link to a
    directory outside the project would then make the tree listing and the
    default ZIP include files the project does not own.  This walker skips the
    link itself *and* never descends through it.
    """
    current = directory if directory is not None else target
    for path in _iter_safe_children(target, current):
        yield path
        if path.is_dir():
            yield from _walk_project(target, path)


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
    try:
        target = (base / slug).resolve()
        # Belt and braces: even after the alphabet filter, confirm the result is
        # a direct child of the root. Cheap, and it makes the invariant local.
        if target.parent != base.resolve():
            return None
    except OSError:
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
    if _is_link_like(target):
        raise OSError(f"refusing to write through a link: {target}")
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
    if (
        not manifest.is_file()
        or _is_link_like(manifest)
        or not _is_within(target, manifest)
    ):
        return _error(
            "project_not_found",
            f"No project {target.name!r}. Known projects: {', '.join(list_projects(root)) or '(none)'}",
            available=list_projects(root),
        )
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _error("unreadable_manifest", f"{type(exc).__name__}: {exc}", path=str(manifest))


def _safe_export_component(value: Any) -> str | None:
    """A user-facing folder/archive component that cannot escape the archive.

    Custom folder names come from a prompt, so they are untrusted input.  The
    function keeps Unicode letters (a user may legitimately ask for ``День 1``)
    but rejects separators, drive letters, ``..`` and Windows device names.
    """
    text = str(value or "").strip()
    if not text or text in (".", ".."):
        return None
    if any(ch in text for ch in ("/", "\\", "\x00")):
        return None
    if re.match(r"^[A-Za-z]:", text):
        return None
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in text)
    cleaned = re.sub(r"_+", "_", cleaned).strip(" ._-")
    if not cleaned or cleaned.endswith(".") or cleaned.endswith(" "):
        return None
    if cleaned.split(".", 1)[0].casefold() in _WINDOWS_RESERVED:
        return None
    if len(cleaned) > 80:
        return None
    return cleaned


def _archive_filename(value: Any, default: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return default
    if text.lower().endswith(".zip"):
        text = text[:-4]
    safe = _safe_export_component(text)
    return f"{safe}.zip" if safe else None


def _load_chart_index(target: Path) -> list[dict[str, Any]]:
    path = target / CHARTS_INDEX
    if not path.is_file() or _is_link_like(path) or not _is_within(target, path):
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    charts = payload.get("charts") if isinstance(payload, dict) else payload
    return [row for row in (charts or []) if isinstance(row, dict)]


def _write_chart_index(target: Path, charts: Sequence[dict[str, Any]]) -> Path:
    path = target / CHARTS_INDEX
    if _is_link_like(path):
        raise OSError(f"refusing to write through a link: {path}")
    payload = {
        "project": target.name,
        "updated_at": utc_now(),
        "charts": list(charts),
    }
    return write_manifest(path, payload)


def _upsert_chart_index(target: Path, entry: dict[str, Any]) -> None:
    """Add/replace one chart entry, keyed by the archive-relative path."""
    charts = _load_chart_index(target)
    key = (entry.get("relative_path"), entry.get("file"))
    out: list[dict[str, Any]] = []
    replaced = False
    for row in charts:
        if (row.get("relative_path"), row.get("file")) == key:
            out.append(entry)
            replaced = True
        else:
            out.append(row)
    if not replaced:
        out.append(entry)
    _write_chart_index(target, out)


def _source_from_relative_path(relative_path: str, filename: str) -> str:
    source = chart_naming.source_from_filename(filename)
    if source != "UNKNOWN":
        return source
    first = str(relative_path or "").split("/", 1)[0].lower()
    return {
        "omni": "OMNI",
        "kyoto": "KYOTO",
        "supermag": "SUPERMAG",
        "noaa": "NOAA",
        "iswa": "ISWA",
    }.get(first, "UNKNOWN")


def _fallback_chart_row(target: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(target).as_posix()
    parts = relative.split("/")
    station = parts[0] if len(parts) >= 3 else ""
    day = parts[1] if len(parts) >= 3 else ""
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return {
        "file": path.name,
        "relative_path": relative,
        "source": _source_from_relative_path(relative, path.name),
        "sources": [_source_from_relative_path(relative, path.name)],
        "station": station,
        "stations": [station] if station else [],
        "date": day,
        "dates": [day] if day else [],
        "components": [],
        "kind": "chart",
        "product": None,
        "handles": [],
        "sha256": None,
        "bytes": size,
    }


def _chart_rows(target: Path) -> list[dict[str, Any]]:
    """The chart index plus any HTML it does not know about.

    Projects created before ``charts.json`` existed have no index; a project
    that gained old and new charts must not silently drop the old ones from the
    archive index.
    """
    rows = _load_chart_index(target)
    known = {
        (str(row.get("relative_path") or ""), str(row.get("file") or ""))
        for row in rows
    }
    for path in sorted(
        (p for p in _walk_project(target) if p.suffix.lower() == ".html"),
        key=lambda item: item.relative_to(target).as_posix(),
    ):
        if not path.is_file():
            continue
        relative = path.relative_to(target).as_posix()
        if "/" not in relative and path.name in _ROOT_METADATA_NAMES:
            continue
        if (relative, path.name) not in known:
            rows.append(_fallback_chart_row(target, path))
    return rows


def read_chart_index(name: str, root: str | Path | None = None) -> list[dict[str, Any]]:
    target = project_dir(name, root)
    if target is None or not target.is_dir():
        return []
    return _chart_rows(target)


_SORT_ALIASES = {
    "date": "start_date",
    "start": "start_date",
    "end": "end_date",
    "component": "components",
    "components": "components",
    "path": "relative_path",
    "file_name": "file",
    "source_name": "source",
}
_SORT_KEYS = {
    "source", "sources", "station", "stations", "start_date", "end_date",
    "date", "dates", "components", "kind", "product", "file",
    "relative_path", "bytes",
}


def _normalise_sort_by(sort_by: Any) -> list[str]:
    default = ["station", "start_date", "components", "source", "file"]
    if sort_by is None or sort_by == "" or sort_by == []:
        return default
    raw = [sort_by] if isinstance(sort_by, str) else list(sort_by or [])
    out: list[str] = []
    for item in raw:
        for part in re.split(r"[,;\s]+", str(item or "").strip()):
            key = _SORT_ALIASES.get(part.lower(), part.lower())
            if key in _SORT_KEYS and key not in out:
                out.append(key)
    return out or default


def _row_value(row: dict[str, Any], key: str) -> Any:
    if key in ("start_date", "date"):
        value = row.get("start_date") or row.get("date")
    elif key in ("end_date", "dates"):
        value = row.get("end_date") or row.get("dates")
    elif key == "stations":
        value = row.get("stations") or row.get("station")
    elif key == "sources":
        value = row.get("sources") or row.get("source")
    else:
        value = row.get(key)
    if isinstance(value, list):
        return "-".join(str(v) for v in value)
    return value


def sort_charts(
    charts: Sequence[dict[str, Any]],
    sort_by: Any = None,
    *,
    reverse: bool = False,
    source_order: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Sort chart metadata by semantic fields, never by filename alone.

    ``sort_by`` accepts a string (``"source,station,date"``) or a list.  The
    final ``file``/``relative_path`` tie-break keeps the result deterministic.
    """
    keys = _normalise_sort_by(sort_by)
    if isinstance(source_order, str):
        source_order = [source_order]
    order = {
        str(source).strip().upper(): index
        for index, source in enumerate(source_order or [])
    }

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        values: list[Any] = []
        for field in keys:
            if field == "source" and order:
                source = str(_row_value(row, "source") or "").upper()
                rank = order.get(source)
                if rank is None:
                    rank = min(
                        (order.get(part, 999) for part in source.split("-")),
                        default=999,
                    )
                values.append((0, rank))
            else:
                value = _row_value(row, field)
                values.append((1, str(value or "").casefold()))
        values.append(
            (1, str(row.get("relative_path") or row.get("file") or "").casefold())
        )
        return tuple(values)

    return sorted(charts, key=key, reverse=reverse)


def _index_row(row: dict[str, Any], archive_path: str, order: int) -> dict[str, Any]:
    components = row.get("components") or []
    if isinstance(components, str):
        components = [components]
    return {
        "order": order,
        "archive_path": archive_path,
        "relative_path": archive_path,
        "file": row.get("file") or Path(archive_path).name,
        "source": row.get("source") or "UNKNOWN",
        "station": row.get("station") or "",
        "date": row.get("date") or row.get("start_date") or "",
        "components": "-".join(str(c) for c in components),
        "kind": row.get("kind") or "chart",
        "product": row.get("product") or "",
        "bytes": int(row.get("bytes") or 0),
        "sha256": row.get("sha256") or "",
    }


def _index_csv(rows: Sequence[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    fields = [
        "order", "archive_path", "file", "source", "station", "date",
        "components", "kind", "product", "bytes", "sha256",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _index_html(rows: Sequence[dict[str, Any]], title: str) -> str:
    body = []
    for row in rows:
        href = quote(str(row.get("archive_path") or ""), safe="/._-")
        body.append(
            "<tr>"
            f"<td>{row['order']}</td>"
            f"<td><a href=\"{html.escape(href)}\">{html.escape(str(row['file']))}</a></td>"
            f"<td>{html.escape(str(row['source']))}</td>"
            f"<td>{html.escape(str(row['station']))}</td>"
            f"<td>{html.escape(str(row['date']))}</td>"
            f"<td>{html.escape(str(row['components']))}</td>"
            f"<td>{html.escape(str(row['kind']))}</td>"
            f"<td>{html.escape(str(row['product']))}</td>"
            "</tr>"
        )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.3rem .6rem}"
        "th{background:#f3f3f3;text-align:left}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        "<table><thead><tr><th>#</th><th>File</th><th>Source</th>"
        "<th>Station</th><th>Date</th><th>Components</th><th>Kind</th>"
        "<th>Product</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></body></html>"
    )


def _resolve_case_insensitive(target: Path, text: str) -> Path:
    """Resolve a project-relative path, matching directory case when needed.

    Canonical folders are lowercase (``irt/2024-05-14``), while a model may
    write ``IRT/2024-05-14``.  Exact matches win; otherwise each component is
    matched case-insensitively, and ambiguity falls back to the original path.
    """
    current = target
    for part in Path(text).parts:
        exact = current / part
        if exact.exists():
            current = exact
            continue
        try:
            matches = [
                candidate
                for candidate in current.iterdir()
                if candidate.name.casefold() == part.casefold()
            ]
        except OSError:
            return target / text
        if len(matches) != 1:
            return target / text
        current = matches[0]
    return current


def _glob_case_insensitive(target: Path, text: str) -> list[Path]:
    """Glob below a case-insensitively resolved literal prefix."""
    parts = Path(text).parts
    prefix: list[str] = []
    for part in parts:
        if any(ch in part for ch in "*?["):
            break
        prefix.append(part)
    base = target
    if prefix:
        base = _resolve_case_insensitive(target, "/".join(prefix))
    remainder = "/".join(parts[len(prefix):]) or "*"
    try:
        return list(base.glob(remainder))
    except (OSError, ValueError):
        return []


def _resolve_export_folders(
    target: Path,
    folders: Any,
) -> tuple[list[tuple[str, str, Path]] | None, dict[str, Any] | None]:
    """Resolve ``{folder: [path/glob, ...]}`` into concrete archive members."""
    if not isinstance(folders, dict) or not folders:
        return None, _error(
            "invalid_folders",
            "folders must be a non-empty object: {folder_name: [paths or globs]}.",
        )
    mapping: list[tuple[str, str, Path]] = []
    seen_folders: set[str] = set()
    seen_members: set[str] = set()
    for raw_name, raw_patterns in folders.items():
        folder = _safe_export_component(raw_name)
        if folder is None:
            return None, _error(
                "invalid_folder_name",
                f"Folder name {raw_name!r} is empty, unsafe or a reserved name.",
            )
        if folder.casefold() in seen_folders:
            return None, _error(
                "duplicate_folder_name",
                f"Folder name {folder!r} is used more than once (case-insensitive).",
            )
        if folder.casefold() in {name.casefold() for name in _ROOT_METADATA_NAMES}:
            return None, _error(
                "reserved_folder_name",
                f"Folder name {folder!r} is reserved for archive metadata.",
            )
        seen_folders.add(folder.casefold())
        patterns = (
            list(raw_patterns)
            if isinstance(raw_patterns, (list, tuple, set))
            else [raw_patterns]
        )
        if not patterns:
            return None, _error(
                "empty_export_folder",
                f"Folder {folder!r} has no paths or globs.",
            )
        matches = 0
        for pattern in patterns:
            text = str(pattern or "").strip().replace("\\", "/")
            try:
                path_parts = Path(text).parts
            except (OSError, ValueError):
                path_parts = ("..",)
            if (
                not text
                or "\x00" in text
                or text.startswith("/")
                or re.match(r"^[A-Za-z]:", text)
                or ".." in path_parts
            ):
                return None, _error(
                    "invalid_folder_pattern",
                    f"Path/glob {pattern!r} in folder {folder!r} escapes the project.",
                )
            candidates: list[Path] = []
            if any(ch in text for ch in "*?["):
                try:
                    candidates = [
                        p
                        for p in _glob_case_insensitive(target, text)
                        if p.is_file()
                        and not _is_link_like(p)
                        and _is_within(target, p)
                    ]
                except (OSError, ValueError) as exc:
                    return None, _error(
                        "invalid_folder_pattern",
                        f"Glob {pattern!r} is invalid: {exc}",
                    )
            else:
                candidate = target / text
                if not candidate.exists():
                    candidate = _resolve_case_insensitive(target, text)
                if (
                    candidate.is_dir()
                    and not _is_link_like(candidate)
                    and _is_within(target, candidate)
                ):
                    candidates = [
                        p for p in _walk_project(target, candidate) if p.is_file()
                    ]
                elif (
                    candidate.is_file()
                    and not _is_link_like(candidate)
                    and _is_within(target, candidate)
                ):
                    candidates = [candidate]
            for path in candidates:
                try:
                    relative = path.resolve().relative_to(target.resolve()).as_posix()
                except (OSError, ValueError):
                    continue
                member = f"{folder}/{relative}"
                if member in seen_members:
                    continue
                seen_members.add(member)
                mapping.append((folder, relative, path))
                matches += 1
        if not matches:
            return None, _error(
                "empty_export_folder",
                f"Folder {folder!r} matched no files.",
                patterns=patterns,
            )
    return mapping, None


def project_manifest(
    name: str,
    stations: Sequence[str],
    dates: Sequence[str],
    artifacts: Sequence[dict[str, Any]] | None = None,
    root: str | Path | None = None,
    extra: dict[str, Any] | None = None,
    created_at: str | None = None,
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
        "created_at": created_at or utc_now(),
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
            "A project name must be non-empty, at most 80 characters, use only "
            "letters, digits, '-' or '_', and not be a Windows device name.",
            requested=name,
        )
    manifest_path = target / _MANIFEST
    if _is_link_like(manifest_path):
        return _error(
            "unsafe_project_path",
            "The project manifest is a symlink or junction; refusing to use it.",
            path=str(manifest_path),
        )
    existed = manifest_path.is_file()
    try:
        target.mkdir(parents=True, exist_ok=True)
        if not existed:
            write_manifest(
                manifest_path,
                project_manifest(target.name, stations or [], dates or [], root=root),
            )
    except OSError as exc:
        return _error(
            "project_create_failed",
            f"{type(exc).__name__}: {exc}",
            requested=name,
            path=str(target),
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


def _geomag_for_station(station_code: str) -> dict | None:
    try:
        code = str(station_code or '').strip().upper()
        regs = iloader.get_available_stations()
        rec = regs.get(code)
        if not rec: return None
        if isinstance(rec, tuple) and len(rec) >= 3:
            _name, lat, lon = rec[0], rec[1], rec[2]
        else: return None
        res = gcoords.geographic_to_geomagnetic(float(lat), float(lon))
        if isinstance(res, dict) and res.get('ok'):
            return {'geo_lat': res.get('geomag_lat'), 'geo_lon': res.get('geomag_lon'), 'model': res.get('model', 'tilted_dipole')}
    except Exception:
        return None
    return None

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
    project_path = project_dir(project, root)
    target = artifact_dir(project, station, day, root)
    if (
        target is None
        or project_path is None
        or _is_link_like(target)
        or not _is_within(project_path, target)
    ):
        return _error(
            "invalid_artifact_path",
            "A project, station and date are all required to save an artifact.",
            project=project, station=station, date=day,
        )
    target.mkdir(parents=True, exist_ok=True)
    filename = f"{_slug(kind) or 'data'}.csv"
    csv_path = target / filename
    if _is_link_like(csv_path) or _is_link_like(target / _MANIFEST):
        return _error(
            "unsafe_artifact_path",
            "Refusing to write an artifact through a symlink or junction.",
            path=str(target),
        )
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
        "geomagnetic_coords": _geomag_for_station(station),
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
    for path in sorted(
        _walk_project(target), key=lambda item: item.relative_to(target).as_posix()
    ):
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
    archive_name: Any = None,
    folders: Any = None,
    sort_by: Any = None,
    source_order: Sequence[str] | None = None,
    include_index: bool = True,
) -> dict[str, Any]:
    """Zip a project and return where the archive is.

    By default the archive mirrors the canonical ``<station>/<date>/`` tree.
    ``folders`` is an explicit presentation view requested by the user, for
    example ``{"1": ["irt/2024-05-14"], "2": ["api/2024-05-14"]}``.  It copies
    the selected files under the requested folder names; it never renames the
    canonical files in the project.  ``archive_name`` changes only the ZIP name.

    ``sort_by`` and ``source_order`` affect ``index.csv``/``index.html``, not the
    filenames.  The index is generated from ``charts.json`` metadata, so a
    source-first or date-first view never depends on lexical filename order.
    """
    target = project_dir(name, root)
    if target is None or not target.is_dir():
        return _error(
            "project_not_found",
            f"No project {name!r} to export.",
            available=list_projects(root),
        )
    include_index = _as_bool(include_index, True)

    manifest_path = target / _MANIFEST
    if _is_link_like(manifest_path):
        return _error(
            "unsafe_project_path",
            "The project manifest is a symlink or junction; refusing to export it.",
            path=str(manifest_path),
        )
    created_at: str | None = None
    if manifest_path.is_file() and _is_within(target, manifest_path):
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(previous, dict):
                created_at = previous.get("created_at")
        except (OSError, ValueError):
            created_at = None

    # Refresh the project manifest from what is actually on disk, so the archive
    # describes the files it contains rather than what was planned at creation.
    stations: list[str] = []
    dates: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    for station_dir in (
        p for p in _iter_safe_children(target, target) if p.is_dir()
    ):
        stations.append(station_dir.name)
        for day_dir in (
            p for p in _iter_safe_children(target, station_dir) if p.is_dir()
        ):
            dates.add(day_dir.name)
            for item in _iter_safe_children(target, day_dir):
                if item.is_file():
                    artifacts.append(
                        {
                            "station": station_dir.name,
                            "date": day_dir.name,
                            "file": item.name,
                            "bytes": item.stat().st_size,
                        }
                    )

    zip_name = _archive_filename(archive_name, f"{target.name}.zip")
    if zip_name is None:
        return _error(
            "invalid_archive_name",
            "Archive name is empty, unsafe or a reserved name.",
            requested=archive_name,
        )
    out_dir = Path(exports) if exports is not None else EXPORTS_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / zip_name

    mapping: list[tuple[str, str, Path]] | None = None
    folder_counts: dict[str, int] = {}
    if folders is not None:
        mapping, error = _resolve_export_folders(target, folders)
        if error is not None:
            return error
        assert mapping is not None
        for folder, _relative, _path in mapping:
            folder_counts[folder] = folder_counts.get(folder, 0) + 1

    # Only now that the archive name and every folder pattern have validated do
    # we touch the project manifest.  A rejected export must leave the project
    # exactly as it was.
    write_manifest(
        manifest_path,
        project_manifest(
            target.name,
            stations,
            sorted(dates),
            artifacts,
            root=root,
            extra={"exported_at": utc_now()},
            created_at=created_at,
        ),
    )

    charts = _chart_rows(target)
    sorted_charts = sort_charts(charts, sort_by, source_order=source_order)
    chart_by_relative = {
        str(row.get("relative_path") or ""): row for row in sorted_charts
    }

    index_rows: list[dict[str, Any]] = []
    if mapping is None:
        for chart in sorted_charts:
            relative = str(chart.get("relative_path") or "")
            if relative and (target / relative).is_file():
                index_rows.append(_index_row(chart, relative, 0))
    else:
        for folder, relative, path in mapping:
            if path.suffix.lower() != ".html" or relative in _ROOT_METADATA_NAMES:
                continue
            chart = chart_by_relative.get(relative) or _fallback_chart_row(
                target, path
            )
            index_rows.append(_index_row(chart, f"{folder}/{relative}", 0))
    index_rows = sort_charts(index_rows, sort_by, source_order=source_order)
    for order, row in enumerate(index_rows, 1):
        row["order"] = order

    if mapping is None:
        planned = [
            path.relative_to(target).as_posix()
            for path in sorted(
                _walk_project(target),
                key=lambda item: item.relative_to(target).as_posix(),
            )
            if path.is_file()
            and path.relative_to(target).as_posix()
            not in {_EXPORT_INDEX_CSV, _EXPORT_INDEX_HTML, _EXPORT_MANIFEST}
        ]
    else:
        planned = [f"{folder}/{relative}" for folder, relative, _path in mapping]
        for root_name in (_MANIFEST, CHARTS_INDEX):
            candidate = target / root_name
            if candidate.is_file():
                planned.append(root_name)
    if include_index:
        planned.extend([_EXPORT_INDEX_CSV, _EXPORT_INDEX_HTML])
    planned.append(_EXPORT_MANIFEST)
    # Preserve order but never write the same member twice.
    planned = list(dict.fromkeys(planned))

    source_order_list = (
        [source_order] if isinstance(source_order, str) else list(source_order or [])
    )
    export_manifest = {
        "project": target.name,
        "archive": zip_name,
        "created_at": utc_now(),
        "agent_commit": agent_commit(),
        "sort_by": _normalise_sort_by(sort_by),
        "source_order": [str(s).strip().upper() for s in source_order_list],
        "folders": (
            {folder: count for folder, count in folder_counts.items()}
            if mapping is not None
            else {"layout": "canonical"}
        ),
        "files": planned,
        "charts": index_rows,
        "counts": {"files": len(planned), "charts": len(index_rows)},
        "index": (
            {"csv": _EXPORT_INDEX_CSV, "html": _EXPORT_INDEX_HTML}
            if include_index
            else None
        ),
    }

    written: set[str] = set()

    def write_member(archive: zipfile.ZipFile, path: Path, arcname: str) -> None:
        if arcname in written:
            return
        archive.write(path, arcname)
        written.add(arcname)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        if mapping is None:
            for path in sorted(
                _walk_project(target),
                key=lambda item: item.relative_to(target).as_posix(),
            ):
                if not path.is_file():
                    continue
                arcname = path.relative_to(target).as_posix()
                if arcname in {
                    _EXPORT_INDEX_CSV,
                    _EXPORT_INDEX_HTML,
                    _EXPORT_MANIFEST,
                }:
                    continue
                write_member(archive, path, arcname)
        else:
            for folder, relative, path in mapping:
                write_member(archive, path, f"{folder}/{relative}")
            for root_name in (_MANIFEST, CHARTS_INDEX):
                candidate = target / root_name
                if candidate.is_file():
                    write_member(archive, candidate, root_name)
        if include_index:
            if _EXPORT_INDEX_CSV not in written:
                archive.writestr(_EXPORT_INDEX_CSV, _index_csv(index_rows))
                written.add(_EXPORT_INDEX_CSV)
            if _EXPORT_INDEX_HTML not in written:
                archive.writestr(
                    _EXPORT_INDEX_HTML,
                    _index_html(index_rows, f"{target.name} chart index"),
                )
                written.add(_EXPORT_INDEX_HTML)
        if _EXPORT_MANIFEST not in written:
            archive.writestr(
                _EXPORT_MANIFEST,
                json.dumps(export_manifest, ensure_ascii=False, indent=2) + "\n",
            )
            written.add(_EXPORT_MANIFEST)

    size = zip_path.stat().st_size
    return {
        "ok": True,
        "project": target.name,
        "path": str(zip_path),
        "filename": zip_name,
        "archive": zip_name,
        "bytes": size,
        "size": _human(size),
        "files": len(written),
        "charts": len(index_rows),
        "stations": sorted(stations),
        "dates": sorted(dates),
        "manifest": str(target / _MANIFEST),
        "folders": (
            {folder: count for folder, count in folder_counts.items()}
            if mapping is not None
            else {"layout": "canonical"}
        ),
        "index": (
            {"csv": _EXPORT_INDEX_CSV, "html": _EXPORT_INDEX_HTML}
            if include_index
            else None
        ),
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
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Copy a chart into every project day its handles point at.

        The chart keeps the canonical name it was drawn under.  A chart built
        from two days is deliberately written into both, because after export the
        two days' folders are separate and a reader looking at one of them should
        find the comparison in it.  Each copy is also recorded in ``charts.json``
        so sorting/grouping can use metadata instead of parsing filenames.
        """
        if not self.project:
            return []
        if isinstance(handles, str):
            handles = [handles]
        chart_path = Path(source)
        try:
            payload = chart_path.read_bytes()
        except OSError:
            return []
        all_sources = chart_naming.sources_for_handles(handles)
        filename = chart_naming.safe_filename(chart_path.name, all_sources)
        digest = hashlib.sha256(payload).hexdigest()
        project_target = project_dir(self.project, self.root)
        if project_target is None:
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
            if (
                target is None
                or not target.is_dir()
                or _is_link_like(target)
                or not _is_within(project_target, target)
            ):
                # Only write into a day that actually exists in the project.
                # Creating it here would invent a folder that no fetch produced.
                continue
            destination = target / filename
            if _is_link_like(destination):
                continue
            try:
                target.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
            except OSError:
                continue
            relative = destination.relative_to(project_target).as_posix()
            entry = chart_naming.chart_metadata(
                filename,
                [handle],
                kind,
                components=(metadata or {}).get("components"),
                product=(metadata or {}).get("product"),
                relative_path=relative,
                sha256=digest,
                size=len(payload),
            )
            entry["handles"] = list(
                (metadata or {}).get("handles") or handles or []
            )
            # The chart's source suffix describes every frame it used, not just
            # the day this copy happens to be filed under.
            entry["sources"] = all_sources
            entry["source"] = "-".join(all_sources)
            entry["filed_at"] = utc_now()
            try:
                _upsert_chart_index(project_target, entry)
            except OSError:
                # A chart is the answer; a missing index entry is bookkeeping.
                pass
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
    for manifest in sorted(
        (p for p in _walk_project(target) if p.name == _MANIFEST),
        key=lambda item: item.relative_to(target).as_posix(),
    ):
        try:
            out.append(json.loads(manifest.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out