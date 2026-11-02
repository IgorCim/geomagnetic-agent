"""Package reviewed sources and selected evidence; never include models or user projects."""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261102", choices=("20261101", "20261102"))
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "exports" / f"geomagnetic_portfolio_source_{options.date}.zip"
    if output.exists():
        raise FileExistsError(f"Preserving existing package: {output}")
    candidates = sorted((root / "verification").glob("*/run.json"))
    latest = next((p for p in reversed(candidates) if json.loads(p.read_text(encoding="utf-8")).get("exit_code") == 0), None)
    if latest is None:
        raise RuntimeError("No successful offline verification report")
    report = json.loads(latest.read_text(encoding="utf-8"))
    for name, digest in report["source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source differs from latest successful offline run: {name}")
    files = {p for p in root.iterdir() if p.is_file() and
             (p.suffix in {".py", ".md", ".txt", ".ini", ".ipynb"} or p.name == ".gitignore")}
    for folder in ("tests", "tools", "docs", ".github"):
        files.update(p for p in (root / folder).rglob("*") if p.is_file() and p.suffix in {".py", ".md", ".yml", ".yaml", ".json"})
    historical = root / "docs/evidence"
    if (historical / "stress_kaggle_summary.json").is_file():
        summary = json.loads((historical / "stress_kaggle_summary.json").read_text(encoding="utf-8"))
        for name, expected in summary["extracted_files"].items():
            path = (historical / name).resolve()
            assert path.is_relative_to(historical.resolve()) and path.name.endswith("_redacted.log")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"], name
            files.add(path)
    evidence = [latest.parent, root / "verification/20261101T130849Z_gradio", root / "verification/20261101T124759Z_sources"]
    for folder in evidence:
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        files.update(p for p in folder.iterdir() if p.is_file())
    if options.date == "20261102":
        for folder in sorted((root / "verification").glob("20261102*")):
            if folder.is_dir():
                files.update(p for p in folder.iterdir() if p.is_file() and p.suffix == ".json" and folder.name != "20261102T062939Z_external_recheck")
    payload = {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(files)}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout.strip())
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "package_date": options.date, "git_revision": revision,
                "dirty_tree": dirty, "offline_evidence": latest.parent.relative_to(root).as_posix(),
                "scope": "Sources, documentation, notebooks and selected evidence including redacted historical notebook output. No models, credentials, user projects or caches.",
                "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, data in payload.items():
            bundle.writestr(name, data)
        bundle.writestr("SOURCE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(output) as bundle:
        assert bundle.testzip() is None
        for name, digest in manifest["sha256"].items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == digest, name
    receipt = {"path": str(output), "bytes": output.stat().st_size, "files": len(payload),
               "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "offline_evidence": manifest["offline_evidence"]}
    (root / f"verification/PORTFOLIO_PACKAGE_{options.date}.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
