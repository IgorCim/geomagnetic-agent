"""Run repository tests in a disposable checkout, preserving demo artifacts.

Usage: python -B tools/verify_offline.py
This does not load a model or run tests marked ``network``.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "verification" / stamp
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(root.glob("*.py")) + sorted((root / "tests").glob("*.py"))
    files += [root / "pytest.ini"]
    # The replay-runner tests read the same checked-in prompt documents as Kaggle.
    files += [root / name for name in [
        'tools/run_chat_acceptance.py', 'tools/kaggle_chat_acceptance.py', 'docs/LONG_CHAT.md',
        'tools/run_full_evidence.py',
        'tools/run_prompt_diagnostics.py', 'tools/kaggle_prompt_diagnostics.py',
        'tools/kaggle_prompt_diagnostics_remaining.py',
        'tools/run_focused_review.py', 'tools/kaggle_focused_review.py',
        'docs/qa/ready_tool_cases.json', 'docs/qa/TOOL_RUNBOOK.md',
        'docs/qa/FORMULA_RUNBOOK.md', 'docs/qa/inventory.json',
    ]]
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    versions = {}
    for name in ("gradio", "pytest", "numpy", "pandas", "plotly", "requests", "pyarrow", "llama-cpp-python"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    env = os.environ.copy()
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    # A developer credential must not leak into a test request or its output.
    env.pop("SUPERMAG_LOGON", None)
    command = [sys.executable, "-B", "-m", "pytest", "-q", "-m", "not network and not live_model", "-p", "no:cacheprovider", f"--junitxml={output / 'pytest.xml'}"]
    with tempfile.TemporaryDirectory(prefix="geomag_verify_") as directory:
        checkout = Path(directory)
        for source in files:
            destination = checkout / source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        # Pin checks need historical script bytes, not a .git directory in the
        # disposable copy. Export exactly the immutable blobs named by launchers.
        pinned_hashes = {}
        for name in ('tools/kaggle_prompt_diagnostics.py', 'tools/kaggle_prompt_diagnostics_remaining.py'):
            tree = ast.parse((root / name).read_text(encoding='utf-8'))
            assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == 'qa_files' for t in n.targets))
            for commit, script, expected in ast.literal_eval(assignment.value):
                blob = subprocess.check_output(['git', 'show', f'{commit}:tools/{script}'], cwd=root)
                digest = hashlib.sha256(blob).hexdigest()
                if digest != expected:
                    raise RuntimeError(f'Pinned QA script hash mismatch: {commit}:tools/{script}')
                destination = checkout / 'tests' / 'qa_pins' / commit / script
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(blob)
                pinned_hashes[f'{commit}:tools/{script}'] = digest
        print(f"Evidence: {output}", flush=True)
        print(f"Disposable checkout: {checkout}", flush=True)
        with (output / "pytest.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=checkout, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
    changed = [name for name, digest in hashes.items() if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
    report = {"time_utc": stamp, "revision": revision, "source_sha256": hashes, "pinned_qa_sha256": pinned_hashes, "versions": versions, "python": sys.version, "command": command, "exit_code": returncode, "original_files_changed_during_test": changed, "scope": "Offline pytest in a disposable source copy. Does not verify real LLM, browser rendering or current service availability."}
    (output / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return returncode or (1 if changed else 0)


if __name__ == "__main__":
    raise SystemExit(main())
