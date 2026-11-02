"""Check notebook syntax and failure cleanup without a GPU, model, or network."""
from contextlib import redirect_stdout, redirect_stderr
import io
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import subprocess
from types import SimpleNamespace
from unittest.mock import patch


def main():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "kaggle_portfolio_acceptance.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 9
    documented = re.findall(r"```python\s*\n(.*?)```", (root / "docs/KAGGLE_ACCEPTANCE.md").read_text(encoding="utf-8"), re.S)
    assert cells == [source.rstrip() for source in documented], "Notebook and documentation differ"
    assert cells[0] == (root / "tools/kaggle_checkout.py").read_text(encoding="utf-8").rstrip()
    for i, source in enumerate(cells):
        compile(source, f"cell-{i+1}", "exec")
    checks = ["Nine code cells compile"]

    # Existing environment value must survive a missing Kaggle Secret.
    for existing, secret in [("fixture-existing", None), (None, None), (None, "fixture-secret")]:
        calls = []
        def get_secret(_):
            if secret is None:
                raise RuntimeError("fixture secret unavailable")
            return secret
        def fetch(*args, **kwargs):
            calls.append(kwargs["logon"])
            return {"ok": False, "error": "fixture_unavailable"}
        modules = {"supermag_loader": SimpleNamespace(fetch_supermag_data=fetch, is_error=lambda x: x.get("ok") is False),
                   "kaggle_secrets": SimpleNamespace(UserSecretsClient=lambda: SimpleNamespace(get_secret=get_secret))}
        with patch.dict(os.environ, {"SUPERMAG_LOGON": existing} if existing else {}, clear=True), patch.dict(sys.modules, modules):
            output = io.StringIO()
            with redirect_stdout(output):
                exec(cells[3], {})
            expected = existing or secret
            assert calls == ([expected] if expected else []), calls
            assert not expected or expected not in output.getvalue()
    checks.append("SuperMAG: existing env, absent secret, configured secret; value never printed")

    verification = root / "verification"
    verification.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="notebook_fixture_", dir=verification) as directory:
        fixture = Path(directory).resolve()
        assert fixture.is_relative_to(verification.resolve())
        original_cwd = Path.cwd()
        original_sys_path = sys.path[:]
        try:
            upstream = fixture / "upstream"
            upstream.mkdir()
            def git(*args):
                return subprocess.check_output(["git", "-C", str(upstream), *args], text=True, stderr=subprocess.STDOUT).strip()
            git("init", "--initial-branch=master")
            required = ["agent_core.py", "stress_300.py", "gradio_app.py", "requirements-dev.txt",
                        "tools/verify_offline.py", "tools/verify_gradio.py"]
            for name in required:
                path = upstream / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# source fixture, never imported\n", encoding="utf-8")
            protocol = upstream / "docs/acceptance_protocol.json"
            protocol.parent.mkdir()
            protocol.write_text(json.dumps({"version": "github-20261102-v1"}), encoding="utf-8")
            git("add", ".")
            git("-c", "user.name=Notebook Test", "-c", "user.email=fixture@example.invalid", "commit", "-m", "fixture")
            expected_head = git("rev-parse", "HEAD")
            working = fixture / "kaggle/working"
            old_models = working / "geomagnetic-agent/models"
            old_models.mkdir(parents=True)
            weights = old_models / "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
            weights.write_bytes(b"fixture weights, not a real model")
            preserved = working / "geomagnetic-agent/projects/old_result.txt"
            preserved.parent.mkdir()
            preserved.write_text("preserve existing work", encoding="utf-8")
            source = cells[0].replace("/kaggle/working", working.as_posix()).replace(
                "https://github.com/IgorCim/geomagnetic-agent.git", upstream.as_posix())
            namespace = {}
            created = []
            for _ in range(2):
                with patch.dict(os.environ, {"GEOMAG_MODELS_DIR": str(old_models)}, clear=False), redirect_stdout(io.StringIO()):
                    exec(source, namespace)
                created.append(namespace["DIR"])
                assert namespace["source"]["git_revision"] == expected_head
                assert namespace["MODELS"] == old_models.resolve()
                for name, digest in namespace["source"]["sha256"].items():
                    assert hashlib.sha256((namespace["DIR"] / name).read_bytes()).hexdigest() == digest
                os.chdir(original_cwd)
                sys.path[:] = original_sys_path
            assert created[0] != created[1]
            assert preserved.read_text() == "preserve existing work"
            assert weights.read_bytes() == b"fixture weights, not a real model"
            for loaded_brain in (False, True):
                count_before = len(list(working.glob("geomagnetic_github_*")))
                modules = {} if loaded_brain else {"agent_core": SimpleNamespace()}
                guard_namespace = {"brain": object()} if loaded_brain else {}
                with patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()):
                    try:
                        exec(source, guard_namespace)
                    except AssertionError:
                        pass
                    else:
                        raise AssertionError("Checkout must reject a loaded model/module")
                assert len(list(working.glob("geomagnetic_github_*"))) == count_before
            checks.append("Git checkout: local upstream, repeat runs, exact HEAD/hashes, old weights/results preserved, loaded kernel rejected")
            os.chdir(fixture)
            for failure in (False, True):
                evidence = fixture / ("failure" if failure else "success")
                evidence.mkdir()
                fake = ("from pathlib import Path\nimport project_store as ps\n"
                        "RUN_DIR = Path(DIR) / 'fake_run'\n"
                        "ps.EXPORTS_ROOT = 'fixture temporary export root'\n")
                if failure:
                    fake += "raise AssertionError('fixture stress failure')\n"
                (fixture / "stress_300.py").write_text(fake, encoding="utf-8")
                ps = SimpleNamespace(EXPORTS_ROOT="fixture original root")
                namespace = {"evidence": evidence, "ps": ps, "brain": object(), "DIR": fixture, "json": json}
                with patch.dict(sys.modules, {"project_store": ps}), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    try:
                        exec(cells[6], namespace)
                    except AssertionError:
                        assert failure
                    else:
                        assert not failure
                assert ps.EXPORTS_ROOT == "fixture original root"
                assert namespace["stress_ns"] == {}
                assert namespace["RUN_DIR"] == fixture / "fake_run"
                status = json.loads((evidence / "stress_status.json").read_text())
                assert status["status"] == ("FAIL" if failure else "PASS")
                assert "fixture stress failure" in (evidence / "stress_300.log").read_text() if failure else True
        finally:
            os.chdir(original_cwd)
            sys.path[:] = original_sys_path
    checks.append("Stress PASS/FAIL: preserve exception, status/run dir, restore export root, release namespace")
    output = verification / "20261102_notebook_review"
    output.mkdir(exist_ok=True)
    report = {"status": "PASS", "checks": checks, "scope": "Synthetic notebook plumbing only; no real Qwen, CUDA, remote data or full stress executed"}
    (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
