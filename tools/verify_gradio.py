"""Exercise the actual Gradio HTTP queue with deterministic data, not an LLM.

Run locally with ``python -B tools/verify_gradio.py``. ``--serve`` keeps the
test interface open for browser inspection. Never use this fixture as a live
Qwen portfolio demonstration: both the brain and loader are test doubles.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GEOMAG_DEBUG_PARSER"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=7861)
    options = parser.parse_args()
    import numpy as np
    import pandas as pd
    import project_store as ps
    import geomag_plotter as gp
    from gradio_client import Client

    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = root / "verification" / (stamp + "_gradio")
    evidence.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="geomag_ui_") as directory:
        scratch = Path(directory)
        ps.PROJECTS_ROOT = scratch / "projects"
        ps.EXPORTS_ROOT = scratch / "exports"
        gp.DEFAULT_OUTPUT_DIR = scratch / "plots"
        import agent_core as ac
        import gradio_app as ga
        ac._WORKSPACE = ps.Workspace(root=ps.PROJECTS_ROOT)
        ga.PLOTS_DIR = gp.DEFAULT_OUTPUT_DIR

        def fetch(station_code, start_date, end_date, **kwargs):
            times = pd.date_range(start_date, periods=24, freq="h", tz="UTC")
            x = 25000.0 + np.arange(24)
            frame = pd.DataFrame({"timestamp": times, "X": x, "Y": -2000.0, "Z": 45000.0})
            frame["F"] = np.hypot(np.hypot(frame["X"], frame["Y"]), frame["Z"])
            frame.attrs["source"] = "DETERMINISTIC UI TEST DATA"
            return frame

        ac.loader.fetch_observatory_data = fetch
        ac.OFFLINE = False

        class TestBrain:
            def chat(self, messages, tools, **kwargs):
                if any(m.get("role") == "tool" for m in messages):
                    return {"content": "TEST FIXTURE: tool finished; inspect the tool log."}
                query = messages[1]["content"].split("Новый запрос:\n")[-1].strip()
                if query == "fetch":
                    name, args = "fetch_observatory_data", {"station_code": "IRT", "start_date": "2024-09-10", "end_date": "2024-09-10"}
                elif query == "plot":
                    name, args = "plot_components", {"df_descriptor": "raw:irt:2024-09-10", "components": ["H"]}
                elif query == "project":
                    name, args = "create_project", {"name": "http_fixture"}
                elif query == "list projects":
                    name, args = "list_projects", {}
                else:
                    name, args = "get_statistics", {"df_descriptor": "raw:irt:2024-09-10", "components": ["X"]}
                return {"content": "<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>"}

        # This fixture controls only offered schemas; the production run_agent,
        # handlers, state, serialization, queue and UI wiring remain in use.
        original_selector = ac.select_tools_for_query
        ac.select_tools_for_query = lambda query: ac.TOOL_SCHEMAS
        demo = ga.build_demo(TestBrain())
        try:
            demo.launch(server_name="127.0.0.1", server_port=options.port, share=False, prevent_thread_lock=True, quiet=True)
            url = f"http://127.0.0.1:{options.port}"
            first = Client(url, verbose=False, download_files=False)
            second = Client(url, verbose=False, download_files=False)
            api = first.view_api(return_format="dict", print_info=False)
            (evidence / "api.json").write_text(json.dumps(api, ensure_ascii=False, indent=2), encoding="utf-8")
            print("API names:", list(api["named_endpoints"]), flush=True)
            names = [name for name in api["named_endpoints"] if name.startswith("/submit")]
            assert len(names) == 2, names
            checks = []

            def predict(client, query, history=None, endpoint=names[0]):
                return client.predict(query, history or [], "", api_name=endpoint)

            fetched = predict(first, "fetch")
            assert "fetch_observatory_data" in str(fetched), fetched
            history = fetched[0]
            plot = predict(first, "plot", history, names[1])
            assert "unknown_frame" not in str(plot), plot
            visible = [item.get("value") for item in plot[1:13] if isinstance(item, dict) and item.get("visible")]
            assert visible and all(isinstance(item, dict) and item.get("type") == "plotly" for item in visible), "No serialized Plotly figure in the visible slots"
            checks.append("Same HTTP session: button and textbox-submit endpoints; Plotly serialized")
            import requests
            downloads = plot[-2]
            assert downloads and isinstance(downloads[0], dict)
            downloaded = requests.get(downloads[0]["url"], timeout=15)
            assert downloaded.status_code == 200 and b"Plotly.newPlot" in downloaded.content
            checks.append("Actual HTML download route returns the complete chart")
            history = plot[0]
            for _ in range(12):
                response = predict(first, "statistics", history)
                assert "unknown_frame" not in str(response), response
                history = response[0]
            checks.append("14 HTTP turns: stored frame survives and transcript grows")
            fresh = predict(second, "statistics")
            assert "unknown_frame" in str(fresh), fresh
            checks.append("Second HTTP session cannot resolve the first session frame")
            created = predict(first, "project", history)
            first.predict(api_name="/lambda")  # the UI's Refresh button populates choices
            refreshed = first.predict("http_fixture", api_name="/lambda_1")
            assert "http_fixture" in str(refreshed), "New empty project could not be selected"
            archive = first.predict("http_fixture", api_name="/_make_zip")
            assert isinstance(archive, dict) and archive.get("url"), "ZIP download was not serialized"
            zipped = requests.get(archive["url"], timeout=15)
            assert zipped.status_code == 200
            import io
            import zipfile
            with zipfile.ZipFile(io.BytesIO(zipped.content)) as bundle:
                assert bundle.testzip() is None
                assert "export_manifest.json" in bundle.namelist()
            checks.append("Create/select empty project, build ZIP through UI endpoint, download and validate ZIP")
            blank = predict(first, "", history)
            assert blank is not None
            checks.append("Empty input does not crash the queued event")
            report = {"time_utc": stamp, "gradio": ga.GRADIO_VERSION, "checks": checks, "status": "PASS", "scope": "Real local Gradio HTTP queue; deterministic brain and station loader; not real Qwen or remote service validation."}
            (evidence / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
            print("EVIDENCE:", evidence, flush=True)
            if options.serve:
                print("TEST UI ONLY:", url, flush=True)
                while True:
                    time.sleep(1)
        finally:
            ac.select_tools_for_query = original_selector
            demo.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
