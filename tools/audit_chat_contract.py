"""Read-only diagnosis of long-chat acceptance gaps; no model, network or repair.

Writes observations to verification/, not a passing acceptance certificate.
"""
from __future__ import annotations

import ast
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    import pandas as pd
    import agent_core as ac
    import gradio_app as ga

    before = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
              for p in ("agent_core.py", "gradio_app.py", "geomag_analyzer.py")}
    queries = {
        "kyoto_user": "Получить Kp/Dst из Kyoto за первый день",
        "kyoto_explicit": "Получи Kyoto Dst/Kp за 2024-09-10, статистику и график",
        "bou_user": "Загрузить BOU за 2024-09-10; явно сменить рабочую станцию. 6–12. Поочерёдно статистика X/Y/Z/F/H и графики для BOU с явной датой. Не просить запоминать прежние числа",
        "omni_user": "Получить OMNI за первый день",
        "return_user": "Вернуться к корректному handle и построить график",
    }
    routes = {name: {"query": query, **ac.explain_tool_selection(query)}
              for name, query in queries.items()}

    # Compare the old selector body with unchanged routing registries/helpers.
    # This comparison concerns selection only, not execution of the old agent.
    old_source = subprocess.check_output(
        ["git", "show", "3bca4a1:agent_core.py"], cwd=ROOT, text=True, encoding="utf-8-sig")
    def definitions(source):
        result = {}
        for node in ast.parse(source).body:
            names = []
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = [node.target.id]
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            for name in names:
                result[name] = ast.dump(node, include_attributes=False)
        return result
    old_defs = definitions(old_source)
    new_defs = definitions((ROOT / "agent_core.py").read_text(encoding="utf-8-sig"))
    dependencies = ("_PHYSICS_METRIC_ALIASES", "_PHYSICS_METRICS", "_METRIC_QUERY_EXTRA",
                    "_METRIC_SOURCE_TOOLS", "_METRIC_DECISIVE_TOOL", "_CAPABILITY_RULES",
                    "_ordered_unique", "_bulk_units", "_is_bulk_query", "_MATRIX_TOOLS",
                    "TOOLS_PER_TASK", "_COMPUTE_TOOLS", "_SOURCE_TOOLS", "_build_metric_query_patterns")
    unchanged = all(old_defs[name] == new_defs[name] for name in dependencies)
    assert unchanged, "Re-evaluate baseline comparison: a routing dependency changed"
    old_node = next(n for n in ast.parse(old_source).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_selection_plan")
    old_ns = dict(vars(ac))
    exec(compile(ast.Module(body=[old_node], type_ignores=[]), "baseline_selector", "exec"), old_ns)
    old_names, old_info = old_ns["_selection_plan"](queries["kyoto_user"])
    baseline = {"revision": "3bca4a1", "selected_names": old_names, "metrics": old_info["metrics"],
                "routing_dependency_definitions_unchanged": unchanged,
                "scope": "Historical selector body with current unchanged routing dependencies; no old runtime execution"}

    store = ac.FrameStore()
    def frame(day):
        return pd.DataFrame({"timestamp": pd.date_range(day, periods=2, freq="h"),
                             "X": [3., 3.], "Y": [4., 4.], "Z": [12., 12.]})
    store.put("raw:irt:2024-09-10", frame("2024-09-10"))
    store.put("raw:irt:2024-01-01", frame("2024-01-01"))
    history = [{"role": "user", "content": "FIRST_REQUEST_ANCHOR IRT 2024-09-10"},
               {"role": "assistant", "content": "Первый результат."}]
    for i in range(4):
        history.extend([{"role": "user", "content": f"Другое сообщение {i}"},
                        {"role": "assistant", "content": f"Другой ответ {i}"}])
    recap = ga._context_from_history(history, store)

    # A deterministic adversarial brain shows what the runtime accepts. It
    # neither predicts Qwen's frequency of failure nor proves these claims true.
    with contextlib.redirect_stdout(io.StringIO()):
        summary = ac.run_agent(
            "Попросить краткую сводку с использованными станциями/датами/файлами. Сверить с журналом и диском",
            brain=ac.ScriptedBrain([{"content": "Проверил журнал и диск. Все файлы существуют."}]),
            verbose=False, store=store)
        refusal = ac._with_tool_failures(
            "Минимум Dst -20 nT, максимум 10 nT.",
            [{"tool": "get_statistics", "ok": False, "error": "tool_not_available",
              "message": "statistics were not executed"}])

    after = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in before}
    assert before == after, "Audit must not alter production sources"
    report = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "acceptance": "NOT_ACCEPTED",
        "scope": "Offline diagnosis only. No Qwen, GPU, network, real artifacts or browser reset reproduced.",
        "production_sources_unchanged": before == after,
        "source_sha256": before,
        "routes": routes,
        "baseline_kyoto_selector": baseline,
        "history": {"first_request_text_retained": "FIRST_REQUEST_ANCHOR" in recap,
                    "first_table_retained": store.get("raw:irt:2024-09-10") is not None,
                    "recap": recap, "raw_alias": store.aliases().get("raw"),
                    "note": "Table existence is not a durable record of the first requested day; alias means most recent write, not first request."},
        "unverified_summary": {"accepted_ok": summary.get("ok"),
                               "tool_calls": summary.get("tool_calls"), "text": summary.get("text")},
        "failed_statistics_text": refusal,
        "http_fixture_bypasses_production_router": "ac.select_tools_for_query = lambda query: ac.TOOL_SCHEMAS" in (ROOT / "tools/verify_gradio.py").read_text(encoding="utf-8"),
    }
    target = ROOT / "verification" / "20261102_long_chat_audit"
    target.mkdir(parents=True, exist_ok=True)
    (target / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
