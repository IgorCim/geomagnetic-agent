"""End-to-end scenarios against the real Qwen brain.

Every scenario is a natural-language question, not a scripted tool sequence: the
model decides which tools to call and in which order. That is the point -- the
two-day failure this suite guards could only appear if the model itself
mis-sequenced the work.

Each scenario declares:
  * required tools, in the order they must appear in the call log;
  * the number of distinct frames the tool arguments must reference;
  * whether a plot artifact must exist;
  * an optional substring the final answer must contain.

Run on a machine that has the model and a GPU (Kaggle/Colab):

    python scenarios_e2e.py                          # all scenarios
    python scenarios_e2e.py --scenario compare       # one of them
    python scenarios_e2e.py --model /kaggle/...gguf  # explicit weights
    python scenarios_e2e.py --dry-run                # show the plan, load nothing

Exit code is non-zero if any scenario fails, so this is usable in CI.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import agent_core as ac


@dataclass
class Scenario:
    name: str
    query: str
    must_call: list[str]
    min_plots: int = 0
    text_contains: list[str] = field(default_factory=list)
    # arguments that must each reference a *different* frame handle
    distinct_args: list[str] = field(default_factory=list)
    explanation: str = ""


SCENARIOS: list[Scenario] = [
    Scenario(
        name="one_day",
        query=(
            "Скачай данные обсерватории IRT за 10 сентября 2024 года, посчитай "
            "горизонтальную составляющую H и построй график H за этот день."
        ),
        must_call=[
            "fetch_observatory_data",
            "calculate_derived_components",
            "plot_components",
        ],
        min_plots=1,
        explanation="single day: the classic flow, must still work unchanged",
    ),
    Scenario(
        name="compare",
        query=(
            "Сравни горизонтальную составляющую H обсерватории IRT за 10 и за "
            "11 сентября 2024 года: построй на одном графике оба дня, чтобы было "
            "видно, насколько сильно они отличаются."
        ),
        must_call=[
            "fetch_observatory_data",
            "fetch_observatory_data",
            "plot_comparison",
        ],
        min_plots=1,
        distinct_args=["df1", "df2"],
        text_contains=["2024-09-10", "2024-09-11"],
        explanation=(
            "the field failure: the second fetch used to overwrite the first, so "
            "the comparison had nothing to compare"
        ),
    ),
    Scenario(
        name="medians",
        query=(
            "Найди медиану горизонтальной составляющей H обсерватории IRT отдельно "
            "за 10 сентября 2024 и за 11 сентября 2024. Сколько нанотесла разница "
            "между этими двумя медианами?"
        ),
        must_call=[
            "fetch_observatory_data",
            "fetch_observatory_data",
            "get_statistics",
        ],
        text_contains=["2024-09-10", "2024-09-11"],
        explanation="two independent days, no chart needed -- numbers must be exact",
    ),
]


# --------------------------------------------------------------------------- #
# checking
# --------------------------------------------------------------------------- #
def _fetches(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in result["tool_calls"] if e["tool"] == "fetch_observatory_data"]


def check(scenario: Scenario, result: dict[str, Any]) -> list[str]:
    """Return a list of problems; empty means the scenario passed."""
    problems: list[str] = []
    log = result["tool_calls"]
    calls = [e["tool"] for e in log]

    # 1. the required calls happened, in order
    cursor = 0
    for wanted in scenario.must_call:
        while cursor < len(calls) and calls[cursor] != wanted:
            cursor += 1
        if cursor == len(calls):
            problems.append(
                f"expected tool {wanted!r} after position {cursor}, "
                f"but the log was {calls}"
            )
            return problems
        cursor += 1

    # 2. nothing failed
    for entry in log:
        if not entry.get("ok"):
            problems.append(
                f"tool {entry['tool']} failed: {entry.get('error')} -- "
                f"{entry.get('message', '')}"
            )

    # 3. two days really were fetched, as two different frames
    fetches = _fetches(result)
    if scenario.must_call.count("fetch_observatory_data") > 1:
        if len(fetches) < 2:
            problems.append(f"expected 2 fetches, got {len(fetches)}")
        dates = {
            (e["arguments"].get("start_date"), e["arguments"].get("end_date"))
            for e in fetches
        }
        if len(dates) < 2:
            problems.append(f"both fetches requested the same window: {dates}")

    # 4. the compared frames were named differently, not the same handle twice
    for entry in log:
        if entry["tool"] != "plot_comparison":
            continue
        handles = {entry["arguments"].get(key) for key in scenario.distinct_args}
        if len(handles) < len(scenario.distinct_args):
            problems.append(
                "plot_comparison was called with the same handle for both days: "
                f"{handles}"
            )

    # 5. artifacts exist
    if scenario.min_plots:
        plots = result.get("plots", [])
        if len(plots) < scenario.min_plots:
            problems.append(f"expected >= {scenario.min_plots} plot(s), got {plots}")
        for path in plots:
            if not Path(path).is_file():
                problems.append(f"plot artifact is missing: {path}")

    # 6. the answer says what it must
    text = (result.get("text") or "").lower()
    for needle in scenario.text_contains:
        if needle.lower() not in text:
            problems.append(f"final answer never mentions {needle!r}")

    return problems


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def run_scenario(scenario: Scenario, brain: Any, verbose: bool) -> bool:
    print("=" * 78)
    print(f"SCENARIO {scenario.name}")
    print(f"  {scenario.explanation}")
    print(f"  query: {scenario.query}")
    print("=" * 78)

    result = ac.run_agent(scenario.query, brain=brain, verbose=verbose)

    for index, entry in enumerate(result["tool_calls"], start=1):
        mark = "ok " if entry.get("ok") else "ERR"
        print(f"  {index}. [{mark}] {entry['tool']}({ac._short(entry['arguments'])})")
        if entry.get("error"):
            print(f"       error: {entry['error']} -- {entry.get('message', '')}")
            if entry.get("hint"):
                print(f"       hint : {entry['hint']}")
        if entry.get("plot"):
            print(f"       plot : {entry['plot']}")

    problems = check(scenario, result)

    print("-" * 78)
    print("ANSWER")
    print("-" * 78)
    print(result.get("text", ""))
    print()

    if problems:
        print(f"FAILED: {scenario.name}")
        for problem in problems:
            print(f"  - {problem}")
        return False
    print(f"PASSED: {scenario.name}")
    return True


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Real-brain end-to-end scenarios")
    parser.add_argument(
        "--scenario", action="append", choices=[s.name for s in SCENARIOS],
        help="run only this scenario (repeatable); default is all",
    )
    parser.add_argument("--model", help="path to the GGUF weights")
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-ctx", type=int, default=ac.N_CTX)
    parser.add_argument("--verbose", action="store_true", help="per-round agent log")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="list the scenarios and exit without loading the model",
    )
    args = parser.parse_args()

    chosen = [s for s in SCENARIOS if not args.scenario or s.name in args.scenario]
    if not chosen:
        print("no scenario matched", file=sys.stderr)
        return 2

    if args.dry_run:
        for scenario in chosen:
            print(f"{scenario.name:10s} {scenario.must_call}")
            print(f"{'':10s} {scenario.query}")
        return 0

    brain = ac.load_brain(
        args.model, n_gpu_layers=args.n_gpu_layers, n_ctx=args.n_ctx, verbose=True
    )
    if ac.is_error(brain):
        print(f"[e2e] brain unavailable: {brain['error']}: {brain['message']}")
        print("[e2e] this suite needs the real model on a GPU; use --dry-run otherwise")
        return 1

    passed = 0
    for scenario in chosen:
        if run_scenario(scenario, brain, args.verbose):
            passed += 1
        print()

    print("=" * 78)
    print(f"RESULT: {passed}/{len(chosen)} scenarios passed")
    print("=" * 78)
    return 0 if passed == len(chosen) else 1


if __name__ == "__main__":
    raise SystemExit(main())
