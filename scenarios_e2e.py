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
import re
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
    # Rewrite every frame handle the model emits to upper case. The store must
    # treat "raw:IRT:2024-09-10" and "raw:irt:2024-09-10" as one handle, so this
    # turns a casing assumption into something the real run has to satisfy.
    uppercase_handles: bool = False
    # The final answer must state how many charts exist, in the exact form
    # "Построено графиков: N", so a truncated run cannot read as complete.
    require_plot_tally: bool = False
    # The run may fumble a tool call on the way, as long as it recovers. The
    # agent picking a wrong tool first is not the failure; ending up with no
    # answer is. Kept separate from the default because a scenario that tolerates
    # errors must still fail when the answer is wrong.
    allow_recovered_errors: bool = False
    # Strings that must NOT reach the user-facing answer. These are the internal
    # error codes and the red footer heading; a run whose answer leaks them has
    # failed the UX even when the number underneath is right.
    forbid_in_answer: list[str] = field(default_factory=list)
    # (min, max) bounds every "N нТ" number in the answer must fall inside. A
    # regex over the digits would pass on 60631 appearing where 33.32 belongs;
    # bounds say what magnitude the quantity must actually have.
    expect_nt_between: list[tuple[float, float]] = field(default_factory=list)
    # Regexes the answer must match. Needed where the natural phrasing has more
    # than one accepted form -- a model may write "10 сентября 2024" or
    # "2024-09-10", and pinning the ISO form alone would fail a correct answer
    # for a formatting choice.
    text_matches: list[str] = field(default_factory=list)


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
    Scenario(
        name="compare_uppercase",
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
        uppercase_handles=True,
        require_plot_tally=True,
        explanation=(
            "the reported failure: the model shouts every handle, e.g. "
            "'RAW:IRT:2024-09-10'. Casing must not change which frame is used, and "
            "every handle a tool advertises must be the key the store really holds"
        ),
    ),
    Scenario(
        name="three_days",
        query=(
            "Построй три отдельных графика горизонтальной составляющей H "
            "обсерватории IRT: за 10, за 11 и за 12 сентября 2024 года, по одному "
            "графику на каждый день."
        ),
        must_call=[
            "fetch_observatory_data",
            "fetch_observatory_data",
            "fetch_observatory_data",
            "plot_components",
            "plot_components",
            "plot_components",
        ],
        min_plots=3,
        text_contains=["2024-09-10", "2024-09-11", "2024-09-12"],
        uppercase_handles=True,
        require_plot_tally=True,
        explanation=(
            "three charts cost nine calls; the tool budget used to cut the turn "
            "short, and a run that saves two of three must say so rather than "
            "claim all three are ready"
        ),
    ),
    Scenario(
        name="median_F",
        query=(
            "Посчитай медиану полной компоненты F обсерватории IRT за 10 сентября "
            "2024 года. Сколько это нанотесла?"
        ),
        must_call=["fetch_observatory_data", "get_statistics"],
        text_contains=["F"],
        forbid_in_answer=[
            "unknown_frame",
            "unknown_component",
            "Не удалось выполнить",
        ],
        allow_recovered_errors=True,
        expect_nt_between=[(55000.0, 65000.0)],
        explanation=(
            "the reported failure: F is a native raw column, so get_statistics on "
            "the raw handle answers directly. The model asked "
            "calculate_derived_components for F -- an H/D/I-only tool -- and then "
            "passed a derived handle that was never created. It recovered and the "
            "number was right, but the answer carried two red errors"
        ),
    ),
    Scenario(
        name="range_F",
        query=(
            "Посчитай размах полной компоненты F обсерватории IRT за 10 сентября "
            "2024 года — разницу между максимумом и минимумом. Сколько это "
            "нанотесла?"
        ),
        must_call=["fetch_observatory_data", "calculate_derived_math"],
        text_contains=["F"],
        forbid_in_answer=[
            "unknown_frame",
            "unknown_component",
            "Не удалось выполнить",
        ],
        allow_recovered_errors=True,
        expect_nt_between=[(1.0, 500.0)],
        explanation=(
            "the reported failure: the range of F belongs to calculate_derived_math "
            "with metric='delta', not to calculate_derived_components. Same "
            "recovery, same red errors in the UI"
        ),
    ),
    Scenario(
        name="compare_range_F",
        query=(
            "Сравни размах полной компоненты F обсерватории IRT за 10 и за "
            "11 сентября 2024 года: сколько нанотесла разница между этими двумя "
            "размахами?"
        ),
        must_call=[
            "fetch_observatory_data",
            "fetch_observatory_data",
            "calculate_derived_math",
        ],
        text_contains=["F"],
        forbid_in_answer=[
            "unknown_frame",
            "unknown_component",
            "Не удалось выполнить",
        ],
        allow_recovered_errors=True,
        expect_nt_between=[(0.0, 500.0)],
        text_matches=[
            r"10\s*сентября|2024-09-10",
            r"11\s*сентября|2024-09-11",
        ],
        explanation=(
            "two days, each needing its own range, then a difference. The "
            "second fetch must not overwrite the first, or the two ranges come "
            "out identical"
        ),
    ),
]

#: Handle arguments that must resolve regardless of the casing used.
_HANDLE_ARGS = ("df", "df1", "df2")


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

    # 1. the required calls happened, in order. A missing call does not stop the
    # audit: the remaining checks still run, so one run reports every problem
    # instead of only the first.
    cursor = 0
    for wanted in scenario.must_call:
        while cursor < len(calls) and calls[cursor] != wanted:
            cursor += 1
        if cursor == len(calls):
            problems.append(
                f"expected tool {wanted!r} after position {cursor}, "
                f"but the log was {calls}"
            )
            break
        cursor += 1

    # 2. nothing failed
    for entry in log:
        if not entry.get("ok") and not scenario.allow_recovered_errors:
            problems.append(
                f"tool {entry['tool']} failed: {entry.get('error')} -- "
                f"{entry.get('message', '')}"
            )

    # 2a. a scenario that tolerates detours still fails if nothing worked: the
    # point is recovery, not silence. Checked explicitly so a run where every
    # single call failed cannot pass by having no red errors to complain about.
    if scenario.allow_recovered_errors and log and not any(e.get("ok") for e in log):
        problems.append("every tool call failed, so the agent never recovered")

    # 2b. a handle that was merely spelled differently must not count as a
    # failure. A real miss has to be an unknown_frame with an empty resolution,
    # which is what the store reports when nothing is stored under any casing.
    # A recovering scenario is judged on its answer instead: probing a handle
    # that does not exist yet is precisely the detour being tolerated.
    for entry in log:
        if scenario.allow_recovered_errors:
            break
        if entry.get("error") != "unknown_frame":
            continue
        for key in _HANDLE_ARGS:
            handle = entry.get("arguments", {}).get(key)
            if handle is None:
                continue
            if not str(handle).strip():
                continue
            problems.append(
                f"{entry['tool']}({key}={handle!r}) could not resolve a frame"
            )

    # 3. every requested day was fetched, as a different window
    expected_fetches = scenario.must_call.count("fetch_observatory_data")
    if expected_fetches > 1:
        fetches = _fetches(result)
        if len(fetches) < expected_fetches:
            problems.append(
                f"expected {expected_fetches} fetches, got {len(fetches)}"
            )
        dates = {
            (e["arguments"].get("start_date"), e["arguments"].get("end_date"))
            for e in fetches
        }
        if len(dates) < expected_fetches:
            problems.append(
                f"expected {expected_fetches} distinct windows, got {dates}"
            )

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

    # 4b. every handle a tool advertised must be a handle the model can reuse.
    # A tool that returns 'raw:IRT:...' while the store holds 'raw:irt:...' makes
    # the model copy a name that is absent from its own 'available' list.
    for entry in log:
        if not entry.get("ok"):
            continue
        produced = entry.get("frame_handle")
        if isinstance(produced, str) and produced != produced.lower():
            problems.append(
                f"{entry['tool']} advertised a non-canonical handle: {produced!r}"
            )

    # 5. artifacts exist
    if scenario.min_plots:
        plots = result.get("plots", [])
        if len(plots) < scenario.min_plots:
            problems.append(f"expected >= {scenario.min_plots} plot(s), got {plots}")
        for path in plots:
            if not Path(path).is_file():
                problems.append(f"plot artifact is missing: {path}")

    # 5b. the answer states the chart count, so a short run is not read as complete
    if scenario.require_plot_tally:
        answer = result.get("text") or ""
        if "Построено графиков" not in answer:
            problems.append("the answer never states how many charts were built")
        else:
            built = len(result.get("plots") or [])
            if f"Построено графиков: {built}" not in answer:
                problems.append(
                    f"the tally disagrees with the saved charts ({built})"
                )
            for path in result.get("plots") or []:
                if Path(path).name not in answer:
                    problems.append(
                        f"the answer does not list the saved file {Path(path).name}"
                    )

    # 6. the answer says what it must
    text = (result.get("text") or "").lower()
    for needle in scenario.text_contains:
        if needle.lower() not in text:
            problems.append(f"final answer never mentions {needle!r}")

    for pattern in scenario.text_matches:
        if not re.search(pattern, result.get("text") or "", flags=re.IGNORECASE):
            problems.append(f"final answer never matches {pattern!r}")

    # 7. internal error codes must not leak into the user-facing answer. The
    # number being right does not excuse the answer also being littered with
    # "unknown_frame": a user who reads that cannot tell it was fine.
    answer = result.get("text") or ""
    for needle in scenario.forbid_in_answer:
        if needle.lower() in answer.lower():
            problems.append(
                f"the answer leaks {needle!r} to the user: a correct number "
                f"arrived wrapped in internal errors"
            )

    # 8. the quantities must have the right magnitude. A regex over digits
    # passes when 60631 shows up in a sentence about a 33 nT range, so the
    # number is parsed out and bounded instead.
    numbers = [float(m) for m in re.findall(r"(\d+(?:[.,]\d+)?)\s*нТ", answer, re.I)]
    for low, high in scenario.expect_nt_between:
        if not numbers:
            problems.append(
                f"the answer states no quantity in nT, so {low}-{high} nT is "
                f"unverified"
            )
        elif not any(low <= value <= high for value in numbers):
            problems.append(
                f"no nT value in the answer falls in {low}-{high}: {numbers}"
            )

    return problems


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
_HANDLE_RE = re.compile(
    r'("(?:df|df1|df2)"\s*:\s*")([^"]*)(")',
    flags=re.IGNORECASE,
)


class _ShoutingBrain:
    """Wrap a brain and upper-case every frame handle it is about to emit.

    The casing is changed in the raw text the model produced, so the whole
    pipeline -- the parser, the store and the tools -- sees exactly what a model
    that shouts would send. Handles are not invented: only values of ``df``,
    ``df1`` and ``df2`` are rewritten, never the tool name or a date.
    """

    def __init__(self, brain: Any) -> None:
        self._brain = brain

    def __call__(self, messages: Any) -> Any:
        reply = self._brain(messages)
        content = getattr(reply, "content", None)
        if isinstance(content, str) and '"df' in content.lower():
            reply.content = _HANDLE_RE.sub(
                lambda m: m.group(1) + m.group(2).upper() + m.group(3), content
            )
        return reply

    def __getattr__(self, name: str) -> Any:
        return getattr(self._brain, name)


def run_scenario(scenario: Scenario, brain: Any, verbose: bool) -> bool:
    print("=" * 78)
    print(f"SCENARIO {scenario.name}")
    print(f"  {scenario.explanation}")
    print(f"  query: {scenario.query}")
    if scenario.uppercase_handles:
        print("  NOTE: every df/df1/df2 handle is upper-cased on the way in")
    print("=" * 78)

    target = _ShoutingBrain(brain) if scenario.uppercase_handles else brain
    result = ac.run_agent(scenario.query, brain=target, verbose=verbose)

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
