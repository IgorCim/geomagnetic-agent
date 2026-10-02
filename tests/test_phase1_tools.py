"""Integration contour for the three Phase 1 tools.

These run the real handler dispatch path -- the same entry point the agent loop
uses -- so a tool that returns a bare dict instead of a ``(payload, note)``
tuple, or that stores an unplottable column, fails here rather than in front of
a user.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import agent_core as ac


@pytest.fixture
def store():
    """A FrameStore holding one realistic day, plus its slot name."""
    n = 240
    rng = np.random.default_rng(5)
    step = np.arange(n)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-09-10", periods=n, freq="min", tz="UTC"),
            "X": 25000 + np.sin(step / 20.0) * 60 + rng.normal(0, 3, n),
            "Y": -2000 + np.cos(step / 31.0) * 40 + rng.normal(0, 3, n),
            "Z": 45000 + np.sin(step / 45.0) * 80 + rng.normal(0, 4, n),
        }
    )
    frame["F"] = np.sqrt(frame["X"] ** 2 + frame["Y"] ** 2 + frame["Z"] ** 2)
    frames = ac.FrameStore()
    slot = frames.put("raw:irt:2024-09-10", frame)
    return frames, slot


def call(frames, tool, **args):
    """Invoke a handler and assert the (payload, note) contract holds."""
    payload, note = ac.HANDLERS[tool](args, frames)
    assert isinstance(note, str) and note, f"{tool} returned no note"
    assert isinstance(payload, dict), f"{tool} returned {type(payload).__name__}"
    return payload


# --------------------------------------------------------------------------- #
# calculate_derived_math
# --------------------------------------------------------------------------- #
def test_delta_returns_a_scalar(store):
    frames, slot = store
    payload = call(frames, "calculate_derived_math", df=slot, metric="delta", component="H")
    assert payload["ok"] is not False
    assert isinstance(payload["delta"], float)
    assert payload["unit"] == "nT"


def test_delta_equals_the_true_range(store):
    """The scalar is JSON-rounded to 2 decimals, so compare at that precision.

    Rounding is deliberate: the payload is handed straight to an LLM and bare
    floats are noise at 6 significant figures for a field measured in nT.
    """
    frames, slot = store
    payload = call(frames, "calculate_derived_math", df=slot, metric="delta", component="H")
    frame = frames.get(slot)
    h = np.hypot(frame["X"], frame["Y"])
    assert payload["delta"] == pytest.approx(h.max() - h.min(), abs=0.01)


def test_windowed_delta_stores_a_frame(store):
    frames, slot = store
    payload = call(
        frames, "calculate_derived_math", df=slot, metric="delta", component="H", window=60
    )
    assert payload["n_blocks"] == 4
    handle = payload["frame_handle"]
    assert handle in frames.names()
    assert len(frames.get(handle)) == 4


def test_dh_dt_stores_a_plottable_column(store):
    frames, slot = store
    payload = call(frames, "calculate_derived_math", df=slot, metric="dH_dt", component="H")
    assert payload["unit"] == "nT/min"
    handle = payload["frame_handle"]
    assert "DH_DT" in frames.get(handle).columns


def test_anomaly_needs_a_baseline_and_says_so(store):
    """The error must be actionable, not a generic failure."""
    frames, slot = store
    payload = call(frames, "calculate_derived_math", df=slot, metric="anomaly", component="H")
    assert payload["ok"] is False
    assert payload["error"] == "missing_baseline"
    assert "calculate_baseline" in payload["hint"]


def test_anomaly_subtracts_the_supplied_baseline(store):
    frames, slot = store
    payload = call(
        frames,
        "calculate_derived_math",
        df=slot,
        metric="anomaly",
        component="H",
        baseline_value=25000.0,
    )
    frame = frames.get(payload["frame_handle"])
    expected = np.hypot(frames.get(slot)["X"], frames.get(slot)["Y"]) - 25000.0
    np.testing.assert_allclose(frame["ANOMALY"].to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_unknown_metric_is_refused_with_the_valid_set(store):
    frames, slot = store
    payload = call(frames, "calculate_derived_math", df=slot, metric="nonsense")
    assert payload["error"] == "unknown_metric"
    assert set(payload["available"]) == {"delta", "anomaly", "dH_dt"}


def test_unknown_handle_is_refused(store):
    frames, _ = store
    payload = call(frames, "calculate_derived_math", df="raw:irt:1999-01-01", metric="delta")
    assert payload["error"] == "unknown_frame"


# --------------------------------------------------------------------------- #
# calculate_baseline
# --------------------------------------------------------------------------- #
def test_night_baseline_is_the_quiet_window(store):
    frames, slot = store
    payload = call(frames, "calculate_baseline", df=slot, component="H", mode="night")
    assert payload["mode"] == "night"
    assert payload["night_hours"] == [0, 4]
    assert payload["n_samples"] == 240  # the fixture is entirely within 00:00-04:00
    assert payload["baseline"] > 24000


def test_full_baseline_covers_every_sample(store):
    frames, slot = store
    payload = call(frames, "calculate_baseline", df=slot, component="H", mode="full")
    assert payload["n_finite"] == 240
    assert payload["baseline"] is not None


def test_night_window_can_be_customised(store):
    frames, slot = store
    payload = call(
        frames,
        "calculate_baseline",
        df=slot,
        component="H",
        mode="night",
        night_hours=[22, 6],
    )
    assert payload["night_hours"] == [22, 6]
    assert payload["n_samples"] == 240


def test_a_malformed_night_window_is_refused(store):
    frames, slot = store
    payload = call(
        frames, "calculate_baseline", df=slot, component="H", mode="night", night_hours=[1]
    )
    assert payload["ok"] is False


def test_unknown_mode_is_refused(store):
    frames, slot = store
    payload = call(frames, "calculate_baseline", df=slot, mode="bogus")
    assert payload["error"] == "unknown_mode"
    assert set(payload["available"]) == {"night", "full"}


def test_baseline_result_feeds_straight_into_the_anomaly_metric(store):
    """The documented two-step workflow must actually compose."""
    frames, slot = store
    baseline = call(frames, "calculate_baseline", df=slot, component="H", mode="night")
    anomaly = call(
        frames,
        "calculate_derived_math",
        df=slot,
        metric="anomaly",
        component="H",
        baseline_value=baseline["baseline"],
    )
    assert anomaly["ok"] is not False
    assert "ANOMALY" in frames.get(anomaly["frame_handle"]).columns


# --------------------------------------------------------------------------- #
# evaluate_custom_formula
# --------------------------------------------------------------------------- #
def test_reference_formula_succeeds_and_stores(store):
    frames, slot = store
    payload = call(
        frames, "evaluate_custom_formula", df=slot, formula="sqrt(X**2 + Y**2) + Z/10"
    )
    assert payload["formula"] == "sqrt(X**2 + Y**2) + Z/10"
    assert payload["n_finite"] == 240
    assert payload["column"] == "FORMULA"


def test_formula_matches_numpy(store):
    frames, slot = store
    call(frames, "evaluate_custom_formula", df=slot, formula="sqrt(X**2 + Y**2) + Z/10")
    frame = frames.get("formula:irt:2024-09-10")
    source = frames.get(slot)
    expected = np.sqrt(source["X"] ** 2 + source["Y"] ** 2) + source["Z"] / 10.0
    np.testing.assert_allclose(frame["FORMULA"].to_numpy(), expected.to_numpy(), rtol=1e-12)


@pytest.mark.parametrize(
    "payload_text",
    [
        "__import__('os').system('rm -rf /')",
        "open('/etc/passwd')",
        "X.__class__",
        "eval('1')",
        "globals()",
    ],
)
def test_injection_reaches_the_tool_layer_and_is_refused(store, payload_text):
    """Security must hold at the tool boundary, not only in the module tests."""
    frames, slot = store
    before = set(frames.names())
    payload = call(frames, "evaluate_custom_formula", df=slot, formula=payload_text)
    assert payload["ok"] is False
    assert payload["error"] == "formula_parse_error"
    assert set(frames.names()) == before, "a refused formula must not store anything"


def test_a_refused_formula_teaches_the_grammar(store):
    """The model gets the accepted vocabulary so it can self-correct."""
    frames, slot = store
    payload = call(frames, "evaluate_custom_formula", df=slot, formula="open('x')")
    assert "sqrt" in payload["allowed_functions"]
    assert "X" in payload["allowed_columns"]
    assert "**" in payload["allowed_operators"]


def test_a_missing_formula_is_refused(store):
    frames, slot = store
    payload = call(frames, "evaluate_custom_formula", df=slot, formula="")
    assert payload["ok"] is False and payload["error"] == "missing_formula"


def test_formula_output_is_plot_through_the_documented_call(store, tmp_path):
    """The handler promises the result is plottable; that promise is checked here.

    The column is upper case because geomag_plotter upper-cases requested names,
    so a lower-case column would be unreachable -- and an unreachable promise is
    worse than no promise.
    """
    frames, slot = store
    call(frames, "evaluate_custom_formula", df=slot, formula="sqrt(X**2 + Y**2)")
    handle = "formula:irt:2024-09-10"
    payload = call(
        frames,
        "plot_components",
        df=handle,
        components=["FORMULA"],
        filename=str(tmp_path / "formula_plot.html"),
    )
    assert payload["ok"] is True


def test_formula_can_build_a_component_the_tools_lack(store, tmp_path):
    """A custom series must flow onward into another tool, not dead-end."""
    frames, slot = store
    call(
        frames,
        "evaluate_custom_formula",
        df=slot,
        formula="degrees(atan2(Z, sqrt(X**2 + Y**2)))",
    )
    handle = "formula:irt:2024-09-10"
    payload = call(
        frames, "get_statistics", df=handle, components=["FORMULA"], digits=2
    )
    assert "FORMULA" in payload
    plotted = call(
        frames,
        "plot_components",
        df=handle,
        components=["FORMULA"],
        filename=str(tmp_path / "inclination.html"),
    )
    assert plotted["ok"] is True


def test_formula_does_not_mutate_the_source_frame(store):
    frames, slot = store
    before = list(frames.get(slot).columns)
    call(frames, "evaluate_custom_formula", df=slot, formula="X + Y")
    assert list(frames.get(slot).columns) == before
    assert "FORMULA" in frames.get("formula:irt:2024-09-10").columns


def test_two_formulas_share_a_handle_without_clobbering(store):
    """Both writes land in one family; the newest is the alias target."""
    frames, slot = store
    first = call(frames, "evaluate_custom_formula", df=slot, formula="X + Y")
    second = call(frames, "evaluate_custom_formula", df=slot, formula="X - Y")
    assert first["frame_handle"] == second["frame_handle"]
    frame = frames.get(second["frame_handle"])
    np.testing.assert_allclose(frame["FORMULA"], frames.get(slot)["X"] - frames.get(slot)["Y"])


# --------------------------------------------------------------------------- #
# registry integrity
# --------------------------------------------------------------------------- #
def test_the_three_new_tools_are_wired_end_to_end():
    for tool in ("calculate_derived_math", "calculate_baseline", "evaluate_custom_formula"):
        assert tool in ac.TOOLS_BY_NAME
        assert tool in ac.HANDLERS


def test_the_formula_tool_description_publishes_the_grammar():
    """Qwen can only write a valid formula if the schema states the rules."""
    schema = next(
        s for s in ac.TOOL_SCHEMAS if s["function"]["name"] == "evaluate_custom_formula"
    )
    description = schema["function"]["description"]
    assert "sqrt" in description
    assert "never executed" in description
    assert schema["function"]["parameters"]["required"] == ["df", "formula"]