"""Validation contour for :mod:`geomag_math`.

Two jobs. First, the **physical identities** -- ``F >= H``, ``H >= 0``,
``|D| <= 180``, ``|I| <= 90`` and the synthetic ``X=1, Y=0 -> H=1, D=0`` case.
These are the invariants a geomagnetic tool must never violate; a regression in
any of them means the sign conventions or the vectorised formulas are wrong, and
would otherwise be invisible in a chart.

Second, the **DSL safety contour**: a battery of injection attempts that must
each be refused by the parser rather than executed. These assert on the error
*code* and on the absence of side effects, not merely that "something went
wrong" -- a formula that raised an unrelated exception would satisfy a weaker
assertion while still leaving the grammar too permissive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import geomag_math as gm


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def simple():
    """Three clean rows with an unambiguous analytic answer."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC"),
            "X": [1.0, 3.0, -4.0],
            "Y": [0.0, 4.0, 3.0],
            "Z": [0.0, 12.0, 0.0],
        }
    )


@pytest.fixture
def day():
    """A day of minute means with realistic magnitudes and injected NaNs."""
    n = 1440
    ts = pd.date_range("2024-09-10", periods=n, freq="min", tz="UTC")
    rng = np.random.default_rng(11)
    x = 25000 + np.sin(np.arange(n) / 180.0) * 80 + rng.normal(0, 4, n)
    y = -2000 + np.cos(np.arange(n) / 240.0) * 50 + rng.normal(0, 4, n)
    z = 45000 + np.sin(np.arange(n) / 300.0) * 120 + rng.normal(0, 5, n)
    x[17] = np.nan  # a realistic gap
    return pd.DataFrame({"timestamp": ts, "X": x, "Y": y, "Z": z})


@pytest.fixture
def extremes():
    """Magnitudes spanning sub-nT to planetary-scale, plus every sign."""
    rng = np.random.default_rng(3)
    n = 500
    scale = np.logspace(-3, 7, n)
    signs = rng.choice([-1.0, 1.0], size=(n, 3))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC"),
            "X": scale * signs[:, 0],
            "Y": scale * signs[:, 1],
            "Z": scale * signs[:, 2],
        }
    )


# --------------------------------------------------------------------------- #
# 1. physical identities -- THE required contour
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture_name", ["simple", "day", "extremes"])
def test_F_is_never_less_than_H(fixture_name, request):
    """F >= H for every row, because F**2 = H**2 + Z**2 with both terms >= 0."""
    df = request.getfixturevalue(fixture_name)
    out = gm.add_derived_components(df)
    assert not gm.is_error(out)
    valid = out[["F", "H"]].dropna()
    assert not valid.empty
    assert (valid["F"] >= valid["H"]).all(), "F must dominate H everywhere"


def test_H_is_never_negative(extremes):
    """H is a magnitude built with hypot, never a signed quantity."""
    df = extremes
    h = gm.horizontal_field(df)
    assert not gm.is_error(h)
    finite = h[np.isfinite(h)]
    assert (finite >= 0).all()
    out = gm.add_derived_components(df)
    assert (out["H"].dropna() >= 0).all()


def test_F_is_never_negative(extremes):
    df = extremes
    f = gm.total_field(df)
    assert not gm.is_error(f)
    assert (f[np.isfinite(f)] >= 0).all()


def test_declination_stays_within_plus_minus_180(extremes):
    """arctan2 returns the full circle directly; no post-hoc wrap needed."""
    d = gm.declination(extremes)
    assert not gm.is_error(d)
    finite = d[np.isfinite(d)]
    assert (finite.abs() <= 180.0 + 1e-12).all()


def test_inclination_stays_within_plus_minus_90(extremes):
    """A field cannot be steeper than vertical, so |I| is bounded by 90."""
    i = gm.inclination(extremes)
    assert not gm.is_error(i)
    finite = i[np.isfinite(i)]
    assert (finite.abs() <= 90.0 + 1e-12).all()


def test_negative_Z_still_keeps_F_above_H():
    """A negative Z tilts the field upward; it does not subtract from magnitude.

    This is the case a naive ``F = sqrt(H + Z)`` or a signed formulation gets
    wrong, so it is asserted explicitly rather than left to the sweep above.
    """
    df = pd.DataFrame({"X": [3.0], "Y": [4.0], "Z": [-100.0]})
    out = gm.add_derived_components(df)
    assert out["H"].iloc[0] == pytest.approx(5.0)
    assert out["F"].iloc[0] == pytest.approx(np.hypot(5.0, 100.0))
    assert out["F"].iloc[0] > out["H"].iloc[0]
    assert out["I"].iloc[0] == pytest.approx(-87.1376, abs=1e-3)


def test_synthetic_unit_field_H_equals_1_D_equals_0(simple):
    """The required synthetic case: X=1, Y=0 -> H=1, D=0."""
    out = gm.add_derived_components(simple)
    assert out["H"].iloc[0] == pytest.approx(1.0)
    assert out["D"].iloc[0] == pytest.approx(0.0)
    assert out["F"].iloc[0] == pytest.approx(1.0)
    assert out["I"].iloc[0] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "x, y, expected_h",
    [(1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (3.0, 4.0, 5.0), (-3.0, -4.0, 5.0), (0.0, 0.0, 0.0)],
)
def test_pythagorean_cases(x, y, expected_h):
    df = pd.DataFrame({"X": [x], "Y": [y], "Z": [0.0]})
    assert gm.horizontal_field(df).iloc[0] == pytest.approx(expected_h)


@pytest.mark.parametrize(
    "x, y, expected_d",
    [(1.0, 0.0, 0.0), (0.0, 1.0, 90.0), (-1.0, 0.0, 180.0), (0.0, -1.0, -90.0)],
)
def test_declination_quadrants(x, y, expected_d):
    """arctan rather than atan2 cannot express 180 or -90 at all."""
    df = pd.DataFrame({"X": [x], "Y": [y], "Z": [0.0]})
    assert gm.declination(df).iloc[0] == pytest.approx(expected_d)


def test_horizontal_only_field_has_inclination_of_90():
    """H == 0 makes I exactly +-90, not NaN from a 0/0 division."""
    df = pd.DataFrame({"X": [0.0], "Y": [0.0], "Z": [100.0]})
    out = gm.add_derived_components(df)
    assert out["I"].iloc[0] == pytest.approx(90.0)
    assert out["H"].iloc[0] == pytest.approx(0.0)


def test_identity_F_squared_equals_H_squared_plus_Z_squared(simple):
    """The defining relation, asserted on the algebra rather than the range."""
    out = gm.add_derived_components(simple).dropna()
    lhs = out["F"] ** 2
    rhs = out["H"] ** 2 + out["Z"] ** 2
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


def test_identity_H_squared_equals_X_squared_plus_Y_squared(simple):
    out = gm.add_derived_components(simple)
    np.testing.assert_allclose(out["H"] ** 2, out["X"] ** 2 + out["Y"] ** 2, rtol=1e-12)


def test_identity_tan_inclination_equals_Z_over_H(simple):
    out = gm.add_derived_components(simple).dropna()
    nonzero_h = out[out["H"] > 0]
    np.testing.assert_allclose(
        np.tan(np.radians(nonzero_h["I"])), nonzero_h["Z"] / nonzero_h["H"], rtol=1e-9
    )


# --------------------------------------------------------------------------- #
# 2. component construction behaviour
# --------------------------------------------------------------------------- #
def test_input_frame_is_not_mutated(day):
    """The loader cache must stay pristine, so a copy is returned.

    Compared with an array-aware equality rather than ``==`` on lists: the frame
    contains a NaN gap, and NaN != NaN would make a correct copy look mutated.
    """
    before = day["X"].to_numpy(copy=True)
    out = gm.add_derived_components(day)
    np.testing.assert_array_equal(day["X"].to_numpy(), before, strict=True)
    assert "H" not in day.columns
    assert "H" in out.columns
    assert out is not day


def test_derived_columns_are_idempotent(day):
    once = gm.add_derived_components(day)
    twice = gm.add_derived_components(once)
    np.testing.assert_allclose(once["H"].dropna(), twice["H"].dropna(), rtol=1e-12)


def test_nan_propagates_without_invention(day):
    """A gap in X must gap every dependent output rather than be interpolated."""
    out = gm.add_derived_components(day)
    assert np.isnan(out.loc[17, "H"])
    assert np.isnan(out.loc[17, "F"])
    assert np.isnan(out.loc[17, "D"])


def test_degenerate_rows_are_counted_in_attrs():
    """D is undefined where X == Y == 0; the count is reported, not hidden."""
    df = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [1.0, 1.0]})
    out = gm.add_derived_components(df)
    assert out.attrs["degenerate_horizontal_rows"] == 1


def test_missing_columns_returns_structured_error():
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=2), "X": [1.0, 2.0]})
    result = gm.add_derived_components(df)
    assert gm.is_error(result)
    assert result["error"] == "missing_columns"
    assert "Y" in result["missing"]


def test_empty_frame_is_rejected():
    result = gm.add_derived_components(pd.DataFrame({"X": [], "Y": [], "Z": []}))
    assert gm.is_error(result) and result["error"] == "empty_dataframe"


def test_non_dataframe_is_rejected():
    assert gm.is_error(gm.horizontal_field([1, 2, 3]))
    assert gm.is_error(gm.total_field("not a frame"))


def test_component_series_derives_on_demand(simple):
    assert gm.component_series(simple, "H").iloc[0] == pytest.approx(1.0)
    assert gm.component_series(simple, "I").iloc[0] == pytest.approx(0.0)


def test_component_series_rejects_unknown_name(simple):
    result = gm.component_series(simple, "Q")
    assert gm.is_error(result) and result["error"] == "unknown_component"


# --------------------------------------------------------------------------- #
# 3. variations and statistics
# --------------------------------------------------------------------------- #
def test_delta_is_max_minus_min(day):
    enriched = gm.add_derived_components(day)
    expected = enriched["H"].max() - enriched["H"].min()
    assert gm.calculate_delta(day, "H") == pytest.approx(expected)


def test_delta_of_constant_series_is_zero():
    df = pd.DataFrame({"X": [5.0] * 10, "Y": [5.0] * 10, "Z": [0.0] * 10})
    assert gm.calculate_delta(df, "H") == pytest.approx(0.0)


def test_delta_windowed_returns_one_row_per_block(day):
    result = gm.calculate_delta(day, "H", window=60)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 24  # 1440 samples / 60
    assert {"block", "min", "max", "delta"} <= set(result.columns)


def test_windowed_delta_never_exceeds_whole_range(day):
    whole = gm.calculate_delta(day, "H")
    windowed = gm.calculate_delta(day, "H", window=60)
    assert windowed["delta"].max() <= whole + 1e-12


def test_delta_rejects_zero_window(day):
    assert gm.is_error(gm.calculate_delta(day, "H", window=0))


def test_anomaly_subtracts_baseline(day):
    series = gm.calculate_anomaly(day, "H", 25000.0)
    assert not gm.is_error(series)
    assert series.iloc[0] == pytest.approx(
        gm.add_derived_components(day)["H"].iloc[0] - 25000.0
    )


def test_anomaly_requires_explicit_baseline(day):
    """No implicit zero: subtracting nothing would look like a correction."""
    result = gm.calculate_anomaly(day, "H")
    assert gm.is_error(result) and result["error"] == "missing_baseline"


def test_dH_dt_is_the_first_derivative_in_nT_per_minute():
    df = pd.DataFrame({"X": [1.0, 2.0, 4.0, 7.0], "Y": [0.0] * 4, "Z": [0.0] * 4})
    result = gm.calculate_dH_dt(df)
    assert not gm.is_error(result)
    # H is 1,2,4,7, so the first differences are +1, +2, +3. n values give
    # n-1 differences, hence exactly one leading NaN.
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(1.0)
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[3] == pytest.approx(3.0)


def test_dH_dt_is_not_the_second_derivative():
    """Guards the Dst-proxy mix-up: dH/dt and d2H/dt2 are different quantities.

    A single one-hour excursion changes ``dH/dt`` by 1 nT/min but ``d2H/dt2``
    by 0; conflating them would mislabel storm curvature as a rate.
    """
    df = pd.DataFrame({"X": [0.0, 1.0, 2.0], "Y": [0.0] * 3, "Z": [0.0] * 3})
    first = gm.calculate_dH_dt(df)
    second = gm.calculate_dst_proxy(df)
    assert first.iloc[1] == pytest.approx(1.0)
    assert second.iloc[2] == pytest.approx(0.0)


def test_dst_proxy_needs_three_samples_and_returns_curvature():
    df = pd.DataFrame({"X": [1.0, 2.0, 4.0, 8.0], "Y": [0.0] * 4, "Z": [0.0] * 4})
    result = gm.calculate_dst_proxy(df)
    assert not gm.is_error(result)
    assert np.isnan(result.iloc[0]) and np.isnan(result.iloc[1])
    # H = 1,2,4,8 -> first differences 1,2,4 -> second differences 1,2
    assert result.iloc[2] == pytest.approx(1.0)
    assert result.iloc[3] == pytest.approx(2.0)


def test_dH_dt_rejects_a_non_positive_sample_rate():
    df = pd.DataFrame({"X": [1.0, 2.0], "Y": [0.0, 0.0], "Z": [0.0, 0.0]})
    assert gm.is_error(gm.calculate_dH_dt(df, samples_per_minute=0))
    assert gm.is_error(gm.calculate_dH_dt(df, samples_per_minute=-1))


def test_dH_dt_preserves_nan_gaps(day):
    result = gm.calculate_dH_dt(day)
    assert np.isnan(result.iloc[17]) or np.isnan(result.iloc[18])


# --------------------------------------------------------------------------- #
# 4. nighttime baseline
# --------------------------------------------------------------------------- #
def test_night_baseline_averages_only_the_window():
    ts = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "X": np.arange(24.0), "Y": np.zeros(24), "Z": np.zeros(24)})
    result = gm.get_nighttime_baseline(df, "H", night_hours=(0, 4))
    assert not gm.is_error(result)
    # hours 0,1,2,3 -> H = 0,1,2,3 -> mean 1.5
    assert result["baseline"] == pytest.approx(1.5)
    assert result["n_samples"] == 4


def test_night_window_wraps_past_midnight():
    ts = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "X": np.arange(24.0), "Y": np.zeros(24), "Z": np.zeros(24)})
    result = gm.get_nighttime_baseline(df, "H", night_hours=(22, 6))
    # hours 22,23,0,1,2,3,4,5 -> H = 22,23,0,1,2,3,4,5
    assert result["n_samples"] == 8


def test_baseline_reports_none_for_an_empty_window():
    ts = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "X": np.arange(24.0), "Y": np.zeros(24), "Z": np.zeros(24)})
    result = gm.get_nighttime_baseline(df, "H", night_hours=(12, 13))
    assert result["baseline"] == pytest.approx(12.0)
    # and a window with no data at all must not invent a zero
    empty = gm.get_nighttime_baseline(df.head(1), "H", night_hours=(5, 6))
    assert empty["baseline"] is None


def test_baseline_needs_a_timestamp_column():
    df = pd.DataFrame({"X": [1.0], "Y": [1.0], "Z": [1.0]})
    result = gm.get_nighttime_baseline(df, "H")
    assert gm.is_error(result) and result["error"] == "missing_timestamp"


def test_baseline_rejects_impossible_hours(day):
    assert gm.is_error(gm.get_nighttime_baseline(day, "H", night_hours=(0, 99)))


# --------------------------------------------------------------------------- #
# 5. DSL -- correctness
# --------------------------------------------------------------------------- #
def test_reference_formula_matches_numpy(simple):
    result = gm.evaluate_formula(simple, "sqrt(X**2 + Y**2) + Z/10")
    assert not gm.is_error(result)
    expected = np.sqrt(simple["X"] ** 2 + simple["Y"] ** 2) + simple["Z"] / 10.0
    np.testing.assert_allclose(result.to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_formula_can_use_derived_columns(day):
    """H and F are not in a raw frame but must be usable inside a formula."""
    result = gm.evaluate_formula(day, "F - H")
    assert not gm.is_error(result)
    assert (result.dropna() >= 0).all()


def test_formula_atan2_gives_inclination_in_radians(simple):
    result = gm.evaluate_formula(simple, "atan2(Z, sqrt(X**2 + Y**2))")
    assert not gm.is_error(result)
    expected = np.arctan2(simple["Z"], np.hypot(simple["X"], simple["Y"]))
    np.testing.assert_allclose(result.to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_formula_degrees_matches_the_I_column(day):
    """The DSL must reproduce the module's own inclination exactly."""
    result = gm.evaluate_formula(day, "degrees(atan2(Z, H))")
    assert not gm.is_error(result)
    enriched = gm.add_derived_components(day)
    valid = enriched["I"].notna()
    np.testing.assert_allclose(
        result.to_numpy()[valid.to_numpy()],
        enriched["I"].to_numpy()[valid.to_numpy()],
        rtol=1e-12,
        atol=1e-9,
    )


def test_formula_constants_are_available(simple):
    result = gm.evaluate_formula(simple, "pi")
    assert not gm.is_error(result)
    assert result.iloc[0] == pytest.approx(np.pi)


def test_scalar_formula_broadcasts(simple):
    result = gm.evaluate_formula(simple, "42")
    assert not gm.is_error(result)
    assert len(result) == len(simple)
    assert (result == 42).all()


def test_unary_minus_works(simple):
    result = gm.evaluate_formula(simple, "-X")
    assert not gm.is_error(result)
    np.testing.assert_allclose(result.to_numpy(), -simple["X"].to_numpy())


def test_operator_precedence_is_pythonic(simple):
    result = gm.evaluate_formula(simple, "2 + 3 * 4")
    assert result.iloc[0] == pytest.approx(14.0)


def test_division_by_zero_is_reported_not_inf():
    """A well-formed formula on degenerate data must be reported, not returned.

    numpy signals division by zero with a warning and an ``inf`` result rather
    than raising, so without an explicit check ``1/0`` would hand ``inf`` to an
    LLM -- not valid strict JSON -- and stretch a chart axis to infinity.
    """
    df = pd.DataFrame({"X": [1.0, 0.0], "Y": [0.0, 0.0], "Z": [0.0, 0.0]})
    result = gm.evaluate_formula(df, "1/0")
    assert gm.is_error(result)
    assert result["error"] == "formula_eval_error"


def test_column_divided_by_zero_is_reported(day):
    """Partial degeneracy is not an error: only a wholly non-finite result is."""
    df = pd.DataFrame({"X": [0.0, 2.0], "Y": [0.0, 0.0], "Z": [0.0, 0.0]})
    result = gm.evaluate_formula(df, "1/X")
    assert not gm.is_error(result), "one bad row must not fail the whole formula"
    assert np.isinf(result.iloc[0])


def test_wholly_overflowing_formula_is_reported():
    """Every row overflowing to inf must be an error.

    The check is "no finite value at all", so a *partially* overflowing formula
    stays usable -- see the sibling test, where one bad row among good ones is
    returned rather than discarding real data.
    """
    df = pd.DataFrame({"X": [2.0, 3.0], "Y": [0.0, 0.0], "Z": [0.0, 0.0]})
    assert gm.is_error(gm.evaluate_formula(df, "X**9999"))


def test_formula_result_is_json_clean(simple):
    """Results travel to an LLM, so NaN must become None at the boundary."""
    df = pd.DataFrame({"X": [1.0, np.nan], "Y": [0.0, 0.0], "Z": [0.0, 0.0]})
    result = gm.evaluate_formula(df, "X")
    assert not gm.is_error(result)
    assert np.isnan(result.iloc[1])  # NaN is preserved in the Series...
    # ...but _jsonable, the LLM-facing conversion, must yield None.
    assert gm._jsonable(result.iloc[1]) is None


# --------------------------------------------------------------------------- #
# 6. DSL -- the required injection contour
# --------------------------------------------------------------------------- #
def test_required_injection_attempt_is_refused(simple):
    """The exact payload from the Phase 1 brief must fail to parse.

    Asserted on the error *code* and on the payload, because a generic exception
    would also "not execute" the payload while still meaning the grammar is too
    permissive somewhere else.
    """
    payload = "__import__('os').system('rm -rf /')"
    result = gm.evaluate_formula(simple, payload)
    assert gm.is_error(result), "the injection payload must be refused"
    assert result["error"] == "formula_parse_error"
    assert result["formula"] == payload


@pytest.mark.parametrize(
    "payload",
    [
        "__import__('os').system('rm -rf /')",
        "__import__(\"os\")",
        "open('/etc/passwd').read()",
        "eval('1+1')",
        "exec('import os')",
        "globals()",
        "locals()",
        "vars()",
        "getattr(__builtins__, 'x')",
        "().__class__.__bases__",
        "X.__class__",
        "X.real",
        "X.__dict__",
        "[x for x in range(10)]",
        "{k: 1 for k in range(3)}",
        "(lambda: 1)()",
        "X[0]",
        "X['__class__']",
        "os.system('ls')",
        "input('x')",
        "compile('1','','eval')",
        "breakpoint()",
        "exit()",
        "print('x')",
        "dir()",
        "id(X)",
        "help",
        "X if X else 0",
        "X and Y",
        "not X",
        "X == Y",
        "X < Y",
        "[1,2,3]",
        "{'a':1}",
        "(1,2)",
        "f'{X}'",
        "X.bit_length()",
        "abs.__call__(X)",
        "sqrt.__globals__",
        "*X",
        "yield",
        "await X",
        "X := 5",
        "assert False",
        "del X",
        "raise Exception()",
        "import os",
        "while True: pass",
        "X.__init__.__globals__['os'].system('ls')",
    ],
)
def test_injection_payloads_are_all_refused(simple, payload):
    """Nothing outside the arithmetic grammar may parse.

    ``eval``/``exec`` appear in the list on purpose: their mere presence as
    strings must not be mistaken for their being callable here.
    """
    result = gm.evaluate_formula(simple, payload)
    assert gm.is_error(result), f"payload was accepted: {payload!r}"
    assert result["error"] == "formula_parse_error", payload


def test_dunder_names_are_refused(simple):
    for name in ("__builtins__", "__import__", "__class__", "__globals__"):
        assert gm.is_error(gm.evaluate_formula(simple, name)), name


def test_only_whitelisted_functions_are_callable(simple):
    """Any callable not in the table is refused by name, not by accident."""
    for name in ("open", "eval", "exec", "compile", "getattr", "setattr", "sum", "len"):
        assert gm.is_error(gm.evaluate_formula(simple, f"{name}(X)")), name


def test_every_whitelisted_function_actually_works(simple):
    """Guards against a typo leaving a function in the table that never fires."""
    one_arg = [
        "sqrt(abs(X))",
        "sin(X/1000)",
        "cos(X/1000)",
        "tan(X/1000)",
        "atan(X/1000)",
        "asin(X/1000)",
        "acos(X/1000)",
        "abs(X)",
        "exp(X/1000)",
        "log(abs(X)+1)",
        "log10(abs(X)+1)",
        "degrees(X)",
        "radians(X)",
        "sign(X)",
    ]
    two_arg = ["atan2(Z, X)", "hypot(X, Y)", "power(abs(X)+1, 2)", "minimum(X, Y)", "maximum(X, Y)"]
    for formula in one_arg + two_arg:
        result = gm.evaluate_formula(simple, formula)
        assert not gm.is_error(result), f"{formula} -> {result}"


def test_wrong_arity_is_refused(simple):
    """Arity is checked, so sqrt(X, Y) cannot smuggle a second argument."""
    result = gm.evaluate_formula(simple, "sqrt(X, Y)")
    assert gm.is_error(result)
    assert "argument" in result["message"]


def test_string_literals_are_refused(simple):
    """A string literal has no place in a numeric grammar and is refused."""
    assert gm.is_error(gm.evaluate_formula(simple, "'abc'"))


def test_boolean_literals_are_refused(simple):
    """bool subclasses int, so True must not silently become 1."""
    assert gm.is_error(gm.evaluate_formula(simple, "True"))


def test_oversized_formula_is_refused(simple):
    assert gm.is_error(gm.evaluate_formula(simple, "1+" * 500 + "1"))


def test_empty_formula_is_refused(simple):
    assert gm.is_error(gm.evaluate_formula(simple, "   "))


def test_unknown_column_lists_the_valid_ones(simple):
    result = gm.evaluate_formula(simple, "Q * 2")
    assert gm.is_error(result)
    assert "X" in (result.get("available") or [])


def test_error_payload_lists_the_grammar(simple):
    """A refused formula must teach the model how to write a valid one."""
    result = gm.evaluate_formula(simple, "open('x')")
    assert result.get("available"), "the error should list callable functions"


# --------------------------------------------------------------------------- #
# 7. the module must contain no eval or exec
# --------------------------------------------------------------------------- #
def test_source_contains_no_eval_or_exec():
    """Belt and braces on the security claim: read the source, not the docstring.

    The layered argument is the real defence; this guards against a future
    contributor reaching for ``eval`` to implement one more function.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(gm))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in {"eval", "exec", "compile"}:
            pytest.fail(f"{inspect.getsourcefile(gm)} uses {node.id}()")
        if isinstance(node, ast.Attribute) and node.attr in {"system", "popen"}:
            pytest.fail(f"{inspect.getsourcefile(gm)} reaches {node.attr}")
    assert "ast.parse" in inspect.getsource(gm)


def test_dsl_tables_are_the_only_reachable_callables():
    """The dispatcher is a dict literal, so its keys are the whole attack surface."""
    import ast
    import inspect

    # The docstring mentions getattr to explain that it is NOT used, so the
    # assertion runs against the executable body only.
    tree = ast.parse(inspect.getsource(gm._eval_node))
    executable = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Call, ast.Attribute, ast.Import, ast.ImportFrom))
    ]
    for node in executable:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "getattr", "the DSL must not use getattr"
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"getattr", "__import__", "system"}
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            pytest.fail("the DSL must not import anything")

    assert set(gm.formula_functions()) == set(gm.DSL_FUNCTIONS)