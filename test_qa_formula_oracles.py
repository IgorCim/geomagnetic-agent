"""Independent numerical control points for every registered DSL function.

These controls do not certify physics models or replace live tool-call tests.
"""
import numpy as np
import pandas as pd
import pytest
import geomag_math as gm


CONTROLS = [
    ('abs(-3)', 3.), ('acos(1)', 0.), ('asin(0)', 0.), ('atan(0)', 0.),
    ('atan2(0,1)', 0.), ('cos(0)', 1.), ('degrees(pi)', 180.), ('exp(0)', 1.),
    ('hypot(3,4)', 5.), ('log(e)', 1.), ('log10(100)', 2.), ('maximum(2,3)', 3.),
    ('minimum(2,3)', 2.), ('power(2,3)', 8.), ('radians(180)', np.pi),
    ('sign(-3)', -1.), ('sin(pi/2)', 1.), ('sqrt(9)', 3.), ('tan(0)', 0.),
]


@pytest.mark.parametrize('expression,expected', CONTROLS)
def test_dsl_control_point_on_vector_with_gap(expression, expected):
    frame = pd.DataFrame({'X': [2., np.nan, -8.]})
    actual = gm.evaluate_formula(frame, f'{expression}+0*X')
    assert isinstance(actual, pd.Series), actual
    np.testing.assert_allclose(actual.to_numpy(), [expected, np.nan, expected], rtol=1e-12, atol=1e-12, equal_nan=True)
    assert frame['X'].iloc[0] == 2. and pd.isna(frame['X'].iloc[1])


def test_control_catalog_names_cover_the_actual_dsl_registry():
    assert {e.split('(')[0] for e, _ in CONTROLS} == set(gm.formula_functions())


def test_cartesian_reference_vectors_and_declination_quadrants():
    frame = pd.DataFrame({'X': [3., -3., -3., 3.], 'Y': [4., 4., -4., -4.], 'Z': [12., 12., -12., -12.]})
    np.testing.assert_allclose(gm.horizontal_field(frame), [5.] * 4, atol=1e-12)
    np.testing.assert_allclose(gm.total_field(frame), [13.] * 4, atol=1e-12)
    np.testing.assert_allclose(gm.declination(frame), [53.13010235415598, 126.86989764584402, -126.86989764584402, -53.13010235415598], atol=1e-10)
