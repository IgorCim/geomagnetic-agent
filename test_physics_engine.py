import pandas as pd
import numpy as np

import physics_engine as pe
from test_utils import create_mock_solar_wind_frame, make_solar_wind_frame


def test_extract_sq_baseline_night_median():
    idx = pd.date_range('2024-09-10 00:30:00', periods=6, freq='min', tz='UTC')
    df = pd.DataFrame({'timestamp': idx, 'X': [100, 102, 98, 101, 99, 103], 'Y': [0]*6, 'Z': [-300]*6})
    res = pe.extract_sq_baseline(df)
    assert not pe.is_error(res)
    assert 'X_sq' in res.columns
    assert 'H_sq' in res.columns
    assert abs(res['X_sq'].iloc[0] - 100.5) < 1e-9

def test_calculate_deltas_has_dh_fdelta():
    idx = pd.date_range('2024-09-10 00:00:00', periods=3, freq='min', tz='UTC')
    df = pd.DataFrame({'timestamp': idx, 'X': [100, 101, 100], 'Y': [0, 1, 0], 'Z': [0, 0, 0]})
    df['X_sq'] = 100; df['Y_sq'] = 0; df['Z_sq'] = 0
    res = pe.calculate_deltas(df)
    assert not pe.is_error(res)
    assert 'dH' in res.columns
    assert 'F_delta' in res.columns

def test_calculate_gic_proxy_has_dhdt():
    idx = pd.date_range('2024-09-10 00:00:00', periods=4, freq='min', tz='UTC')
    df = pd.DataFrame({'timestamp': idx, 'H': [100, 120, 110, 115]})
    res = pe.calculate_gic_proxy(df)
    assert not pe.is_error(res)
    assert 'dH_dt' in res.columns

def test_calculate_dst_proxy_basic():
    idx = pd.date_range('2024-09-10 00:00:00', periods=3, freq='min', tz='UTC')
    df = pd.DataFrame({'timestamp': idx, 'H': [100, 110, 105], 'H_sq': [105, 105, 105]})
    res = pe.calculate_dst_proxy(df, latitude=0)
    assert not pe.is_error(res)
    assert 'dst_proxy' in res.columns

def test_calculate_coupling_function_needs_solar_wind():
    idx = pd.date_range('2024-09-10 00:00:00', periods=2, freq='min', tz='UTC')
    df = pd.DataFrame({'timestamp': idx, 'H': [100, 101]})
    res = pe.calculate_coupling_function(df)
    assert pe.is_error(res)
    assert res.get('error') == 'missing_solar_wind'

def test_calculate_coupling_function_computes_ey():
    idx = pd.date_range('2024-09-10 00:00:00', periods=2, freq='min', tz='UTC')
    df = pd.DataFrame({'timestamp': idx, 'H': [100, 101]})
    sw = pe.align_solar_wind(
        df, make_solar_wind_frame(V=400.0, Bz=-5.0, periods=2, start='2024-09-10 00:00:00')
    )
    assert not pe.is_error(sw)
    res = pe.calculate_coupling_function(df, sw)
    assert not pe.is_error(res)
    assert 'E_y' in res.columns

def test_detect_ssc_finds_jump():
    idx = pd.date_range('2024-09-10 00:00:00', periods=5, freq='min', tz='UTC')
    df = pd.DataFrame({'timestamp': idx, 'H': [100, 100, 150, 151, 151]})
    res = pe.detect_ssc(df, threshold_nT_per_min=40)
    assert not pe.is_error(res)
    assert 'ssc_flag' in res.columns
    assert res['ssc_flag'].tolist()[2] == 1


def test_detect_ssc_clusters_one_step_into_one_event():
    """A one-minute step spans two samples; it is still one SSC, not two."""
    idx = pd.date_range('2024-09-10 00:00:00', periods=6, freq='min', tz='UTC')
    df = pd.DataFrame({'timestamp': idx, 'H': [100, 100, 140, 160, 160, 160]})
    res = pe.detect_ssc(df, threshold_nT_per_min=5)
    assert res.attrs['ssc_count'] == 1
    first = res.attrs['ssc_events'][0]
    assert first['dH_dt_nT_per_min'] == 40.0
    assert first['index'] == 2


def test_detect_ssc_two_steps_an_hour_apart_are_two_events():
    idx = pd.date_range('2024-09-10 00:00:00', periods=125, freq='min', tz='UTC')
    h = [100.0] * 125
    for i in range(2, 125):
        h[i] += 40.0
    for i in range(62, 125):
        h[i] += 40.0
    res = pe.detect_ssc(pd.DataFrame({'timestamp': idx, 'H': h}), threshold_nT_per_min=5)
    assert res.attrs['ssc_count'] == 2


def test_detect_ssc_normalises_the_rate_to_minutes():
    """+10 nT per 5-minute sample is 2 nT/min, not a 10 nT/min onset."""
    idx = pd.date_range('2024-09-10 00:00:00', periods=5, freq='5min', tz='UTC')
    df = pd.DataFrame({'timestamp': idx, 'H': [100, 100, 110, 120, 130]})
    res = pe.detect_ssc(df, threshold_nT_per_min=5)
    assert res.attrs['ssc_count'] == 0


def test_detect_ssc_reports_the_true_count_when_the_payload_is_capped():
    periods = 60 * 12 + 3
    h = [100.0] * periods
    for step in range(13):
        for i in range(2 + step * 60, periods):
            h[i] += 40.0
    idx = pd.date_range('2024-09-10 00:00:00', periods=periods, freq='min', tz='UTC')
    res = pe.detect_ssc(pd.DataFrame({'timestamp': idx, 'H': h}), threshold_nT_per_min=5)
    assert res.attrs['ssc_count'] == 13
    assert len(res.attrs['ssc_events']) == 10
    assert res.attrs['ssc_events_truncated'] is True


# --- Phase 6.2: solar-wind alignment and the vector coupling function -------

def _observatory(day='2024-09-12 00:00:00', periods=6):
    idx = pd.date_range(day, periods=periods, freq='min', tz='UTC')
    return pd.DataFrame({
        'timestamp': idx,
        'X': [100.0] * periods,
        'Y': [0.0] * periods,
        'Z': [-300.0] * periods,
    })


def _solar_wind(periods=6, v=None, bz=None):
    v = v if v is not None else [100.0 + 10.0 * i for i in range(periods)]
    bz = bz if bz is not None else [-1.0 - i for i in range(periods)]
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-09-12 00:00:00', periods=periods, freq='min', tz='UTC'),
        'V': v,
        'Bz': bz,
    })


def test_align_solar_wind_bridges_short_gaps_only():
    """A one-minute dropout is interpolated; a long outage stays NaN."""
    obs = _observatory()
    short = _solar_wind(v=[100.0, 110.0, float('nan'), 130.0, 140.0, 150.0])
    res = pe.align_solar_wind(obs, short)
    assert not pe.is_error(res)
    assert {'V', 'Bz', 'Bz_gsm'} <= set(res.columns)
    assert res['V'].notna().all()
    assert abs(res['V'].iloc[2] - 120.0) < 1e-9  # halfway between 110 and 130

    long = _solar_wind(v=[100.0, float('nan'), float('nan'), float('nan'), float('nan'), 150.0])
    res_long = pe.align_solar_wind(obs, long, limit=1)
    assert not pe.is_error(res_long)
    # only one sample either side of a 4-minute outage may be invented
    assert res_long['V'].isna().sum() >= 3


def test_align_solar_wind_applies_a_propagation_lag():
    """A positive lag moves L1 samples later, onto the station's clock."""
    obs = _observatory(periods=5)
    res = pe.align_solar_wind(obs, _solar_wind(periods=5), lag_minutes=2)
    assert not pe.is_error(res)
    # sw(00:00)=100 now lands at 00:02; the leading edge is outside coverage
    assert abs(res['V'].iloc[2] - 100.0) < 1e-9
    assert pd.isna(res['V'].iloc[0])


def test_align_solar_wind_requires_speed_and_bz():
    obs = _observatory()
    sw = pd.DataFrame({
        'timestamp': pd.date_range('2024-09-12 00:00:00', periods=6, freq='min', tz='UTC'),
        'density': [5.0] * 6,
    })
    res = pe.align_solar_wind(obs, sw)
    assert pe.is_error(res)
    assert res.get('error') == 'missing_columns'


def test_calculate_coupling_function_vectorizes_ey():
    """The aligned frame's V/Bz columns give per-row Ey and a peak summary."""
    obs = _observatory(periods=5)
    sw = make_solar_wind_frame(periods=5)
    sw['V'] = [400.0, 410.0, 420.0, 430.0, 440.0]
    sw['Bz'] = [-1.0, -2.0, -3.0, -4.0, -5.0]
    aligned = pe.align_solar_wind(obs, sw)
    assert not pe.is_error(aligned)
    res = pe.calculate_coupling_function(obs, aligned)
    assert not pe.is_error(res)
    np.testing.assert_allclose(
        res['E_y'].to_numpy(),
        (aligned['V'] * aligned['Bz']).to_numpy(),
    )
    info = res.attrs['ey_coupling']
    assert info['mode'] == 'vector'
    assert info['n_samples'] == 5
    assert abs(info['E_y_mV_m_peak_abs'] - 2200.0) < 1e-9
    assert abs(info['E_y_mV_m_min'] + 2200.0) < 1e-9


def test_calculate_coupling_function_requires_an_aligned_frame():
    """A raw (unaligned) solar-wind frame is refused, not silently reused."""
    obs = _observatory(periods=5)
    res = pe.calculate_coupling_function(obs, _solar_wind(periods=2))
    assert pe.is_error(res)
    assert res.get('error') == 'length_mismatch'


def test_calculate_coupling_function_constant_frame_records_scalar_summary():
    obs = _observatory(periods=2)
    aligned = pe.align_solar_wind(obs, make_solar_wind_frame(V=400.0, Bz=-5.0, periods=2))
    assert not pe.is_error(aligned)
    res = pe.calculate_coupling_function(obs, aligned)
    assert not pe.is_error(res)
    info = res.attrs['ey_coupling']
    assert info['mode'] == 'scalar'
    assert abs(info['E_y_mV_m'] + 2000.0) < 1e-9


# --- Phase 5.2: five more physics metrics -----------------------------------

def test_burton_coupling_needs_solar_wind():
    res = pe.calculate_burton_coupling(_observatory(periods=2))
    assert pe.is_error(res)
    assert res.get('error') == 'missing_solar_wind'


def test_burton_coupling_reports_coupling_and_predicted_dst():
    obs = _observatory(periods=2)
    sw = make_solar_wind_frame(V=450.0, Bz=-10.0, Np=5.0, periods=2)
    indices = pd.DataFrame({
        'timestamp': obs['timestamp'],
        'Dst': [-30.0, -30.0],
    })
    res = pe.calculate_burton_coupling(obs, sw, indices)
    assert not pe.is_error(res)
    assert 'burton_coupling' in res.columns
    info = res.attrs['burton']
    # Bs = -Bz = 10; coupling = 0.005 * sqrt(5) * 450 * 10
    expected_coupling = 0.005 * (5.0 ** 0.5) * 450.0 * 10.0
    assert abs(info['coupling_function'] - expected_coupling) < 1e-3
    # dDst/dt = -coupling - Dst/tau ; predicted = Dst + dDst/dt (1-hour Euler step)
    expected_ddst = -expected_coupling - (-30.0) / 8.0
    assert abs(info['dDst_dt_nT_per_h'] - expected_ddst) < 1e-3
    assert info['dDst_dt_nT_per_h'] < 0          # southward IMF deepens Dst
    assert abs(info['predicted_dst'] - (-30.0 + expected_ddst)) < 1e-2
    assert info['predicted_dst'] < -30.0


def test_burton_quiet_field_recovers_toward_zero():
    obs = _observatory(periods=2)
    sw = make_solar_wind_frame(V=450.0, Bz=5.0, Np=5.0, periods=2)
    indices = pd.DataFrame({
        'timestamp': obs['timestamp'],
        'Dst': [-30.0, -30.0],
    })
    res = pe.calculate_burton_coupling(obs, sw, indices)
    info = res.attrs['burton']
    assert info['coupling_function'] == 0.0
    assert info['dDst_dt_nT_per_h'] > 0          # decay drives Dst back to zero
    assert abs(info['predicted_dst'] - (-26.25)) < 1e-2


def test_burton_coupling_without_indices_starts_from_quiet_dst():
    """No Dst frame: the forecast starts at 0 with the injection term only."""
    obs = _observatory(periods=2)
    res = pe.calculate_burton_coupling(obs, make_solar_wind_frame(Bz=-10.0, periods=2))
    assert not pe.is_error(res)
    info = res.attrs['burton']
    assert info['Dst_input'] == 0.0
    assert info['coupling_function'] > 0


def test_burton_coupling_northward_bz_gives_zero_injection():
    obs = _observatory(periods=2)
    res = pe.calculate_burton_coupling(obs, make_solar_wind_frame(Bz=5.0, periods=2))
    assert not pe.is_error(res)
    info = res.attrs['burton']
    assert info['Bs'] == 0.0
    assert info['coupling_function'] == 0.0


def test_burton_coupling_missing_column_is_an_error():
    obs = _observatory(periods=2)
    sw = make_solar_wind_frame(periods=2).drop(columns=['Np'])
    res = pe.calculate_burton_coupling(obs, sw)
    assert pe.is_error(res)
    assert res.get('error') == 'invalid_solar_wind'
    assert 'Np' in res['missing']


def test_akasofu_epsilon_needs_solar_wind():
    res = pe.calculate_akasofu_epsilon(_observatory(periods=2))
    assert pe.is_error(res)
    assert res.get('error') == 'missing_solar_wind'


def test_akasofu_epsilon_reports_both_units():
    obs = _observatory(periods=2)
    sw = make_solar_wind_frame(V=450.0, Bx=2.0, By=-3.0, Bz=-5.0, periods=2)
    res = pe.calculate_akasofu_epsilon(obs, sw)
    assert not pe.is_error(res)
    assert 'akasofu_epsilon' in res.columns
    info = res.attrs['akasofu']
    # The SI value is V * B^2 * l0^2 * sin^4(theta/2) / mu0, i.e. watts.
    b_total_nt = (2.0**2 + 3.0**2 + 5.0**2) ** 0.5
    theta = np.arctan2(-3.0, -5.0)
    expected = (
        450e3
        * (b_total_nt * 1e-9) ** 2
        * (7 * 6.371e6) ** 2
        * np.sin(theta / 2.0) ** 4
        / pe.MU_0_H_PER_M
    )
    assert abs(info['epsilon_W'] - expected) / expected < 1e-6
    assert 1e8 < info['epsilon_W'] < 1e12     # a storm-time power, not 1e5 W
    # W -> erg/s is a fixed 1e7 factor; attrs are rounded, so compare loosely
    assert abs(info['epsilon_erg_s'] - info['epsilon_W'] * 1e7) < 1e-2 * info['epsilon_erg_s']


def test_akasofu_epsilon_northward_field_has_small_orientation_factor():
    obs = _observatory(periods=2)
    res = pe.calculate_akasofu_epsilon(
        obs, make_solar_wind_frame(V=450, Bx=0, By=0, Bz=5, periods=2)
    )
    assert not pe.is_error(res)
    # theta = arctan(0/5) = 0 -> sin^4(0) = 0
    assert res.attrs['akasofu']['epsilon_W'] == 0.0


def test_peak_dhdt_grades_the_peak():
    idx = pd.date_range('2024-09-10 00:00:00', periods=6, freq='min', tz='UTC')
    # a 300 nT/min jump -> high
    df = pd.DataFrame({'timestamp': idx, 'H': [100, 100, 400, 400, 400, 400]})
    res = pe.calculate_peak_dhdt(df)
    assert not pe.is_error(res)
    assert 'peak_dhdt' in res.columns
    info = res.attrs['peak_dhdt']
    assert info['warning_level'] == 'high'
    assert info['peak_dhdt_nT_min'] >= 200
    assert info['peak_time'] is not None


def test_peak_dhdt_low_and_medium_levels():
    idx = pd.date_range('2024-09-10 00:00:00', periods=4, freq='min', tz='UTC')
    low = pe.calculate_peak_dhdt(pd.DataFrame({'timestamp': idx, 'H': [100, 110, 110, 110]}))
    assert low.attrs['peak_dhdt']['warning_level'] == 'low'
    medium = pe.calculate_peak_dhdt(pd.DataFrame({'timestamp': idx, 'H': [100, 180, 180, 180]}))
    assert medium.attrs['peak_dhdt']['warning_level'] == 'medium'


def test_peak_dhdt_window_must_be_positive():
    res = pe.calculate_peak_dhdt(_observatory(periods=3), window_seconds=0)
    assert pe.is_error(res)
    assert res.get('error') == 'invalid_input'


def test_substorm_onset_needs_sml_column():
    res = pe.detect_substorm_onset(_observatory(periods=3))
    assert pe.is_error(res)
    assert res.get('error') == 'missing_columns'


def test_substorm_onset_finds_sustained_drop():
    idx = pd.date_range('2024-09-10 00:00:00', periods=8, freq='min', tz='UTC')
    # onset: SML falls -60 nT/min for 3 consecutive minutes starting at index 2
    sml = [-100, -110, -170, -230, -290, -290, -290, -290]
    df = pd.DataFrame({'timestamp': idx, 'SML': sml})
    res = pe.detect_substorm_onset(df)
    assert not pe.is_error(res)
    info = res.attrs['substorm']
    assert info['onset_time'] is not None
    assert info['sml_drop'] > 0
    assert info['confidence'] in ('low', 'medium', 'high')


def test_substorm_onset_quiet_period_has_no_onset():
    idx = pd.date_range('2024-09-10 00:00:00', periods=6, freq='min', tz='UTC')
    df = pd.DataFrame({'timestamp': idx, 'SML': [-50, -51, -50, -52, -51, -50]})
    res = pe.detect_substorm_onset(df)
    assert not pe.is_error(res)
    assert res.attrs['substorm']['onset_time'] is None
    assert res.attrs['substorm']['confidence'] is None


def test_eej_index_needs_reference():
    res = pe.calculate_eej_index(_observatory(periods=3))
    assert pe.is_error(res)
    assert res.get('error') == 'missing_reference'


def test_eej_index_subtracts_the_reference_h():
    idx = pd.date_range('2024-09-10 00:00:00', periods=4, freq='min', tz='UTC')
    eq = pd.DataFrame({'timestamp': idx, 'H': [100.0, 120.0, 150.0, 130.0]})
    ref = pd.DataFrame({'timestamp': idx, 'H': [100.0, 105.0, 110.0, 110.0]})
    res = pe.calculate_eej_index(eq, ref)
    assert not pe.is_error(res)
    np.testing.assert_allclose(res['eej_index'].to_numpy(), [0.0, 15.0, 40.0, 20.0])
    info = res.attrs['eej']
    assert abs(info['eej_nT'] - 40.0) < 1e-9
    assert info['peak_time'] is not None


def test_eej_index_length_mismatch_is_an_error():
    idx = pd.date_range('2024-09-10 00:00:00', periods=4, freq='min', tz='UTC')
    eq = pd.DataFrame({'timestamp': idx, 'H': [100.0] * 4})
    res = pe.calculate_eej_index(eq, [1.0, 2.0])
    assert pe.is_error(res)
    assert res.get('error') == 'length_mismatch'
