import pandas as pd
import numpy as np

import physics_engine as pe


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

def test_calculate_coupling_function_needs_external():
    idx = pd.date_range('2024-09-10 00:00:00', periods=2, freq='min', tz='UTC')
    df = pd.DataFrame({'timestamp': idx, 'H': [100, 101]})
    res = pe.calculate_coupling_function(df)
    assert pe.is_error(res)
    assert res.get('error') == 'missing_external_params'

def test_calculate_coupling_function_computes_ey():
    idx = pd.date_range('2024-09-10 00:00:00', periods=2, freq='min', tz='UTC')
    df = pd.DataFrame({'timestamp': idx, 'H': [100, 101]})
    res = pe.calculate_coupling_function(df, {'V': 400, 'Bz': -5})
    assert not pe.is_error(res)
    assert 'E_y' in res.columns

def test_detect_ssc_finds_jump():
    idx = pd.date_range('2024-09-10 00:00:00', periods=5, freq='min', tz='UTC')
    df = pd.DataFrame({'timestamp': idx, 'H': [100, 100, 150, 151, 151]})
    res = pe.detect_ssc(df, threshold_nT_per_min=40)
    assert not pe.is_error(res)
    assert 'ssc_flag' in res.columns
    assert res['ssc_flag'].tolist()[2] == 1
