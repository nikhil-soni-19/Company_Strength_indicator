"""Smoke tests for the Step-2 metric calculations."""

from __future__ import annotations

from src.metrics import (
    compute_adv_dollar,
    compute_amihud,
    compute_volume_cv,
    ema,
    sma,
)


def test_sma_and_ema_align_for_constant_series(synthetic_ohlcv):
    close = synthetic_ohlcv["Close"]
    assert sma(close, 30).iloc[-1] > 0
    assert ema(close, 30).iloc[-1] > 0


def test_adv_dollar_returns_both_windows(synthetic_ohlcv):
    result = compute_adv_dollar(synthetic_ohlcv)
    assert result.adv_dollar_30d is not None
    assert result.adv_dollar_90d is not None
    assert result.adv_dollar_30d > 0
    assert result.adv_dollar_90d > 0


def test_adv_dollar_clipping_is_applied(synthetic_ohlcv):
    df = synthetic_ohlcv.copy()
    df.loc[df.index[-2], "Volume"] *= 100  # huge spike
    result = compute_adv_dollar(df)
    raw_max = (df["Close"] * df["Volume"]).max()
    assert result.daily_dollar_volume_clipped.max() < raw_max


def test_volume_cv_is_low_for_steady_volume(synthetic_ohlcv):
    cv = compute_volume_cv(synthetic_ohlcv).volume_cv_30d
    assert cv is not None
    assert cv < 0.4, "Synthetic steady stock should be in the 'predictable' band"


def test_volume_cv_is_high_for_thin_float(thin_float_ohlcv):
    cv = compute_volume_cv(thin_float_ohlcv).volume_cv_30d
    assert cv is not None
    assert cv > 0.6, "Erratic stock should exceed the moderate-risk threshold"


def test_amihud_returns_a_finite_value(synthetic_ohlcv):
    a = compute_amihud(synthetic_ohlcv).amihud_30d
    assert a is not None
    assert a >= 0


def test_amihud_higher_for_thin_float(synthetic_ohlcv, thin_float_ohlcv):
    a_liquid = compute_amihud(synthetic_ohlcv).amihud_30d
    a_thin = compute_amihud(thin_float_ohlcv).amihud_30d
    assert a_liquid is not None and a_thin is not None
    assert a_thin > a_liquid
