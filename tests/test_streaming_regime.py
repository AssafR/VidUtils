"""Tests for ``streaming_regime`` (online stats, jump ratio, regime break)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from streaming_regime import (
    EWMean,
    EWStd,
    FinalRegimeBreakDetector,
    MetricStep,
    RollingTopHalfMean,
    StreamingJumpRatioProcessor,
    notebook_aligned_jump_ratio_processor,
    run_jump_ratio_on_metric_rows,
    run_pipeline,
    run_processor_on_indexed_stream,
)


def _rolling_mean_top_half(arr: np.ndarray) -> float:
    v = arr[~np.isnan(arr)]
    if v.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(0.5 * v.size)))
    return float(np.mean(np.partition(v, -k)[-k:]))


def test_ew_mean_matches_pandas_ewm() -> None:
    rng = np.random.default_rng(0)
    values = rng.standard_normal(80)
    span = 17
    s = pd.Series(values)
    expected = s.ewm(span=span).mean()

    m = EWMean(span)
    for i, x in enumerate(values):
        got = m.update(float(x))
        assert got == pytest.approx(expected.iloc[i], rel=0, abs=1e-12)


def test_rolling_top_half_matches_partition_definition() -> None:
    window = 12
    rng = np.random.default_rng(1)
    values = rng.standard_normal(50)
    rt = RollingTopHalfMean(window)
    for i, x in enumerate(values):
        got = rt.update(float(x))
        if i < window - 1:
            assert got != got  # NaN
        else:
            chunk = values[i - window + 1 : i + 1]
            exp = _rolling_mean_top_half(np.asarray(chunk, dtype=np.float64))
            assert got == pytest.approx(exp, rel=0, abs=1e-10)


def test_jump_ratio_matches_pandas_series() -> None:
    rng = np.random.default_rng(2)
    values = np.abs(rng.standard_normal(60)) * 10 + 1.0
    span = 9
    window = 11
    epsilon = 1e-5

    s = pd.Series(values)
    ema = s.ewm(span=span, adjust=True).mean()
    rolling = s.rolling(window=window, min_periods=window).apply(
        _rolling_mean_top_half, raw=True
    )
    expected_jr = (epsilon + rolling) / (epsilon + ema)

    proc = StreamingJumpRatioProcessor(
        numerator=RollingTopHalfMean(window),
        baseline=EWMean(span),
        epsilon=epsilon,
    )
    steps, _ = run_pipeline(iter(values.tolist()), proc)
    for step in steps:
        i = step.index
        exp = expected_jr.iloc[i]
        if exp != exp:
            assert step.jump_ratio is None or step.jump_ratio != step.jump_ratio
        else:
            assert step.jump_ratio is not None
            assert step.jump_ratio == pytest.approx(float(exp), rel=0, abs=1e-9)


def test_run_processor_on_indexed_stream_preserves_frame_index() -> None:
    proc = StreamingJumpRatioProcessor(
        numerator=RollingTopHalfMean(3),
        baseline=EWMean(4),
    )
    pairs = [(100, 1.0), (250, 2.0), (999, 3.0)]
    steps, _ = run_processor_on_indexed_stream(iter(pairs), proc)
    assert [s.index for s in steps] == [100, 250, 999]
    assert all(s.x == x for s, (_, x) in zip(steps, pairs))


def test_final_regime_break_last_crossing() -> None:
    det = FinalRegimeBreakDetector(1.0, treat_initial_above_as_cross=True)
    # Above threshold throughout, then dip below, then final upward cross at index 5.
    ratios = [2.0, 2.0, 2.0, 2.0, 0.5, 2.0]
    for i, r in enumerate(ratios):
        det.consume(MetricStep(index=i, x=float(i), jump_ratio=r))
    assert det.result() is not None
    assert det.result().index == 5
    assert det.result().value == 2.0


def test_final_regime_break_no_cross_when_initial_high_and_strict() -> None:
    det = FinalRegimeBreakDetector(1.0, treat_initial_above_as_cross=False)
    det.consume(MetricStep(index=0, x=0.0, jump_ratio=2.0))
    assert det.result() is None


def test_final_regime_break_ignores_nan() -> None:
    det = FinalRegimeBreakDetector(1.0)
    det.consume(MetricStep(index=0, x=0.0, jump_ratio=float("nan")))
    det.consume(MetricStep(index=1, x=1.0, jump_ratio=2.0))
    assert det.result() is not None
    assert det.result().index == 1


def test_notebook_aligned_factory_and_metric_rows_roundtrip() -> None:
    """Metric rows as dicts (like CSV rows) produce the same jump_ratio as pandas on the column."""
    rng = np.random.default_rng(3)
    n = 40
    lap = np.abs(rng.standard_normal(n)) * 15 + 20.0
    rows = [{"frame": i, "laplacian_variance": float(lap[i])} for i in range(n)]

    span = 10
    win = 8
    eps = 1e-5
    s = pd.Series(lap)
    ema = s.ewm(span=span, adjust=True).mean()
    rolling = s.rolling(window=win, min_periods=win).apply(
        _rolling_mean_top_half, raw=True
    )
    expected = (eps + rolling) / (eps + ema)

    proc = notebook_aligned_jump_ratio_processor(
        ewm_span=span, rolling_window=win, epsilon=eps
    )
    steps, _ = run_jump_ratio_on_metric_rows(iter(rows), processor=proc)
    for step in steps:
        exp = expected.iloc[step.index]
        if exp != exp:
            assert step.jump_ratio is None or step.jump_ratio != step.jump_ratio
        else:
            assert step.jump_ratio == pytest.approx(float(exp), rel=0, abs=1e-9)


def test_ew_std_finite_after_one_sample() -> None:
    s = EWStd(5)
    assert s.update(3.0) == 0.0
