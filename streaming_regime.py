"""
Streaming jump-ratio style metrics and final regime-break detection.

Designed for experiments: swap online estimators (EWM vs ring buffer), preprocess
the raw stream, and plug different break rules without rewriting the pipeline.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol, TypeAlias, runtime_checkable
import numpy as np


# ---- Shared type aliases -------------------------------------------------

# One row of metrics or stats keyed by name.
MetricRow: TypeAlias = dict[str, Any]
MetricStatsRow: TypeAlias = dict[str, float | int]


def ewm_alpha_from_span(span: float) -> float:
    """Smoothing factor ``alpha = 2 / (span + 1)`` (pandas ``span`` convention)."""
    return 2.0 / (span + 1.0)


@runtime_checkable
class OnlineStatistic(Protocol):
    """Single scalar stream -> scalar estimate; NaN when not ready."""

    def reset(self) -> None: ...

    def update(self, x: float) -> float:
        """Ingest one sample; return current estimate (may be NaN)."""
        ...


class EWMean:
    """
    Exponentially weighted mean, O(1) memory.

    With default ``adjust=True``, matches pandas ``Series.ewm(span=span).mean()`` (same as
    ``adjust=True``). With ``adjust=False``, matches the simple recursive EWMA
    ``y_t = (1-α) y_{t-1} + α x_t``.
    """

    def __init__(self, span: float, *, adjust: bool = True) -> None:
        if span <= 0:
            raise ValueError("span must be positive")
        self._alpha = ewm_alpha_from_span(span)
        self._adjust = adjust
        self._mean: float | None = None
        self._adj_num: float | None = None
        self._adj_den: float | None = None

    def reset(self) -> None:
        self._mean = None
        self._adj_num = None
        self._adj_den = None

    @property
    def current(self) -> float | None:
        if self._adjust:
            if self._adj_num is None or self._adj_den is None:
                return None
            return self._adj_num / self._adj_den
        return self._mean

    def update(self, x: float) -> float:
        xf = float(x)
        a = self._alpha
        if self._adjust:
            if self._adj_num is None:
                self._adj_num = xf
                self._adj_den = 1.0
            else:
                self._adj_num = xf + (1.0 - a) * self._adj_num
                self._adj_den = 1.0 + (1.0 - a) * self._adj_den
            return self._adj_num / self._adj_den
        if self._mean is None:
            self._mean = xf
        else:
            self._mean = (1.0 - a) * self._mean + a * xf
        return self._mean


class EWStd:
    """
    Exponentially weighted std via E[x^2] and E[x] with the same alpha (biased, simple).
    O(1) memory. Returns NaN until at least one sample.
    """

    def __init__(self, span: float) -> None:
        if span <= 0:
            raise ValueError("span must be positive")
        self._alpha = ewm_alpha_from_span(span)
        self._mean = EWMean(span)
        self._mean_sq = EWMean(span)

    def reset(self) -> None:
        self._mean.reset()
        self._mean_sq.reset()

    def update(self, x: float) -> float:
        xf = float(x)
        self._mean.update(xf)
        self._mean_sq.update(xf * xf)
        mu = self._mean.current
        if mu is None:
            return float("nan")
        ms = self._mean_sq.current
        if ms is None:
            return float("nan")
        v = ms - mu * mu
        if v <= 0:
            return 0.0
        return v**0.5


class RollingTopHalfMean:
    """Exact mean of the largest ceil(n/2) values in the last ``window`` samples. O(window) memory."""

    def __init__(self, window: int, min_periods: int | None = None) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._window = window
        self._min = window if min_periods is None else min_periods
        self._buf: deque[float] = deque(maxlen=window)

    def reset(self) -> None:
        self._buf.clear()

    def update(self, x: float) -> float:
        self._buf.append(float(x))
        n = len(self._buf)
        if n < self._min:
            return float("nan")
        arr = np.asarray(self._buf, dtype=np.float64)
        k = max(1, int(np.ceil(0.5 * n)))
        return float(np.mean(np.partition(arr, -k)[-k:]))


@dataclass
class MetricStep:
    """
    Snapshot of all streaming jump-ratio state for a single time step.

    This is the "rich" record returned by jump-ratio oriented helpers
    (``StreamingJumpRatioProcessor``, ``run_pipeline``, etc.). For lighter
    usage (e.g. just wanting updated stats as dicts), prefer the functional
    iterators ``iter_online_stats`` and ``iter_jump_ratio_from_stats`` below.
    """

    #: Logical index of this sample (e.g. frame number from the metrics row).
    index: int
    #: Raw scalar value fed into the processor for this step.
    x: float
    #: Jump ratio at this step, when both numerator and baseline are finite.
    jump_ratio: float | None = None
    #: Current value of the "short-range" / numerator statistic.
    numerator: float | None = None
    #: Current value of the "long-range" / baseline statistic.
    baseline: float | None = None
    #: Free-form storage for additional per-step values when experimenting.
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MetricStepPipeline(Protocol):
    """
    Minimal interface for anything that turns a scalar stream into ``MetricStep``s.

    ``StreamingJumpRatioProcessor`` implements this protocol; you can provide
    your own pipelines as long as they expose the same surface.
    """

    def reset(self) -> None: ...

    def step(self, index: int, x: float) -> MetricStep: ...

    def process(self, stream: Iterator[float]) -> Iterator[MetricStep]: ...


class StreamingJumpRatioProcessor:
    """
    High-level helper that ties together two ``OnlineStatistic`` objects
    (numerator + baseline) into jump-ratio style ``MetricStep`` records.

    - For each incoming scalar ``x`` it optionally applies ``value_fn(x)``
      (e.g. to scale or transform).
    - It feeds that value into both the numerator and baseline statistics.
    - It then produces a ``MetricStep`` carrying the raw ``x``, the two
      updated statistics, and a ``jump_ratio`` field equal to
      ``(epsilon + numerator) / (epsilon + baseline)`` when both are finite.

    This class is convenient when you want an OO-style processor. If you
    prefer a more "pipe and filter" functional style, see:
    ``iter_online_stats`` + ``iter_jump_ratio_from_stats``.
    """

    def __init__(
        self,
        numerator: OnlineStatistic,
        baseline: OnlineStatistic,
        *,
        epsilon: float = 1e-5,
        value_fn: Callable[[float], float] | None = None,
    ) -> None:
        self._num = numerator
        self._base = baseline
        self._epsilon = epsilon
        self._value_fn = value_fn or (lambda v: float(v))

    def reset(self) -> None:
        self._num.reset()
        self._base.reset()

    def step(self, index: int, x: float) -> MetricStep:
        v = self._value_fn(x)
        n = self._num.update(v)
        b = self._base.update(v)
        step = MetricStep(index=index, x=x, numerator=n, baseline=b)
        if n == n and b == b:  # finite
            step.jump_ratio = (self._epsilon + n) / (self._epsilon + b)
        return step

    def process(self, stream: Iterator[float]) -> Iterator[MetricStep]:
        for i, x in enumerate(stream):
            yield self.step(i, x)


class FinalRegimeBreakDetector:
    """
    Tracks the *last* upward crossing of a threshold on a scalar series extracted from
    each step (default: ``jump_ratio``). "Upward" = current > threshold and previous <= threshold
    (missing previous counts as at/below).

    Call ``consume`` for each ``MetricStep``, then ``result`` after the stream ends.
    """

    def __init__(
        self,
        threshold: float,
        *,
        value_getter: Callable[[MetricStep], float | None] | None = None,
        treat_initial_above_as_cross: bool = True,
    ) -> None:
        self._threshold = threshold
        self._getter = value_getter or (lambda s: s.jump_ratio)
        self._treat_initial = treat_initial_above_as_cross
        self.reset()

    def reset(self) -> None:
        self._prev: float | None = None
        self._prev_defined = False
        self._last_break: RegimeBreak | None = None

    def consume(self, step: MetricStep) -> None:
        val = self._getter(step)
        if val is None or val != val:
            return
        if not self._prev_defined:
            prev_at_or_below = self._treat_initial
        else:
            prev_at_or_below = self._prev is not None and self._prev <= self._threshold
        crossed = val > self._threshold and prev_at_or_below
        if crossed:
            self._last_break = RegimeBreak(index=step.index, x=step.x, value=val, step=step)
        self._prev = val
        self._prev_defined = True

    @property
    def last_break(self) -> RegimeBreak | None:
        return self._last_break

    def result(self) -> RegimeBreak | None:
        """Alias for ``last_break`` after the full stream has been consumed."""
        return self._last_break


@dataclass(frozen=True)
class RegimeBreak:
    index: int
    x: float
    value: float
    step: MetricStep


# Collections of steps / results reused across the API.
MetricStepSeries: TypeAlias = tuple[list[MetricStep], RegimeBreak | None]


def run_processor_on_indexed_stream(
    indexed_values: Iterator[tuple[int, float]],
    processor: MetricStepPipeline,
    detector: FinalRegimeBreakDetector | None = None,
) -> MetricStepSeries:
    """
    Like ``run_pipeline`` but each sample carries its own index (e.g. video frame number).

    Use with rows from ``save_frame_from_file.iter_frame_metrics_rows`` so ``MetricStep.index``
    matches the ``frame`` column used in the notebook CSV.
    """
    steps: list[MetricStep] = []
    processor.reset()
    if detector is not None:
        detector.reset()
    for idx, x in indexed_values:
        step = processor.step(idx, x)
        steps.append(step)
        if detector is not None:
            detector.consume(step)
    br = detector.result() if detector is not None else None
    return steps, br


def run_pipeline(
    values: Iterator[float],
    processor: MetricStepPipeline,
    detector: FinalRegimeBreakDetector | None = None,
) -> MetricStepSeries:
    """
    Consume a stream: collect all steps and optionally track final regime break.

    Indices are ``0 .. n-1``. For frame-number alignment, use ``run_processor_on_indexed_stream``.

    Returns ``(steps, break_info)`` where ``break_info`` is None if no detector or no crossing.
    """
    return run_processor_on_indexed_stream(
        ((i, float(v)) for i, v in enumerate(values)),
        processor,
        detector,
    )


def notebook_aligned_jump_ratio_processor(
    *,
    ewm_span: float = 50,
    rolling_window: int = 50,
    epsilon: float = 1e-5,
) -> StreamingJumpRatioProcessor:
    """
    Defaults aligned with ``analyze_stats.ipynb`` (rolling top-half vs EWM baseline on the same scalar).
    """
    return StreamingJumpRatioProcessor(
        numerator=RollingTopHalfMean(rolling_window),
        baseline=EWMean(ewm_span),
        epsilon=epsilon,
    )


def run_jump_ratio_on_metric_rows(
    rows: Iterator[MetricRow],
    *,
    column: str = "laplacian_variance",
    processor: MetricStepPipeline | None = None,
    detector: FinalRegimeBreakDetector | None = None,
) -> MetricStepSeries:
    """
    Wire CSV-style metric dicts into the jump-ratio processor (streaming, no file).

    Typical source: ``save_frame_from_file.iter_frame_metrics_rows(...)``.
    """
    proc = processor or notebook_aligned_jump_ratio_processor()
    return run_processor_on_indexed_stream(
        ((int(r["frame"]), float(r[column])) for r in rows),
        proc,
        detector,
    )


def iter_online_stats(
    rows: Iterator[MetricRow],
    *,
    specs: dict[str, tuple[str, OnlineStatistic]],
    frame_key: str = "frame",
) -> Iterator[MetricStatsRow]:
    """
    Stream online statistics for multiple fields from a dict-row iterator.

    Parameters
    ----------
    rows:
        Iterator of input rows, typically from ``iter_frame_metrics_rows``.
    specs:
        Mapping ``output_name -> (source_key, OnlineStatistic)``.
        For each output name, the given ``OnlineStatistic`` is updated using
        ``row[source_key]`` on every step.
    frame_key:
        Optional key in the input rows that carries the frame/index value
        (default: ``"frame"``). When present, it is copied through to the
        output rows as an ``int``.

    Yields
    ------
    dict[str, float | int]
        For each input row, a dict containing:
        - ``frame_key`` (if present in the input row), and
        - one entry per ``output_name`` in ``specs``, holding the current
          statistic value after ingesting that row.

    Examples
    --------
    Basic usage with frame metric rows::

        from save_frame_from_file import iter_frame_metrics_rows
        from streaming_regime import EWMean, RollingTopHalfMean, iter_online_stats

        rows = iter_frame_metrics_rows(image_iterator, total_frames=total_frames, fps=fps)
        specs = {
            "lap_short": ("laplacian_variance", RollingTopHalfMean(window=50)),
            "lap_long":  ("laplacian_variance", EWMean(span=50)),
        }

        for stats_row in iter_online_stats(rows, specs=specs):
            print(stats_row["frame"], stats_row["lap_short"], stats_row["lap_long"])

    The same iterator can be materialized into a DataFrame instead of being
    consumed online::

        import pandas as pd

        df_stats = pd.DataFrame(iter_online_stats(rows, specs=specs))

    This function is intentionally generic and readable: you can feed its
    output into jump-ratio analysis, logging, plotting, or any other
    downstream logic.
    """

    # Reset all statistics before streaming.
    for _out_name, (_src_key, stat) in specs.items():
        stat.reset()

    for row in rows:
        metrics_out: MetricStatsRow = {}
        if frame_key in row:
            metrics_out[frame_key] = int(row[frame_key])
        for out_name, (src_key, stat) in specs.items():
            value = float(row[src_key])
            metrics_out[out_name] = stat.update(value)
        yield metrics_out


def iter_jump_ratio_from_stats(
    stats_rows: Iterator[MetricStatsRow],
    *,
    frame_key: str = "frame",
    numerator_key: str,
    baseline_key: str,
    epsilon: float = 1e-5,
) -> Iterator[MetricStatsRow]:
    """
    Compute a jump ratio from a stream of stats dicts.

    This is a thin, readable layer over the core jump-ratio definition:
    for each row, it takes ``numerator_key`` and ``baseline_key`` from the
    stats dict and emits a new dict with a ``"jump_ratio"`` field.

    Parameters
    ----------
    stats_rows:
        Iterator of stats dicts, e.g. produced by ``iter_online_stats``.
    frame_key:
        Optional key that carries the frame/index value (default: ``"frame"``).
        When present, it is copied through to each output row.
    numerator_key, baseline_key:
        Keys in each stats row that provide the numerator and baseline values.
    epsilon:
        Small stabilizer added to numerator and baseline to avoid division by
        zero (default: ``1e-5``).

    Yields
    ------
    dict[str, float | int]
        For each stats row, a dict containing:
        - ``frame_key`` (if present),
        - the raw numerator and baseline values under their original keys, and
        - a ``"jump_ratio"`` field.

    Examples
    --------
    Combine with ``iter_online_stats`` to compute jump ratios in a second
    streaming pass::

        from streaming_regime import (
            EWMean,
            RollingTopHalfMean,
            iter_online_stats,
            iter_jump_ratio_from_stats,
        )

        rows = iter_frame_metrics_rows(image_iterator, total_frames=total_frames, fps=fps)
        specs = {
            "lap_short": ("laplacian_variance", RollingTopHalfMean(window=50)),
            "lap_long":  ("laplacian_variance", EWMean(span=50)),
        }

        stats_rows = iter_online_stats(rows, specs=specs)
        jr_rows = iter_jump_ratio_from_stats(
            stats_rows,
            numerator_key="lap_short",
            baseline_key="lap_long",
        )

        for row in jr_rows:
            print(row["frame"], row["jump_ratio"])
    """


def iter_metrics_with_jump_ratios(
    rows: Iterator[MetricRow],
    *,
    source_key: str = "laplacian_variance",
    short_name: str = "lap_short",
    long_name: str = "lap_long",
    jump_name: str = "lap_jump_ratio",
    ewm_span: float = 50,
    rolling_window: int = 50,
    epsilon: float = 1e-5,
) -> Iterator[MetricRow]:
        
    """ 
    Convenience iterator: original metric rows + online jump-ratio columns.

    For each incoming metrics row (e.g. from ``iter_frame_metrics_rows``),
    this function:

    - reads ``row[source_key]``,
    - maintains a short-range statistic (rolling top-half mean) and a
      long-range statistic (EWM) on that value, and
    - yields a new dict that contains all original keys plus:

      - ``short_name``: current short-range statistic,
      - ``long_name``: current long-range statistic,
      - ``jump_name``: jump ratio
        ``(epsilon + short_name) / (epsilon + long_name)`` when both are finite,
        or ``nan`` otherwise.

    Parameters
    ----------
    rows:
        Iterator of metric rows, typically from ``iter_frame_metrics_rows``.
    source_key:
        Key in each row that provides the scalar to analyse (default:
        ``"laplacian_variance"``).
    short_name, long_name, jump_name:
        Column names to attach to each output row for the short/long
        statistics and the resulting jump ratio.
    ewm_span, rolling_window, epsilon:
        Parameters forwarded to the underlying EWM and rolling top-half
        implementations and the jump-ratio definition.

    Yields
    ------
    dict[str, Any]
        For each input row, a shallow copy containing all original keys plus
        the three additional jump-ratio related columns.

    Notes
    -----
    This is equivalent in spirit to wiring ``rows`` through
    ``iter_online_stats`` and then ``iter_jump_ratio_from_stats``, but keeps
    the public surface very simple: a single enriched iterator that you can
    consume online or materialize into a DataFrame.
    """

    short_stat = RollingTopHalfMean(rolling_window)
    long_stat = EWMean(ewm_span)

    short_stat.reset()
    long_stat.reset()

    for row in rows:
        # Shallow copy so callers can keep using the original row if they want.
        out: MetricRow = dict(row)

        value = float(row[source_key])
        short_val = short_stat.update(value)
        long_val = long_stat.update(value)

        out[short_name] = short_val
        out[long_name] = long_val

        if short_val == short_val and long_val == long_val:
            jr = (epsilon + short_val) / (epsilon + long_val)
        else:
            jr = float("nan")
        out[jump_name] = jr

        yield out
