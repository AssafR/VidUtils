"""
Streaming jump-ratio style metrics and final regime-break detection.

Designed for experiments: swap online estimators (EWM vs ring buffer), preprocess
the raw stream, and plug different break rules without rewriting the pipeline.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol, runtime_checkable

import numpy as np


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
    """One time step through the streaming pipeline."""

    index: int
    x: float
    jump_ratio: float | None = None
    numerator: float | None = None
    baseline: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MetricStepProcessor(Protocol):
    """Implement for custom metrics; must match ``StreamingJumpRatioProcessor`` shape."""

    def reset(self) -> None: ...

    def step(self, index: int, x: float) -> MetricStep: ...

    def process(self, stream: Iterator[float]) -> Iterator[MetricStep]: ...


class StreamingJumpRatioProcessor:
    """
    For each raw sample ``x`` (optionally transformed), update numerator and baseline
    estimators and emit ``(epsilon + num) / (epsilon + den)`` when both are finite.

    Inject any ``OnlineStatistic`` implementations to experiment (EWM vs ring, etc.).
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


def run_processor_on_indexed_stream(
    indexed_values: Iterator[tuple[int, float]],
    processor: MetricStepProcessor,
    detector: FinalRegimeBreakDetector | None = None,
) -> tuple[list[MetricStep], RegimeBreak | None]:
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
    processor: MetricStepProcessor,
    detector: FinalRegimeBreakDetector | None = None,
) -> tuple[list[MetricStep], RegimeBreak | None]:
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
    rows: Iterator[dict[str, Any]],
    *,
    column: str = "laplacian_variance",
    processor: MetricStepProcessor | None = None,
    detector: FinalRegimeBreakDetector | None = None,
) -> tuple[list[MetricStep], RegimeBreak | None]:
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


def run_jump_ratio_on_metric_rows_multi(
    rows: Iterator[dict[str, Any]],
    *,
    columns: list[str] | None = None,
    processor_factory: Callable[[str], MetricStepProcessor] | None = None,
    detector_factory: Callable[[str], FinalRegimeBreakDetector | None] | None = None,
) -> dict[str, tuple[list[MetricStep], RegimeBreak | None]]:
    """
    Run the jump-ratio processor on **multiple numeric columns** from the same metric rows stream.

    This lets you consume rows (e.g. from ``iter_frame_metrics_rows``) **once** and
    compute jump-ratios for several fields in a single streaming pass.

    Parameters
    ----------
    rows:
        Iterator of dict-like metric rows. Each row must contain a ``"frame"`` key
        plus one or more numeric columns.
    columns:
        Optional explicit list of column names to process. If ``None``, the
        columns are inferred from the first row as all numeric keys except ``"frame"``.
    processor_factory:
        Optional factory ``factory(column_name) -> MetricStepProcessor``.
        Defaults to ``notebook_aligned_jump_ratio_processor`` for every column.
    detector_factory:
        Optional factory ``factory(column_name) -> FinalRegimeBreakDetector | None``
        to attach independent regime-break detectors per column. If omitted,
        no detectors are used.

    Returns
    -------
    dict[str, tuple[list[MetricStep], RegimeBreak | None]]:
        Mapping ``column_name -> (steps, last_break)`` mirroring
        ``run_jump_ratio_on_metric_rows`` for each processed column.
    """

    try:
        first_row = next(rows)
    except StopIteration:
        return {}

    # Infer default columns from the first row if not provided.
    if columns is None:
        inferred: list[str] = []
        for k, v in first_row.items():
            if k == "frame":
                continue
            # Treat ints / floats (including numpy scalars) as numeric.
            if isinstance(v, (int, float, np.floating, np.integer)):
                inferred.append(k)
        columns = inferred

    if not columns:
        return {}

    proc_factory = processor_factory or (lambda _col: notebook_aligned_jump_ratio_processor())
    det_factory = detector_factory or (lambda _col: None)

    processors: dict[str, MetricStepProcessor] = {col: proc_factory(col) for col in columns}
    detectors: dict[str, FinalRegimeBreakDetector | None] = {
        col: det_factory(col) for col in columns
    }

    # Reset all processors / detectors before streaming.
    for p in processors.values():
        p.reset()
    for d in detectors.values():
        if d is not None:
            d.reset()

    steps_by_col: dict[str, list[MetricStep]] = {col: [] for col in columns}

    def _consume_row(row: dict[str, Any]) -> None:
        idx = int(row["frame"])
        for col in columns or ():
            val = float(row[col])
            step = processors[col].step(idx, val)
            steps_by_col[col].append(step)
            det = detectors[col]
            if det is not None:
                det.consume(step)

    # Feed the first row we already pulled for inference, then the rest.
    _consume_row(first_row)
    for row in rows:
        _consume_row(row)

    results: dict[str, tuple[list[MetricStep], RegimeBreak | None]] = {}
    for col in columns:
        det = detectors[col]
        last_break = det.result() if det is not None else None
        results[col] = (steps_by_col[col], last_break)

    return results
