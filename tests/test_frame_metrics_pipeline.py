"""Tests aligning ``save_frame_from_file`` metrics with streaming jump-ratio pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from save_frame_from_file import (
    FRAME_METRICS_COMBINED_LOCAL_WEIGHT,
    FRAME_METRICS_COMBINED_TEMPORAL_WEIGHT,
    compute_frame_metrics_row,
    iter_frame_metrics_rows,
)
from streaming_regime import (
    notebook_aligned_jump_ratio_processor,
    run_processor_on_indexed_stream,
)


def test_combined_entropy_weights_constant_matches_implied_is_proper_frame_default() -> None:
    w_temp = 0.3
    w_loc = 1.0 - w_temp
    assert w_loc == pytest.approx(FRAME_METRICS_COMBINED_LOCAL_WEIGHT)
    assert w_temp == pytest.approx(FRAME_METRICS_COMBINED_TEMPORAL_WEIGHT)


def test_compute_frame_metrics_row_first_frame_temporal_zero() -> None:
    h, w = 24, 24
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    row = compute_frame_metrics_row(frame, None, frame_no=0, fps=25.0)
    assert row["frame"] == 0
    assert row["temporal_entropy"] == 0.0
    assert row["timestamp"] == 0.0
    assert "laplacian_variance" in row
    assert row["mean"] == 0.0


def test_iter_frame_metrics_rows_matches_sequential_compute() -> None:
    """Iterator must mirror the CSV loop: temporal entropy 0 on first row, advancing ``previous_bgr``."""
    frames = [
        (0, np.full((16, 16, 3), 10, dtype=np.uint8)),
        (1, np.full((16, 16, 3), 200, dtype=np.uint8)),
    ]
    rows = list(iter_frame_metrics_rows(iter(frames), fps=50.0))
    assert len(rows) == 2
    assert rows[0]["frame"] == 0
    assert rows[1]["frame"] == 1
    assert rows[0]["temporal_entropy"] == 0.0
    assert rows[1]["timestamp"] == pytest.approx(1 / 50.0)
    # Second frame differs strongly from the first (constant 10 vs 200).
    assert rows[1]["mean"] > rows[0]["mean"]
    assert set(rows[0].keys()) == {
        "frame",
        "timestamp",
        "mean",
        "local_entropy",
        "temporal_entropy",
        "combined_entropy",
        "edge_density",
        "hist_entropy",
        "laplacian_variance",
    }


_REPO_ROOT = Path(__file__).resolve().parent.parent
_CSV_PATH = _REPO_ROOT / "frame_metrics.csv"


@pytest.mark.skipif(
    not _CSV_PATH.is_file(),
    reason="frame_metrics.csv not present in repo root",
)
def test_streaming_jump_ratio_matches_pandas_on_saved_csv() -> None:
    """End-to-end: CSV ``laplacian_variance`` column vs same pipeline in streaming form."""
    df = pd.read_csv(_CSV_PATH)
    rows = df.to_dict("records")

    span = 50
    window = 50
    eps = 1e-5
    s = df["laplacian_variance"].astype(float)
    ema = s.ewm(span=span, adjust=True).mean()

    def _top_half(a: np.ndarray) -> float:
        v = a[~np.isnan(a)]
        k = max(1, int(np.ceil(0.5 * v.size)))
        return float(np.mean(np.partition(v, -k)[-k:]))

    rolling = s.rolling(window=window, min_periods=window).apply(_top_half, raw=True)
    expected_jr = (eps + rolling) / (eps + ema)

    proc = notebook_aligned_jump_ratio_processor(
        ewm_span=span, rolling_window=window, epsilon=eps
    )

    indexed = ((int(r["frame"]), float(r["laplacian_variance"])) for r in rows)
    steps, _ = run_processor_on_indexed_stream(indexed, proc)

    assert len(steps) == len(df)
    for i, step in enumerate(steps):
        assert step.index == int(df.iloc[i]["frame"])
        exp = expected_jr.iloc[i]
        if exp != exp:
            assert step.jump_ratio is None or step.jump_ratio != step.jump_ratio
        else:
            assert step.jump_ratio == pytest.approx(float(exp), rel=0, abs=1e-9)
