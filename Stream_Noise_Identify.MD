# Stream noise identification

This document describes how **per-frame image metrics**, **streaming jump-ratio analysis**, and **regime-break detection** fit together in VidUtils. The goal is to spot when video content transitions from stable "order" toward chaotic noise (e.g. black frames, decoder garbage) **without writing a CSV**, while staying aligned with the same definitions used when a CSV **is** produced (`save_frame_from_file.py`) and analyzed (`analyze_stats.ipynb`).

---

## 1. Problem and intent

- **Offline path:** Decode frames, compute metrics, append rows to a CSV (`frame_metrics.csv`, `frame_metrics_2.csv`, …), load in pandas in a notebook, plot EMA / rolling statistics / jump ratio.
- **Streaming path:** For each frame, compute the **same metric row** as the CSV would contain, then feed a chosen scalar (often `laplacian_variance`) into an **online** jump-ratio pipeline and optionally a **last upward crossing** detector.

The streaming stack is meant for **experimentation**: swap the scalar column, EWM vs ring buffer, thresholds, and break rules without changing the underlying frame metric definitions.

---

## 2. End-to-end data flow

```mermaid
flowchart LR
  subgraph video [Video decode]
    V[Video file]
    I[get_image_iterator_from_file]
    V --> I
  end
  subgraph metrics [Per-frame metrics]
    R[iter_frame_metrics_rows]
    D[dict row: same keys as CSV]
    I --> R
    R --> D
  end
  subgraph jump [Jump ratio]
    P[StreamingJumpRatioProcessor]
    S[MetricStep stream]
    D --> P
    P --> S
  end
  subgraph regime [Regime break]
    F[FinalRegimeBreakDetector]
    B[RegimeBreak or None]
    S --> F
    F --> B
  end
```

1. **Iterator of `(frame_no, frame_bgr)`** — same source as `find_last_proper_video_frame` (e.g. `get_image_iterator_from_file`).
2. **`iter_frame_metrics_rows(iterator, fps)`** — yields one **dict per frame**; keys match the CSV written by `debug_csv` in `find_last_proper_video_frame`.
3. **Scalar selection** — typically `laplacian_variance`; could be `mean`, `combined_entropy`, etc.
4. **`run_jump_ratio_on_metric_rows`** / **`run_jump_ratio_on_metric_rows_multi`** (or `run_processor_on_indexed_stream`) — produce **`MetricStep`** objects with `jump_ratio` when the online numerator and baseline are both finite.
5. **`FinalRegimeBreakDetector`** — updates on each step; after the stream ends, **`result()`** is the **last** time the chosen value **crossed upward** above a threshold (see §6).

---

## 3. Frame metrics (CSV parity): `save_frame_from_file.py`

### 3.1 Constants

| Name | Value | Role |
|------|--------|------|
| `FRAME_METRICS_COMBINED_LOCAL_WEIGHT` | `0.7` | Weight of local entropy in `combined_entropy`. |
| `FRAME_METRICS_COMBINED_TEMPORAL_WEIGHT` | `0.3` | Weight of temporal entropy in `combined_entropy`. |

These weights are used in **`compute_frame_metrics_row`** and match the loop inside **`find_last_proper_video_frame`** (0.7 / 0.3). The helper **`is_proper_frame`** uses the same idea with a parameter `temporal_entropy_weight` defaulting to `0.3`, i.e. \((1 - w)\cdot \text{local} + w\cdot \text{temporal}\).

### 3.2 Per-metric definitions (high level)

| CSV key | Source in code | Notes |
|---------|------------------|--------|
| `frame` | Caller-supplied `frame_no` | Integer index from the decoder iterator. |
| `timestamp` | `frame_no / fps` (0 if `fps <= 0`) | Seconds if `fps` is accurate. |
| `mean` | `np.mean(frame_bgr)` | All channels. |
| `local_entropy` | `calculate_local_entropy(frame_bgr)` | Gaussian blur residual, then Shannon entropy (`utils._compute_entropy`). |
| `temporal_entropy` | `calculate_temporal_entropy(frame_bgr, previous_bgr)` | `0.0` if `previous_bgr is None`; else entropy of `absdiff` between consecutive frames (BGR). |
| `combined_entropy` | Weighted sum per §3.1 | Single scalar for "proper frame" heuristics. |
| `edge_density` | `calculate_edge_density(gray)` | Grayscale from BGR when needed; Canny edge fraction. |
| `hist_entropy` | `calculate_histogram_entropy(frame_bgr)` | Entropy of pixel values (multi-channel handled in `_compute_entropy`). |
| `laplacian_variance` | `calculate_laplacian_variance(gray)` | Variance of Laplacian (sharpness / high-frequency energy). |

### 3.3 Public API for streaming (no CSV)

- **`compute_frame_metrics_row(frame_bgr, previous_bgr, frame_no, fps) -> dict`**  
  One row, same keys and formulas as the CSV row dict.

- **`iter_frame_metrics_rows(image_iterator, fps)`**  
  Generator that keeps **`previous_bgr`** exactly like **`find_last_proper_video_frame`**: after each yield, `previous_bgr = frame_bgr`.

**Contract:** Any row produced here should match what would have been appended to `metrics_log` for that frame when `debug_csv` is enabled (assuming the same `fps`, frame order, and frame arrays).

---

## 4. Streaming jump ratio: `streaming_regime.py`

### 4.1 Idea

For a scalar time series \(x_t\) (e.g. Laplacian variance):

- **Baseline** \(b_t\): slow moving average of \(x_t\) — implemented as an exponentially weighted mean (EWM) consistent with pandas' default `ewm(span=...)`.
- **Numerator** \(n_t\): a "high side" statistic over recent \(x\) — default is the **mean of the largest \(\lceil k/2 \rceil\)** samples in a fixed-length window of length \(W\) (rolling "top half" mean).
- **Jump ratio** (when both are finite):

\[
\text{jr}_t = \frac{\varepsilon + n_t}{\varepsilon + b_t}
\]

with small \(\varepsilon\) (default `1e-5`) to avoid division by zero.

This mirrors the notebook's spirit: compare a **responsive upper-tail level** to a **slow baseline**. You can replace `RollingTopHalfMean` or `EWMean` with other **`OnlineStatistic`** implementations for experiments.

### 4.2 `OnlineStatistic` protocol

Implementations must provide:

- **`reset()`** — clear internal state.
- **`update(x: float) -> float`** — ingest one sample; return the current estimate, or **`nan`** if not ready (e.g. buffer not full).

### 4.3 `EWMean` (exponentially weighted mean)

- **`alpha = 2 / (span + 1)`** — same `span` convention as pandas.
- **`adjust=True` (default):** matches **`Series.ewm(span=span).mean()`** (pandas default `adjust=True`), using the standard recursive numerator/denominator form.
- **`adjust=False`:** simple recursive EWMA \(y_t = (1-\alpha)y_{t-1} + \alpha x_t\).

Use **`adjust=True`** when you want **bit-for-bit alignment** with notebook/pandas EWM on the same series.

### 4.4 `EWStd`

O(1) approximate exponentially weighted standard deviation: separate EWM on \(x\) and on \(x^2\), then \(\sqrt{\max(0, E[x^2] - E[x]^2)}\). Both inner means use the default **`EWMean(..., adjust=True)`**. This is a **simple biased** variant, not a full pandas `ewm().std()` port; use when you need a cheap online spread for custom metrics.

### 4.5 `RollingTopHalfMean(window, min_periods=None)`

- Stores the last **`window`** values (ring buffer).
- When `len(buffer) >= min_periods` (default `window`), returns the mean of the **largest \(\lceil n/2 \rceil\)** values in the buffer (same rule as in `analyze_stats.ipynb` for the rolling "top 50%" mean).
- **Memory:** \(\Theta(\text{window})\); **exact** top-half mean (no sketch).

### 4.6 `StreamingJumpRatioProcessor`

- **`numerator`**, **`baseline`**: any **`OnlineStatistic`**.
- **`epsilon`**: stabilizer in the ratio.
- **`value_fn`**: optional `x -> float` applied to each incoming sample before updates (e.g. `log1p`, scale).

**`step(index, x) -> MetricStep`:** sets `numerator`, `baseline`, and `jump_ratio` when both estimates are finite; **`MetricStep.x`** is the **raw** `x` passed in (not the transformed value), so it stays comparable to the original stream.

**`process(stream)`** uses indices `0..n-1` only; for real frame numbers use **`run_processor_on_indexed_stream`** or **`run_jump_ratio_on_metric_rows`**.

### 4.7 `MetricStep` and `MetricStepProcessor`

- **`MetricStep`**: `index`, `x`, optional `jump_ratio` / `numerator` / `baseline`, and **`extras`** for custom fields.
- **`MetricStepProcessor`**: protocol for alternate processors with the same **`reset` / `step` / `process`** surface.

### 4.8 Convenience builders

| Function | Purpose |
|----------|---------|
| **`notebook_aligned_jump_ratio_processor(ewm_span=50, rolling_window=50, epsilon=1e-5)`** | Returns a **`StreamingJumpRatioProcessor`** with **`RollingTopHalfMean(rolling_window)`** and **`EWMean(ewm_span)`** (pandas-aligned default). |
| **`run_processor_on_indexed_stream(indexed_values, processor, detector=None)`** | `indexed_values` is `Iterator[tuple[int, float]]`; **`MetricStep.index`** is the provided index (e.g. CSV `frame`). |
| **`run_pipeline(values, processor, detector=None)`** | Same as indexed stream with **`enumerate(values)`**. |
| **`run_jump_ratio_on_metric_rows(rows, column="laplacian_variance", ...)`** | Consumes dict rows (e.g. from **`iter_frame_metrics_rows`**); builds **`(int(r["frame"]), float(r[column]))`** stream. |
| **`run_jump_ratio_on_metric_rows_multi(rows, columns=None, processor_factory=None, detector_factory=None)`** | Like `run_jump_ratio_on_metric_rows` but for **multiple numeric columns at once**; consumes the row iterator a single time and returns a mapping `column -> (steps, last_break)`. |

---

## 5. Regime break: `FinalRegimeBreakDetector`

### 5.1 "Last" upward crossing

The detector tracks the **last** time a scalar **crosses upward** through a threshold \(\theta\):

- At time \(t\), let \(v_t\) be the value (default: **`step.jump_ratio`**).
- Define **cross** when \(v_t > \theta\) **and** the **previous finite** \(v_{t-1}\) satisfies \(v_{t-1} \le \theta\).
- **First step:** if **`treat_initial_above_as_cross=True`** (default), the "previous is at or below threshold" condition is treated as true before any prior sample exists — so if the very first \(v\) is already above \(\theta\), it counts as a cross.

Every time a cross occurs, **`last_break`** is overwritten; **`result()`** after the full stream is therefore the **final** such event (useful for "when did things last go chaotic relative to this rule?").

### 5.2 Customization

- **`value_getter(step) -> float | None`** — e.g. another column encoded in **`extras`**, or a function of `numerator`/`baseline`.
- **`treat_initial_above_as_cross`** — set **`False`** if the first sample must not count as a cross unless a prior value existed and was \(\le \theta\).

### 5.3 `RegimeBreak`

Frozen dataclass: **`index`**, **`x`** (raw scalar from the step), **`value`** (the compared value, e.g. jump ratio), and **`step`** (full **`MetricStep`**).

---

## 6. Alignment with `analyze_stats.ipynb`

The notebook typically:

1. Loads **`frame_metrics.csv`** (or equivalent).
2. Picks a metric column (e.g. **`laplacian_variance`**).
3. Computes pandas **`ewm(span=...).mean()`** on that column (default **`adjust=True`**).
4. Computes a rolling "top half" mean over a window (same mathematical definition as **`RollingTopHalfMean`**).
5. Forms a ratio with **`epsilon`** and may define events on threshold crossings.

**Streaming alignment checklist:**

- Use **`EWMean(span, adjust=True)`** (default) for the baseline to match pandas **`ewm(span=span).mean()`**.
- Use the same **`rolling_window`**, **`ewm_span`**, **`epsilon`**, and **metric column** as in the notebook.
- Use **`run_processor_on_indexed_stream`** or **`run_jump_ratio_on_metric_rows`** so **`MetricStep.index`** equals **`frame`** in the CSV, not a 0-based enumeration offset.

The notebook may evolve independently (e.g. extra plots, `metric_legends`); this doc describes the **code** contracts above.

---

## 7. Complexity and memory (reference)

| Component | Time per step (typical) | Memory |
|-----------|-------------------------|--------|
| **`EWMean`** | \(O(1)\) | \(O(1)\) |
| **`EWStd`** | \(O(1)\) | \(O(1)\) |
| **`RollingTopHalfMean(W)`** | \(O(W)\) with partition on buffer | \(O(W)\) |
| **`StreamingJumpRatioProcessor`** | Sum of numerator + baseline | Same as components |
| **`FinalRegimeBreakDetector`** | \(O(1)\) | \(O(1)\) |

---

## 8. Example: full streaming pipeline (video → last break)

```python
import cv2
from save_frame_from_file import get_image_iterator_from_file, iter_frame_metrics_rows
from streaming_regime import (
    notebook_aligned_jump_ratio_processor,
    FinalRegimeBreakDetector,
    run_jump_ratio_on_metric_rows,
)

video_path = "example.mp4"
cap = cv2.VideoCapture(video_path)
fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
cap.release()

proc = notebook_aligned_jump_ratio_processor(ewm_span=50, rolling_window=50, epsilon=1e-5)
det = FinalRegimeBreakDetector(1.5)  # threshold on jump_ratio; tune for your data

frames = get_image_iterator_from_file(video_path)
rows = iter_frame_metrics_rows(frames, fps)
steps, last_break = run_jump_ratio_on_metric_rows(rows, processor=proc, detector=det)

if last_break is not None:
    print("Last upward cross:", last_break.index, last_break.value, last_break.x)
else:
    print("No crossing under current rule")
```

**Note:** `get_image_iterator_from_file` uses **`sys.argv[1]`** in its default signature; in library code you may pass **`filename=video_path`** explicitly (see `save_frame_from_file.py`).

---

## 9. Example: metrics only (no jump ratio)

```python
from save_frame_from_file import iter_frame_metrics_rows, get_image_iterator_from_file

fps = 25.0
for row in iter_frame_metrics_rows(get_image_iterator_from_file("clip.mp4"), fps):
    if row["combined_entropy"] > some_limit:
        ...
```

You can also run jump-ratio analysis on **several metrics in one pass** using `run_jump_ratio_on_metric_rows_multi`:

```python
from streaming_regime import run_jump_ratio_on_metric_rows_multi

rows = iter_frame_metrics_rows(get_image_iterator_from_file("clip.mp4"), fps)
results = run_jump_ratio_on_metric_rows_multi(
    rows,
    columns=["mean", "local_entropy", "laplacian_variance"],
)

mean_steps, mean_break = results["mean"]
lap_steps, lap_break = results["laplacian_variance"]
```

---

## 10. Automated tests

- **Location:** `tests/test_streaming_regime.py`, `tests/test_frame_metrics_pipeline.py`.
- **Runner:** from repo root, with dev deps:  
  `uv sync --extra dev`  
  `uv run pytest tests/`
- **Config:** `pyproject.toml` sets **`pythonpath = ["."]`** and **`testpaths = ["tests"]`** so imports like `streaming_regime` and `save_frame_from_file` resolve.

**What is covered (summary):**

- **`EWMean`** vs pandas **`ewm(span=...).mean()`** (default **`adjust=True`**).
- **`RollingTopHalfMean`** vs the partition definition used in the notebook.
- **`StreamingJumpRatioProcessor`** / **`run_pipeline`** vs pandas on the same synthetic series.
- **`run_processor_on_indexed_stream`** preserves explicit indices.
- **`FinalRegimeBreakDetector`** crossing semantics and NaN handling.
- **`run_jump_ratio_on_metric_rows`** roundtrip vs pandas on dict rows.
- Frame metrics: constants, first-frame temporal entropy, iterator keys and basic invariants.
- Optional: if **`frame_metrics.csv`** exists at the **repository root**, jump ratios are checked against pandas on the **`laplacian_variance`** column (path resolved from `tests/`).

---

## 11. File reference

| File | Role |
|------|------|
| `save_frame_from_file.py` | Metric definitions, CSV logging, **`compute_frame_metrics_row`**, **`iter_frame_metrics_rows`**, **`find_last_proper_video_frame`**. |
| `streaming_regime.py` | Online statistics, jump-ratio processor, regime-break detector, run helpers. |
| `utils.py` | **`_compute_entropy`** and other shared image helpers. |
| `analyze_stats.ipynb` | Exploratory analysis of CSV columns; should stay consistent with §6 when comparing numbers. |
| `tests/test_streaming_regime.py` | Tests for `streaming_regime`. |
| `tests/test_frame_metrics_pipeline.py` | Tests for frame metrics + CSV alignment. |
| `Stream_Noise_Identify.MD` | This document. |

---

## 12. Document location

This file is **`Stream_Noise_Identify.MD`** in the VidUtils repository root.
