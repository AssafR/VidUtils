# Content-focused video padding detection and trimming

Tooling to find and remove uninformative padding at the beginning or end of a
video, including blue screens, black frames, and high-entropy static ("snow"),
without re-encoding.

## `trim_padding.py`

```
uv run trim_padding.py tests\data\trailing-keyframe-overlap.mkv --dry-run
uv run trim_padding.py tests\data\trailing-keyframe-overlap.mkv -o tests\artifacts\sample.mkv --overwrite --dry-run
uv run trim_padding.py tests\data\trailing-keyframe-overlap.mkv --threshold 0.1 --dry-run
uv run trim_padding.py tests\data\trailing-keyframe-overlap.mkv --noise-entropy-threshold 4.5 --dry-run
uv run trim_padding.py tests\data\trailing-keyframe-overlap.mkv --black-entropy-threshold 1.9 --dry-run
uv run trim_padding.py tests\data\trailing-keyframe-overlap.mkv --black-brightness-threshold 5 --dry-run
uv run trim_padding.py tests\data\trailing-keyframe-overlap.mkv --min-removable-streak-seconds 10 --dry-run
uv run trim_padding.py tests\data\trailing-keyframe-overlap.mkv --report-signal-types --dry-run
```

These examples use tracked test media, so they can be run from the repository
root without substituting a filename. They are dry runs and do not write video
files. Remove `--dry-run` to perform the cut.

By default, the kept video is written next to the input as
`{input}_trimmed.mkv`. When a beginning or ending area is removed, that area is
also saved separately as `{input}_begin.mkv` or `{input}_end.mkv`, respectively,
for inspection. These inspection files use the input's filename stem even when
`-o` selects a different path for the trimmed video. `--overwrite` applies to
the trimmed output and any inspection clips. `--dry-run` prints every command
without creating any files.

### How blue and static are classified

- **Blue:** `calculate_blue_screen_distance` returns the fraction of pixels not
  recognized as blue by the HSV mask. The default `--threshold 0.1` means at
  least 90% of pixels must match the blue mask for a frame to be treated as
  blue/removable. Raising the threshold is more permissive and can mistake
  blue-toned footage for a blue screen.
- **Black screen:** a frame that is not classified as blue is labeled `BLACK`
  only when both its local entropy is at or below
  `--black-entropy-threshold` (default `1.9`) and its mean 8-bit BGR intensity
  is at or below `--black-brightness-threshold` (default `5`). Requiring both
  measurements avoids treating dim, low-detail footage as black.
- **Static/noise:** `calculate_local_entropy` measures entropy in the
  high-frequency residual after Gaussian blur. A frame with local entropy at or
  above `--noise-entropy-threshold` is treated as static/removable. The default
  is `4.5`. The older `is_proper_frame_params` heuristic uses `5.0`; the
  trimming threshold is intentionally more sensitive.
  Raise the cutoff to be more selective; lower it to classify more frames as static.
- **Combined rule:** a frame is removable if either rule matches. The same
  per-frame rule applies to blue, black-proxy, and static signals. An edge is
  removed only after it passes the sustained-window check below.

### Sustained edge check and reporting

The default minimum is 10 seconds, configured with
`--min-removable-streak-seconds` (the older `--min-black-streak-seconds` spelling
is still accepted). It applies to all removable signals, including STATIC.

#### Window score and the 90% rule

`calculate_removable_window_distance` samples a time interval and returns the
fraction of samples that are **not removable**. The sample interval is 0.5
seconds, or one frame interval for sources below 2 fps. With the default
10-second interval and a 25 fps source, this gives 20 samples.

Both edge searches use the same `is_confirmed_window` predicate:

| Window being confirmed | Pass condition | Meaning |
|---|---:|---|
| Removable | non-removable fraction `<= 0.10` | At least 90% of samples are BLUE, BLACK, or STATIC. |
| Informative | non-removable fraction `>= 0.90` | At least 90% of samples are not classified as removable. |

The rule tolerates up to 10% of samples that disagree with the expected signal.
It confirms a sustained pattern; it does not require every frame in a passing
window to have the same label.

#### Beginning trim

The beginning search proceeds as follows:

1. Check the first `--min-removable-streak-seconds` of the file. It must pass
   the removable-window test (at least 90% removable).
2. If it passes, scan forward for an informative candidate frame. The candidate
   is accepted as the start of content only if the window from that frame
   through the following minimum-streak interval passes the informative-window
   test (at least 90% informative).
3. The first candidate that passes is the leading content boundary. An
   isolated content-classified frame inside a longer BLUE/BLACK/STATIC lead is
   ignored if its following window is not sufficiently informative.
4. If the initial window is removable but no informative streak can be
   confirmed, the script keeps the beginning rather than cutting based on a
   transient frame.

The 90% allowance means a few removable samples may occur in the confirmed
informative window, so the kept video can begin slightly before fully stable
content. There is one short-opening exception: if the detected leading
interval is shorter than the minimum, the script checks a blue-only window of
up to one second from the start. At least 90% of its samples must be blue to
remove that short BLUE lead. There is no corresponding short-opening exception
for BLACK-only or STATIC-only footage.

When the initial window passes, step 2 displays an open-ended progress bar with
the decoded frame count, current source timestamp, and frames-per-second rate.
It intentionally has no percentage or ETA: the scan stops as soon as a
following minimum-streak window confirms informative content, rather than
scanning the whole file. The displayed speed includes time spent validating
candidate windows. If no such window is found, it can scan to the last point
where a full confirmation window fits. To see this stage on a tracked sample,
run:

```
uv run trim_padding.py tests\data\blue-black-opening-content-streak.mkv --dry-run
```

#### Ending trim

The ending search uses a time-based bisection rather than decoding the whole
video sequentially:

1. A frame classified as informative is treated as content.
2. A frame classified as removable is treated as part of the ending streak
   only if the following minimum-streak interval passes the removable-window
   test (at least 90% removable).
3. The final minimum-streak interval is checked first to establish that a
   removable tail exists. Bisection then locates the last content frame before
   that confirmed tail.

This places the cut near the start of the confirmed removable tail, not 10
seconds after it. The trailing search assumes one final removable run;
multiple separated transitions near EOF can violate its bisection assumption.

By default, the script reports removable time ranges without labeling every
frame. Add `--report-signal-types` to scan the removed intervals and report
BLUE, BLACK, STATIC, or combinations. This optional scan can be slow on long
intervals; it does not change or validate the selected boundaries.

### Static heuristic in detail

`calculate_local_entropy(frame_bgr)` does not measure raw-frame entropy. It
first applies a 5-by-5 Gaussian blur, then subtracts that blurred image from
the original with an absolute difference:

```
residual = abs(frame - GaussianBlur(frame, 5x5))
```

This residual emphasizes small, high-frequency changes such as fine-grained
snow while suppressing broad smooth regions. `_compute_entropy` builds a
256-value histogram for each channel and computes Shannon entropy:

```
H = -sum(p(value) * log2(p(value)))
```

The channel entropies are averaged. For an 8-bit channel, the theoretical range
is 0–8 bits. A flat residual has entropy near 0; a residual whose values are
spread across many bins has higher entropy. The current script classifies a
frame as high-entropy static when `H >= 4.5` by default. It classifies a
non-blue frame as `BLACK` only when `H <= 1.9` and mean BGR intensity is at
most 5 on the 0–255 scale.

This is a useful signal, not a semantic guarantee that a frame is snow.
Fine texture, film grain, detailed foliage, text, or compression artifacts can
also raise local entropy. Conversely, heavily blurred, denoised, or low-bitrate
static may score below the cutoff. Blue, high-entropy static, and the low-entropy black proxy are classified
per-frame. A signal must also pass the 10-second sustained-window rule before
an edge is removed. The black entropy and brightness limits can be adjusted
independently with `--black-entropy-threshold` and
`--black-brightness-threshold`. Adjust `--min-removable-streak-seconds` for
sources where edge signals typically last longer or shorter.

Entropy scores depend on the clip, resolution, encoding, and decoder; the
default cutoff is not a universal calibration value. Use the runnable sample
measurement below, then compare representative static and content frames from
the video being tuned.

#### Tuning the current cutoff

The only static-classifier setting currently exposed by `trim_padding.py` is
`--noise-entropy-threshold`. To tune it for a new source:

1. Decode representative frames known to be snow/static and representative
   frames of real content, including the most detailed content.
2. Measure their scores with `calculate_local_entropy(frame_bgr)` from
   `save_frame_from_file.py`. For a quick, copy-pasteable sample measurement:

   ```
   uv run python -c "import av,sys; from save_frame_from_file import calculate_local_entropy; c=av.open(sys.argv[1]); f=next(c.decode(video=0)); print('first-frame local entropy:', calculate_local_entropy(f.to_ndarray(format='bgr24')))" tests\data\low-entropy-black-ending.mkv
   ```

3. Choose a cutoff that separates those two sets with some margin. If the
   score ranges overlap, a single entropy cutoff cannot reliably distinguish
   them; a higher cutoff favors avoiding false removals, while a lower cutoff
   catches weaker static at the cost of more false positives.
4. Run the command below and review the printed removable intervals before
   running a cut:

```
uv run trim_padding.py tests\data\low-entropy-black-ending.mkv --noise-entropy-threshold 5.5 --dry-run
```

#### Other metrics and possible variants

The repository has other noise-related metrics, but `trim_padding.py` does
not currently use them or expose them as alternatives:

| Metric/variant | What it could add | Main caveat |
|---|---|---|
| **Temporal entropy** (`calculate_temporal_entropy`) | Entropy of the pixel-wise difference between adjacent frames; flickering snow may score high. | Motion, scene cuts, and animation also produce large differences. |
| **Combined local + temporal entropy** (`is_proper_frame`) | Uses local and temporal entropy together; the existing helper also checks mean brightness. | Its combined thresholds and “proper frame” semantics are not wired into this cutter. |
| **Laplacian variance** (`calculate_laplacian_variance`) | Measures fine edges/high-frequency detail; static can cause a strong rise. | Sharp real content can also have high values, so absolute cutoffs are source-dependent. |
| **FFT high-band ratio** (`calculate_fft_highband_ratio`) | Measures the proportion of image energy in high spatial frequencies. | More computation and parameters; high-frequency real texture may resemble static. |
| **Per-frame hysteresis** | Require several consecutive decoded frames, or a smoothed score, before changing a frame's label. | The cutter currently confirms sampled fixed-duration windows instead; per-frame hysteresis is not implemented. |
| **Adaptive/source-calibrated threshold** | Estimate a cutoff from labeled static and content samples, or normalize scores against a known content baseline. | Requires representative samples and must handle sources where the score distributions overlap. |

`locate_noise.py` and `streaming_regime.py` are exploratory regime-change
tools: they analyze a stream of metrics such as Laplacian variance and detect
upward jump-ratio crossings. They are not currently used by this script as a
per-frame static classifier. To run the exploratory scan on tracked media, use:

```
uv run locate_noise.py tests\data\blue-black-opening-content-streak.mkv
```

The scan displays a frame window on a desktop. The underlying regime-break
logic has a non-visual automated test:

```
uv run pytest tests\test_streaming_regime.py -q
```

### End-to-end processing pipeline

1. **Classify frames.** Blue-mask matches, high-entropy static, and guarded
   near-black frames are removable signals. Other frames are treated as
   informative for edge detection. “Informative” here means only “not
   classified as removable”; it is not a semantic content detector.
2. **Find the beginning boundary.** `find_content_range` gets an initial
   content candidate, confirms the initial removable window, then searches for
   the first candidate whose following minimum-streak window is at least 90%
   informative. This is why a brief content-like dropout inside an opening
   black/blue run does not by itself end the trim.
3. **Find the ending boundary.** `find_last_dissimilar_frame_time` bisects on
   presentation time. A removable candidate must have a following
   minimum-streak window that is at least 90% removable. This avoids decoding
   the long middle sequentially.
4. **Snap and cut.** `find_keyframe_bounds` inspects packets and selects nearby
   keyframe bounds. FFmpeg then stream-copies with
   `-map 0 -map_metadata 0 -map_chapters 0 -c copy`. `-ss` uses the detected
   boundary time, and `-t` is calculated from the seek point to the end bound.
5. **Validate and report.** The output video span (`video_pts_span`) is checked
   against the detected content span; validation failure exits with an error.
   By default, the script reports interval times. `--report-signal-types`
   additionally scans removed intervals and labels them BLUE, BLACK, STATIC,
   or combinations; this can be slow on long intervals and does not change the
   boundaries.

The trailing bisection assumes the tail is one final removable run. Multiple
separated transitions near EOF can violate that assumption. The classifier can
also mistake detailed real footage for static, and the 90% rule intentionally
allows a small number of samples that disagree with a confirmed window. Review
the proposed ranges with `--dry-run` when using a new source. Variable frame
rate is untested; nominal frame rate is used only as a half-frame tolerance.

The module docstring also documents a possible future "smart cut" (re-encode
only the head GOP, copy the rest, concat) that is NOT implemented.

### Tracked regression-media examples

The following commands preview tracked regression inputs without writing
outputs:

```
uv run trim_padding.py tests\data\dark-transition-boundary.mkv --dry-run
uv run trim_padding.py tests\data\low-entropy-black-ending.mkv --dry-run
uv run trim_padding.py tests\data\leading-black-outlier.mkv --dry-run --min-removable-streak-seconds 8
uv run trim_padding.py tests\data\blue-black-opening-content-streak.mkv --dry-run
uv run trim_padding.py tests\data\trailing-keyframe-overlap.mkv --dry-run
```

The tracked `tests/data/dark-transition-boundary.mkv` regression clip is a
video-only, CRF 36 recompression of 30:50–32:50 from a customer recording. It
contains blue-toned footage before the true black screen. The detector must
preserve content through about 24.64 s and snap the kept end to the 25 s
keyframe; the earlier permissive blue threshold incorrectly cut around 14.8 s.

The tracked `tests/data/low-entropy-black-ending.mkv` clip is a video-only,
CRF 36 recompression of 2:25:00–2:26:10 from a second recording. It contains
dim content followed by near-black frames with compressed-video residual
entropy. The regression test verifies that both low entropy and very low mean
brightness are required, so dim content is retained while the true black tail
is removed.

The tracked `tests/data/leading-black-outlier.mkv` clip is a video-only, CRF 36
recompression of the first 30 seconds of another recording. Its first frame is
a single flat gray outlier, followed by a sustained black lead and then content.
The regression uses an 8-second confirmation interval because the clip ends
before it contains a full 10-second informative streak.

The tracked `tests/data/blue-black-opening-content-streak.mkv` clip is a
video-only, CRF 36 recompression of the first 75 seconds of a recording. It
contains brief content-classified interruptions within an opening blue/black
lead, followed by a sustained informative streak. The regression verifies that
the isolated interruptions do not prematurely end the leading trim.

The tracked `tests/data/trailing-keyframe-overlap.mkv` clip is a compact
recompression of the reported ending transition. Its regression generates both
the kept output and the `_end.mkv` inspection clip, then verifies by decoded
frame hashes that pre-cut frames in the inspection clip are also kept, while
end-only frames after the cut are BLACK.

### Reproducing a trailing-cut comparison

Run this PowerShell sequence from the repository root to create the sample
outputs and inspect them:

```
New-Item -ItemType Directory -Force tests\artifacts | Out-Null
Copy-Item -Force tests\data\trailing-keyframe-overlap.mkv tests\artifacts\trailing-keyframe-overlap.mkv
uv run trim_padding.py tests\artifacts\trailing-keyframe-overlap.mkv --overwrite
uv run inspect_trim_boundary.py tests\artifacts\trailing-keyframe-overlap.mkv
```

The inspector reruns the detector by default and reads the matching
`*_trimmed.mkv` and `*_end.mkv` outputs. To inspect a specific boundary or
non-default outputs, supply the source-stream cut time and paths. For the
fixture above:

```
uv run inspect_trim_boundary.py tests\artifacts\trailing-keyframe-overlap.mkv --cut-time 9.000 --kept-output tests\artifacts\trailing-keyframe-overlap_trimmed.mkv --end-clip tests\artifacts\trailing-keyframe-overlap_end.mkv
```

Use `--window-seconds` to control how far on either side of the boundary to
compare. The report hashes each decoded BGR24 frame with SHA-256 and matches
source frames to each output by their full pixel hashes; it does not infer
identity from timestamps, which can be shifted by stream-copy seeking. The
table marks whether each source frame occurs in the kept output and/or the
ending inspection clip, along with its signal classification, mean brightness,
entropy, and hash prefix. Shared frames before the boundary are expected
keyframe-seek overlap. The final summary counts only end-clip frames at or
after the cut that have no matching frame in the kept output; those are the
frames actually removed there. BLACK counts in that group confirm black frames
were removed, rather than merely appearing in the inspection clip.

## Changes to `save_frame_from_file.py`

- `find_first_dissimilar_frame(..., with_time=False)`: with `True` returns
  `(frame_no, image, time_s)` / `(None, None, None)`. Default return unchanged.
- `find_last_dissimilar_frame(...)`: bisection by frame index (constant frame rate), with an
  `on_probe(index, distance)` callback and helper `_load_reference_bgr`.
- `find_last_dissimilar_frame_time(...)`: bisection by presentation time (no frame rate
  assumption), `on_probe(time_s, distance)` callback, and an optional
  time-aware distance callback for window-based checks. Returns `(None, None)`
  when there is no content after the known frame.
- `calculate_blue_screen_distance`: distance = 1 - blue fraction (HSV), default threshold 0.1.

Run the tests for these frame-matching APIs with:

```
uv run pytest tests\test_reference_frame_matching.py -q
```

## Automated regression tests

The focused suite is `tests/test_reference_frame_matching.py`:

```
uv run pytest tests\test_reference_frame_matching.py -q
```

It covers the frame classifiers, blue matching, boundary detection, reporting,
and the tracked media regressions described above. In particular,
`test_leading_boundary_waits_for_sustained_informative_content` verifies that
brief content-classified interruptions do not stop a leading trim before a
sustained informative streak is confirmed.

Run only the end-overlap regression with:

```
uv run pytest tests\test_reference_frame_matching.py -q -k trailing_inspection_clip_overlaps_kept_content_but_not_removed_black
```

Tracked regression inputs belong in `tests/data/`; temporary or local-only
artifacts belong in the git-ignored `tests/artifacts/`.

## Inspecting packets and input-seek behavior

The commands below use the kept output and fixture copy created in the
preceding example.

To independently inspect the kept output's packet timestamps and decode it:

```
ffprobe -v error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=p=0 tests\artifacts\trailing-keyframe-overlap_trimmed.mkv
ffmpeg -v error -i tests\artifacts\trailing-keyframe-overlap_trimmed.mkv -f null -
```

To observe how input-side stream-copy seeking lands on keyframes around the
fixture's 9-second cut, run:

```
foreach($s in '8.40','8.52','8.96','9.00','9.04'){
  ffmpeg -hide_banner -loglevel error -y -ss $s -i tests\artifacts\trailing-keyframe-overlap.mkv -map 0:v -c copy -copyts -frames:v 5 tests\artifacts\d.mkv
  ffprobe -v error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=p=0 tests\artifacts\d.mkv | Select-Object -First 2
}
```

List the fixture's keyframe times with:

```
ffprobe -v error -select_streams v:0 -show_entries packet=pts_time,dts_time,flags -of csv=p=0 tests\data\trailing-keyframe-overlap.mkv | Select-String K
```

## Environment notes

- Run Python commands through `uv run` (not bare `python`). If `uv run pytest` fails with a "trampoline" error, the venv's
  launcher stubs point to an old project location, re-sync the environment with:

  ```
  uv sync
  ```

- Test videos that are meant to be portable are tracked in `tests/data/`.
