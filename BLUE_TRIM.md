# Blue-screen and static-noise detection and trimming

Tooling to find leading and trailing blue "no signal" or high-entropy static
("snow") segments and cut them off a video without re-encoding.

## `trim_blue_ends.py` (new script)

```
uv run trim_blue_ends.py input.mp4                 # writes input_trimmed.mkv
uv run trim_blue_ends.py input.mp4 -o out.mkv --overwrite
uv run trim_blue_ends.py input.mp4 --dry-run       # report ranges + ffmpeg commands only
uv run trim_blue_ends.py input.mp4 --threshold 0.6  # blue threshold
uv run trim_blue_ends.py input.mp4 --noise-entropy-threshold 4.5
uv run trim_blue_ends.py input.mp4 --black-entropy-threshold 1.5
uv run trim_blue_ends.py input.mp4 --min-removable-streak-seconds 10
uv run trim_blue_ends.py input.mp4 --report-signal-types --dry-run
```

By default, the kept video is written next to the input as
`{input}_trimmed.mkv`. When a beginning or ending area is removed, that area is
also saved separately as `{input}_begin.mkv` or `{input}_end.mkv`, respectively,
for inspection. These inspection files use the input's filename stem even when
`-o` selects a different path for the trimmed video. `--overwrite` applies to
the trimmed output and any inspection clips. `--dry-run` prints every command
without creating any files.

### How blue and static are classified

- **Blue:** `calculate_blue_screen_distance` returns the fraction of pixels not
  recognized as blue by the HSV mask. The default `--threshold 0.6` means a
  frame with distance at or below 0.6 is treated as blue/removable.
- **Static/noise:** `calculate_local_entropy` measures entropy in the
  high-frequency residual after Gaussian blur. A frame with local entropy at or
  above `--noise-entropy-threshold` is treated as static/removable. The default
  is `4.5`, tuned to detect the sustained static in
  `tests/artifacts/vid_noise_at_end.mkv`. The older `is_proper_frame_params`
  heuristic uses `5.0`; the trimming threshold is intentionally more sensitive.
  Raise the cutoff to be more selective; lower it to classify more frames as static.
- **Black-screen proxy:** a frame that is not classified as blue and whose local
  entropy is at or below `--black-entropy-threshold` is labeled `BLACK` and is
  removable. The default cutoff is `1.5`. This intentionally uses very low
  entropy as a temporary black-screen proxy; it does not inspect pixel
  brightness, so flat non-black images can also match.
- **Combined rule:** a frame is removable if either rule matches. The same
  per-frame rule applies to blue, black-proxy, and static signals. An edge is
  removed only after it passes the sustained-window check below.

### Sustained edge check and reporting

The default minimum is 10 seconds, configured with
`--min-removable-streak-seconds` (the older `--min-black-streak-seconds` spelling
is still accepted). It applies to all removable signals, including STATIC.

The trailing bisection checks each candidate against the 10 seconds
immediately after it. It samples at 0.5-second intervals (or at the native
frame interval for sources below 2 fps); at least 90% of those samples must
match a removable signal, and the candidate frame itself must also match. This
confirms a sustained run while allowing the cut to begin near its onset,
instead of waiting for 10 seconds of signal to accumulate before the boundary.
The terminal 10-second window is separately validated to provide the bisection
with a confirmed removable upper bound. A leading edge likewise must last at
least 10 seconds, with at least 90% of samples in the initial window matching.
Exception: if the leading interval is shorter than 10 seconds, the script
checks a blue-only window of up to one second from the start. At least 90% of
its samples must be blue to confirm the edge; when confirmed, the leading
interval is removed even though it is shorter than 10 seconds. A short
static-only or black-only opening still needs to meet the 10-second minimum.

By default the script reports the time ranges to remove without labeling their
signal types. This avoids decoding every frame in a long removed tail just to
produce a label. Add `--report-signal-types` to scan the full removed intervals
and report BLUE, BLACK, STATIC, or combinations; this optional report can take
substantial time on long tails and does not validate or change the bisection
boundary.

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
frame as high-entropy static when `H >= 4.5` by default. The temporary
black-screen proxy marks non-blue frames with `H <= 1.5` as `BLACK`.

This is a useful signal, not a semantic guarantee that a frame is snow.
Fine texture, film grain, detailed foliage, text, or compression artifacts can
also raise local entropy. Conversely, heavily blurred, denoised, or low-bitrate
static may score below the cutoff. Blue, high-entropy static, and the low-entropy black proxy are classified
per-frame. A signal must also pass the 10-second sustained-window rule before
an edge is removed. Adjust `--min-removable-streak-seconds` for sources where
edge signals typically last longer or shorter.

The supplied models illustrate why the cutoff is source-dependent:

- In `vid_with_noise.mkv`, inspected static sections score roughly 5.2–7.1.
- In `vid_noise_at_end.mkv`, the sustained tail is mostly around 4.7–5.0,
  below the previous 5.0 cutoff. The 10-second forward check confirms the
  trailing run while allowing the cut to begin near its onset.

These values depend on the clip, resolution, encoding, and decoder; they are
not universal calibration values.

#### Tuning the current cutoff

The only static-classifier setting currently exposed by `trim_blue_ends.py` is
`--noise-entropy-threshold`. To tune it for a new source:

1. Decode representative frames known to be snow/static and representative
   frames of real content, including the most detailed content.
2. Measure their scores with `calculate_local_entropy(frame_bgr)` from
   `save_frame_from_file.py`.
3. Choose a cutoff that separates those two sets with some margin. If the
   score ranges overlap, a single entropy cutoff cannot reliably distinguish
   them; a higher cutoff favors avoiding false removals, while a lower cutoff
   catches weaker static at the cost of more false positives.
4. Run `trim_blue_ends.py ... --dry-run` and review the printed removable
   intervals before running a cut. For example:

```
uv run trim_blue_ends.py input.mkv --noise-entropy-threshold 5.5 --dry-run
```

#### Other metrics and possible variants

The repository has other noise-related metrics, but `trim_blue_ends.py` does
not currently use them or expose them as alternatives:

| Metric/variant | What it could add | Main caveat |
|---|---|---|
| **Temporal entropy** (`calculate_temporal_entropy`) | Entropy of the pixel-wise difference between adjacent frames; flickering snow may score high. | Motion, scene cuts, and animation also produce large differences. |
| **Combined local + temporal entropy** (`is_proper_frame`) | Uses local and temporal entropy together; the existing helper also checks mean brightness. | Its thresholds and “proper frame” semantics are not wired into this cutter; a black-frame rule would be a separate behavior change. |
| **Laplacian variance** (`calculate_laplacian_variance`) | Measures fine edges/high-frequency detail; static can cause a strong rise. | Sharp real content can also have high values, so absolute cutoffs are source-dependent. |
| **FFT high-band ratio** (`calculate_fft_highband_ratio`) | Measures the proportion of image energy in high spatial frequencies. | More computation and parameters; high-frequency real texture may resemble static. |
| **Persistence / hysteresis** | Require several consecutive frames, or a smoothed score, before declaring a streak. This can suppress isolated noisy frames. | It needs state across frames and a boundary search that respects the consecutive-frame rule; it is not implemented. |
| **Adaptive/source-calibrated threshold** | Estimate a cutoff from labeled static and content samples, or normalize scores against a known content baseline. | Requires representative samples and must handle sources where the score distributions overlap. |

`locate_noise.py` and `streaming_regime.py` are exploratory regime-change
tools: they analyze a stream of metrics such as Laplacian variance and detect
upward jump-ratio crossings. They are not currently used by this script as a
per-frame static classifier.

How it works:

1. **Find the first content frame**: forward scan with `find_first_dissimilar_frame(..., with_time=True)`.
   A frame is content only if it is neither sufficiently blue nor above the
   static-noise cutoff. Returns the frame number and its real presentation time.
2. **Find the last content frame**: `find_last_dissimilar_frame_time` bisects on presentation
   time using random-access seeks, so the long middle of the file is not decoded.
   It only searches after the first content frame, so it cannot find the start
   streak. If the last frame is content there is no trailing removable streak
   and it is returned after a single probe.
3. **Snap outward to keyframes** (`find_keyframe_bounds`, packets only, no decoding):
   start = last keyframe at or before the first content frame; end = first keyframe after the last
   content frame (or end of file). Content frames are never lost; extra blue or
   static frames may be kept.
4. **Cut with ffmpeg stream copy** (`build_ffmpeg_command`): `-map 0 -map_metadata 0 -map_chapters 0 -c copy`.
   - `-ss` is the *first content frame's time*, not the keyframe time: seeking at the keyframe
     time landed on an earlier keyframe on the sample file (Matroska); a time inside the GOP lands
     on the right keyframe.
   - `-t` is `end - seek_s` (ffmpeg measures it from the seek point). It is applied on packet DTS,
     so a couple of frames past the end keyframe can be included (extra blue only).
5. **Check**: the output's video span (`video_pts_span`) must cover the content span, else exit 1.

The trailing bisection assumes the tail is one final removable run and does not
fully scan the long interior. Multiple separated terminal/interior transitions
can violate the binary-search assumption. The noise cutoff can also classify
high-detail content as static, so review the reported removable areas (or use
`--dry-run`) when working with a new source. Variable frame rate is untested;
nominal frame rate is used only as a half-frame tolerance.

By default, before ffmpeg runs the script reports only each removable interval
and its time range. `--report-signal-types` opts into classifying every frame in
the removed intervals as `BLUE`, `BLACK`, `STATIC`, or combinations. In
particular, this extra report scan decodes a long ending streak after the
bisection, so it can take substantial time; it does not decode the kept middle.
The same report is printed again after ffmpeg and output validation finish, so
it remains visible after ffmpeg's output.

The module docstring also documents a possible future "smart cut" (re-encode
only the head GOP, copy the rest, concat) that is NOT implemented.

### Static-noise model clip

The provided `tests/artifacts/vid_with_noise.mkv` contains a leading blue
streak, short static bursts inside the footage, and a short static tail. With
the default thresholds and 10-second minimum:

```
uv run trim_blue_ends.py tests\artifacts\vid_with_noise.mkv --dry-run
```

finds the first content frame at #141 / 5.64 s, but keeps from the beginning
because the leading streak is shorter than 10 seconds. The trailing signal also
does not pass the sustained-window check, so the clip is kept through EOF.
Short internal static bursts are not treated as removable edges.

The additional `tests/artifacts/vid_noise_at_end.mkv` model is a 53-second clip
with no leading blue/noise and a sustained static ending. With the default
entropy cutoff of 4.5, the dry run finds a trailing boundary at about 26.32 s.
Each candidate is checked against the following 10 seconds, so the cut stays
near the onset once a sustained run is confirmed.

## Changes to `save_frame_from_file.py`

- `find_first_dissimilar_frame(..., with_time=False)`: with `True` returns
  `(frame_no, image, time_s)` / `(None, None, None)`. Default return unchanged.
- `find_last_dissimilar_frame(...)`: bisection by frame index (constant frame rate), with an
  `on_probe(index, distance)` callback and helper `_load_reference_bgr`.
- `find_last_dissimilar_frame_time(...)`: bisection by presentation time (no frame rate
  assumption), `on_probe(time_s, distance)` callback, and an optional
  time-aware distance callback for window-based checks. Returns `(None, None)`
  when there is no content after the known frame.
- `calculate_blue_screen_distance`: distance = 1 - blue fraction (HSV), threshold 0.6.

## Tests: `tests/test_reference_frame_matching.py`

Renamed from `test_find_first_dissimilar_frame.py` (it covers both first and last). The tests
include a real-video test on `tests/vid_with_blue_HD.mp4` (blue frames 0-92 and 2825-2986,
content 93-2824 at 25 fps), a classifier test for blue/static/content, optional model-video
boundary tests for `tests/artifacts/vid_with_noise.mkv` and
`tests/artifacts/vid_noise_at_end.mkv`, and the existing blue bisection test.
The bisection test has a tqdm bar and per-probe lines (run with `-s` to see them live). Artifacts
go to the git-ignored `tests/artifacts/`.

```
uv run pytest tests/test_reference_frame_matching.py -q
```

The focused run after adding static-noise handling and interval reporting passed all 10 tests:

```
uv run pytest tests\test_reference_frame_matching.py -q
```

The no-trailing-blue case is still checked manually, not covered by an automated
test (see below). Two tests in `tests/test_frame_metrics_pipeline.py` fail independently of this work.
Two tests in `tests/test_frame_metrics_pipeline.py` fail independently of this work.

## Temporary clips used for manual checks

All written to the git-ignored `tests/artifacts/` and deleted afterwards.

Clip without a trailing blue streak (first 100 s of the sample, stream copy):

```
ffmpeg -v error -y -t 100 -i tests\vid_with_blue_HD.mp4 -map 0 -c copy tests\artifacts\noend.mkv
uv run trim_blue_ends.py tests\artifacts\noend.mkv -o tests\artifacts\noend_out.mkv --overwrite
Remove-Item tests\artifacts\noend*.mkv
```

Expected: 1 bisection probe, "Keep ... .. end of file".

Inspecting an output (frame count, first/last video packet, decode errors):

```
ffprobe -v error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=p=0 tests\artifacts\k.mkv
ffmpeg -v error -i tests\artifacts\k.mkv -f null -
```

Expected for the sample clip: keep range 2.480s..113.600s, first packet a keyframe, no decode
errors. The output has 2780 video frames (2778 in range + 2 extra from `-t` on DTS).

Probe of how ffmpeg's input seek lands (this showed `-ss 2.40-2.60` landing on the 0.12 s
keyframe instead of 2.48 s):

```
foreach($s in '2.40','2.48','2.60','3.00','3.72'){
  ffmpeg -hide_banner -loglevel error -y -ss $s -i tests\vid_with_blue_HD.mp4 -map 0:v -c copy -copyts -frames:v 5 tests\artifacts\d.mkv
  ffprobe -v error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=p=0 tests\artifacts\d.mkv | Select-Object -First 2
}
```

Keyframe times of the source: `ffprobe -v error -select_streams v:0 -show_entries packet=pts_time,dts_time,flags -of csv=p=0 tests\vid_with_blue_HD.mp4 | Select-String K`

## Environment notes

- Use `uv run ...` (not bare `python`). If `uv run pytest` fails with a "trampoline" error, the venv's
  launcher stubs point to an old project location; re-sync the venv.
- `tests/vid_with_blue_HD.mp4` is actually a Matroska container despite the extension.
