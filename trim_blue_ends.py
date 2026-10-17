"""Remove leading and trailing blue-screen or static-noise segments from a video.

Usage:
    uv run trim_blue_ends.py input.mp4
    uv run trim_blue_ends.py input.mp4 -o trimmed.mkv

Writes ``{input}_trimmed.mkv`` next to the input. Removed beginning and ending
areas are also saved as ``{input}_begin.mkv`` and ``{input}_end.mkv`` when
present. All streams are kept (``-map 0``) and copied without re-encoding
(fast).

Cuts are snapped OUTWARD to keyframes so no content frame is ever lost; some
blue frames may be kept instead:
  * start: the last keyframe at or before the first non-blue frame;
  * end:   the first keyframe after the last non-blue frame, i.e. the whole
           GOP containing the blue transition is kept (or end of file).

Frames classified as blue or high-entropy static are treated as removable.
The first content frame is found by scanning forward, and the last by bisection
with random-access seeks, so the (usually long) middle of the file is not
decoded. The trailing search assumes the removable tail is a single final run.

Possible future option: "smart cut" (NOT implemented)
-----------------------------------------------------
Get an exact start frame while still copying almost everything:
  1. Find K, the first keyframe at or after the desired start frame.
  2. Re-encode only [start frame, K) -- at most one GOP -- with x264 settings
     mirroring the source (codec, profile, level, resolution, pix_fmt, frame
     rate, timebase) so the pieces are splice-compatible.
  3. Stream-copy [K, end of kept range) for video, and cut the other streams
     with ``-c copy``.
  4. Join both pieces with the concat demuxer (``-c copy``).
Caveats: the re-encoded head must match the source stream exactly or the seam
glitches; open GOPs / B-frames right after K may reference frames before K,
so cut at a closed-GOP keyframe or re-encode slightly further; if the start
already is a keyframe, no re-encode is needed. The end is already handled by
the outward keyframe snapping above, so only the start would need this.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

DEFAULT_BLUE_THRESHOLD = 0.6
DEFAULT_NOISE_ENTROPY_THRESHOLD = 4.5
DEFAULT_BLACK_ENTROPY_THRESHOLD = 1.5
DEFAULT_MIN_REMOVABLE_STREAK_SECONDS = 10.0
MIN_REMOVABLE_WINDOW_FRACTION = 0.9
REMOVABLE_WINDOW_SAMPLE_INTERVAL_SECONDS = 0.5
LEADING_BLUE_CONFIRMATION_SECONDS = 1.0


def calculate_removable_signal_distance(
    frame_bgr: np.ndarray,
    _reference_bgr: np.ndarray | None = None,
    noise_entropy_threshold: float = DEFAULT_NOISE_ENTROPY_THRESHOLD,
    black_entropy_threshold: float = DEFAULT_BLACK_ENTROPY_THRESHOLD,
) -> float:
    """Return a blue-distance value, treating static and very-low-entropy frames as removable."""
    from save_frame_from_file import calculate_blue_screen_distance, calculate_local_entropy

    entropy = calculate_local_entropy(frame_bgr)
    if entropy >= noise_entropy_threshold or entropy <= black_entropy_threshold:
        return 0.0
    return calculate_blue_screen_distance(frame_bgr)


def classify_signal_types(
    frame_bgr: np.ndarray,
    blue_threshold: float,
    noise_entropy_threshold: float,
    black_entropy_threshold: float,
) -> set[str]:
    from save_frame_from_file import calculate_blue_screen_distance, calculate_local_entropy

    kinds = set()
    is_blue = calculate_blue_screen_distance(frame_bgr) <= blue_threshold
    entropy = calculate_local_entropy(frame_bgr)
    if is_blue:
        kinds.add("BLUE")
    elif entropy <= black_entropy_threshold:
        kinds.add("BLACK")
    if entropy >= noise_entropy_threshold:
        kinds.add("STATIC")
    return kinds


def calculate_removable_window_distance(
    video: Path,
    start_s: float,
    end_s: float,
    blue_threshold: float,
    noise_entropy_threshold: float,
    black_entropy_threshold: float,
    *,
    blue_only: bool = False,
) -> float:
    """Return the non-removable fraction in a stream-relative time window."""
    if end_s <= start_s:
        return 1.0

    import av
    if blue_only:
        from save_frame_from_file import calculate_blue_screen_distance

    total_samples = 0
    removable_samples = 0
    first_frame_time = None
    last_frame_time = None
    next_sample_time = start_s
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        origin = (stream.start_time or 0) * time_base
        frame_rate = float(stream.average_rate)
        sample_interval = max(
            REMOVABLE_WINDOW_SAMPLE_INTERVAL_SECONDS, 1.0 / frame_rate
        )
        frame_tolerance = max(1.5 / frame_rate, sample_interval)
        if start_s > 0.0:
            target_pts = round((start_s + origin) / time_base)
            container.seek(target_pts, backward=True, any_frame=False, stream=stream)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_time = frame.pts * time_base - origin
            if frame_time < start_s:
                continue
            if frame_time >= end_s:
                break
            if first_frame_time is None:
                first_frame_time = frame_time
            last_frame_time = frame_time
            if frame_time >= next_sample_time:
                image = frame.to_ndarray(format="bgr24")
                if blue_only:
                    distance = calculate_blue_screen_distance(image)
                else:
                    distance = calculate_removable_signal_distance(
                        image,
                        noise_entropy_threshold=noise_entropy_threshold,
                        black_entropy_threshold=black_entropy_threshold,
                    )
                total_samples += 1
                if distance <= blue_threshold:
                    removable_samples += 1
                while next_sample_time <= frame_time:
                    next_sample_time += sample_interval

    expected_samples = int(
        np.ceil((end_s - start_s) / sample_interval)
    )
    if (
        total_samples == 0
        or total_samples < expected_samples * 0.8
        or first_frame_time is None
        or last_frame_time is None
        or first_frame_time > start_s + frame_tolerance
        or last_frame_time < end_s - frame_tolerance
    ):
        return 1.0
    return 1.0 - removable_samples / total_samples


def find_content_range(
    video_file: Path,
    threshold: float,
    noise_entropy_threshold: float = DEFAULT_NOISE_ENTROPY_THRESHOLD,
    black_entropy_threshold: float = DEFAULT_BLACK_ENTROPY_THRESHOLD,
    minimum_streak_seconds: float = DEFAULT_MIN_REMOVABLE_STREAK_SECONDS,
) -> tuple[int, float, float]:
    """Return the leading content frame/time and conservative trailing boundary.

    Times are the frames' real presentation times relative to the start of the
    video stream, so no constant frame rate is assumed anywhere. With a
    nonzero minimum streak, each trailing probe is validated against a
    qualifying look-forward window.
    """
    import av
    from PIL import Image
    from tqdm import tqdm

    from save_frame_from_file import find_first_dissimilar_frame, find_last_dissimilar_frame_time

    with av.open(str(video_file)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        origin = (stream.start_time or 0) * time_base
        duration_s = float(container.duration) / av.time_base
        window_sample_interval = max(
            REMOVABLE_WINDOW_SAMPLE_INTERVAL_SECONDS,
            1.0 / float(stream.average_rate),
        )

    reference = Image.new("RGB", (1, 1), color="black")  # unused by the blue distance

    def distance_method(frame_bgr: np.ndarray, reference_bgr: np.ndarray) -> float:
        return calculate_removable_signal_distance(
            frame_bgr,
            reference_bgr,
            noise_entropy_threshold,
            black_entropy_threshold,
        )

    first_no, _, first_time = find_first_dissimilar_frame(
        video_file,
        reference,
        min_frame=-1,
        with_time=True,
        mse_threshold=threshold,
        distance_method=distance_method,
    )
    if first_no is None or first_time is None:
        raise SystemExit("No content frame found; the video appears entirely blue or static.")

    with tqdm(desc="Bisection search for trailing removable streak", unit=" probes") as bar:
        def on_probe(time_s: float, distance: float) -> None:
            bar.update(1)
            bar.set_postfix(t=f"{time_s:.3f}s", distance=f"{distance:.3f}")

        def sustained_signal_distance(time_s: float, frame_bgr: np.ndarray) -> float:
            frame_distance = calculate_removable_signal_distance(
                frame_bgr,
                noise_entropy_threshold=noise_entropy_threshold,
                black_entropy_threshold=black_entropy_threshold,
            )
            if frame_distance > threshold:
                return 1.0
            if minimum_streak_seconds == 0.0:
                return frame_distance
            window_start = time_s - origin
            window_end = window_start + minimum_streak_seconds
            if window_end > duration_s + window_sample_interval:
                return 1.0
            window_distance = calculate_removable_window_distance(
                video_file,
                window_start,
                window_end,
                threshold,
                noise_entropy_threshold,
                black_entropy_threshold,
            )
            if window_distance > 1.0 - MIN_REMOVABLE_WINDOW_FRACTION:
                return 1.0
            return window_distance

        known_similar_time = None
        if minimum_streak_seconds > 0.0:
            known_similar_time = origin + duration_s - minimum_streak_seconds
            if known_similar_time <= first_time:
                return first_no, first_time - origin, duration_s

        last_time, _ = find_last_dissimilar_frame_time(
            video_file,
            reference,
            known_dissimilar_time=first_time,
            on_probe=on_probe,
            mse_threshold=(
                threshold
                if minimum_streak_seconds == 0.0
                else 1.0 - MIN_REMOVABLE_WINDOW_FRACTION
            ),
            distance_method=distance_method,
            distance_method_at_time=(
                None if minimum_streak_seconds == 0.0 else sustained_signal_distance
            ),
            known_similar_time=known_similar_time,
        )
    if last_time is None:
        if minimum_streak_seconds > 0.0:
            return first_no, first_time - origin, duration_s
        raise SystemExit("Could not locate the end of the content.")
    return first_no, first_time - origin, last_time - origin


def find_keyframe_bounds(
    video_file: Path, first_time: float, last_time: float
) -> tuple[float, float | None]:
    """Return (start_s, end_s) snapped outward to keyframes.

    start_s: time of the last keyframe at or before ``first_time``.
    end_s:   time of the first keyframe after ``last_time``, or None if there
             is none (keep until end of file).
    Times are relative to the start of the video stream.
    """
    import av

    with av.open(str(video_file)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        origin = (stream.start_time or 0) * time_base
        eps = 0.25 / float(stream.average_rate)

        def keyframe_times_from(target_s: float):
            container.seek(
                round((target_s + origin) / time_base),
                backward=True, any_frame=False, stream=stream,
            )
            for packet in container.demux(stream):
                if packet.pts is not None and packet.is_keyframe:
                    yield packet.pts * time_base - origin

        start_s = max(0.0, first_time)
        for key_s in keyframe_times_from(start_s):
            start_s = min(start_s, key_s)
            break

        end_s = None
        for key_s in keyframe_times_from(last_time):
            if key_s > last_time + eps:
                end_s = key_s
                break
    return start_s, end_s

def build_ffmpeg_command(
    video_file: Path,
    output_file: Path,
    seek_s: float,
    start_s: float,
    end_s: float | None,
    overwrite: bool,
) -> list[str]:
    # seek_s is the time of the first content frame: it lies inside the GOP
    # that starts at start_s, so the input seek lands on that keyframe. (Seeking
    # at start_s itself was observed to land on an earlier keyframe.) -t counts from seek_s and limits
    # by packet DTS, which lags PTS, so the end keyframe and a few frames after
    # it may be included: frames can only be added, never lost.
    duration_args = [] if end_s is None else ["-t", f"{end_s - seek_s:.6f}"]
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-stats",
        "-y" if overwrite else "-n",
        "-ss", f"{seek_s:.6f}",
        *duration_args,
        "-i", str(video_file),
        "-map", "0", "-map_metadata", "0", "-map_chapters", "0",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(output_file),
    ]


def _run_ffmpeg(command: list[str]) -> int:
    print("\n\n", flush=True)
    try:
        return subprocess.run(command).returncode
    finally:
        print("\n\n", flush=True)


def build_inspection_clip_commands(
    video_file: Path,
    beginning_output: Path,
    ending_output: Path,
    beginning_end_s: float,
    end_s: float | None,
    duration_s: float,
    overwrite: bool,
) -> list[tuple[Path, list[str]]]:
    commands = []
    if beginning_end_s > 0.005:
        commands.append(
            (
                beginning_output,
                build_ffmpeg_command(
                    video_file,
                    beginning_output,
                    0.0,
                    0.0,
                    beginning_end_s,
                    overwrite,
                ),
            )
        )
    if end_s is not None and duration_s - end_s > 0.005:
        commands.append(
            (
                ending_output,
                build_ffmpeg_command(
                    video_file,
                    ending_output,
                    end_s,
                    end_s,
                    None,
                    overwrite,
                ),
            )
        )
    return commands


def video_pts_span(video_file: Path) -> tuple[float, float]:
    """(first, last) video presentation time in seconds, from packets only."""
    import av

    with av.open(str(video_file)) as container:
        stream = container.streams.video[0]
        times = [
            p.pts * float(stream.time_base)
            for p in container.demux(stream)
            if p.pts is not None
        ]
    return min(times), max(times)

def format_hms(seconds: float) -> str:
    total = round(seconds * 100) / 100
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:05.2f}"


def scan_signal_types(
    video: Path,
    start_s: float,
    end_s: float | None,
    blue_threshold: float,
    noise_entropy_threshold: float,
    black_entropy_threshold: float,
    description: str,
) -> set[str]:
    """Classify every decoded video frame in a stream-relative interval."""
    import av
    from tqdm import tqdm

    kinds: set[str] = set()
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        origin = (stream.start_time or 0) * time_base
        if start_s > 0.0:
            target_pts = round((start_s + origin) / time_base)
            container.seek(target_pts, backward=True, any_frame=False, stream=stream)
        with tqdm(desc=description, unit=" frames") as progress:
            for frame in container.decode(stream):
                progress.update(1)
                if frame.pts is None:
                    continue
                frame_time = frame.pts * time_base - origin
                if frame_time < start_s:
                    continue
                if end_s is not None and frame_time >= end_s:
                    break
                kinds.update(
                    classify_signal_types(
                        frame.to_ndarray(format="bgr24"),
                        blue_threshold,
                        noise_entropy_threshold,
                        black_entropy_threshold,
                    )
                )
    return kinds


def classify_removed_areas(
    video: Path,
    start_s: float,
    end_s: float | None,
    blue_threshold: float,
    noise_entropy_threshold: float,
    black_entropy_threshold: float = DEFAULT_BLACK_ENTROPY_THRESHOLD,
) -> tuple[set[str], set[str]]:
    """Return the signal types present in the full leading and trailing cut areas."""
    import av

    with av.open(str(video)) as container:
        total = float(container.duration) / av.time_base

    beginning = (
        scan_signal_types(
            video, 0.0, start_s, blue_threshold, noise_entropy_threshold,
            black_entropy_threshold,
            "Classifying beginning interval",
        )
        if start_s > 0.005
        else set()
    )
    ending = (
        scan_signal_types(
            video, end_s, None, blue_threshold, noise_entropy_threshold,
            black_entropy_threshold,
            "Classifying ending interval",
        )
        if end_s is not None and total - end_s > 0.005
        else set()
    )
    return beginning, ending


def format_signal_types(kinds: set[str]) -> str:
    ordered = [kind for kind in ("BLUE", "BLACK", "STATIC") if kind in kinds]
    if not ordered:
        return "unclassified"
    if len(ordered) == 1:
        return ordered[0]
    if len(ordered) == 2:
        return f"{ordered[0]} and {ordered[1]} (both)"
    return ", ".join(ordered[:-1]) + f", and {ordered[-1]}"


def describe_removed_areas(
    video: Path,
    start_s: float,
    end_s: float | None,
    beginning_types: set[str] | None,
    ending_types: set[str] | None,
) -> str:
    """Describe the keyframe-aligned leading/trailing intervals the cut drops."""
    import av

    areas = []
    if start_s > 0.005:
        label = (
            f"{format_signal_types(beginning_types)} at beginning"
            if beginning_types is not None
            else "beginning interval"
        )
        areas.append(
            f"{label} from {format_hms(0)} to {format_hms(start_s)}"
        )
    if end_s is not None:
        with av.open(str(video)) as container:
            total = float(container.duration) / av.time_base
        if total - end_s > 0.005:
            label = (
                f"{format_signal_types(ending_types)} at end"
                if ending_types is not None
                else "ending interval"
            )
            areas.append(
                f"{label} from {format_hms(end_s)} to {format_hms(total)}"
            )
    if not areas:
        return "No removable areas found."
    return "Found removable areas: " + "; ".join(areas)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Output file (default: {input}_trimmed.mkv next to the input)",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_BLUE_THRESHOLD,
        help="Fraction of non-blue pixels above which a frame counts as content "
        f"(default {DEFAULT_BLUE_THRESHOLD})",
    )
    parser.add_argument(
        "--noise-entropy-threshold",
        type=float,
        default=DEFAULT_NOISE_ENTROPY_THRESHOLD,
        help="Local-entropy cutoff for classifying a frame as static noise "
        f"(default {DEFAULT_NOISE_ENTROPY_THRESHOLD}; lower values are more sensitive)",
    )
    parser.add_argument(
        "--black-entropy-threshold",
        type=float,
        default=DEFAULT_BLACK_ENTROPY_THRESHOLD,
        help="Local-entropy cutoff at or below which a frame is treated as black "
        f"(default {DEFAULT_BLACK_ENTROPY_THRESHOLD}; temporary proxy heuristic)",
    )
    parser.add_argument(
        "--min-removable-streak-seconds",
        "--min-black-streak-seconds",
        dest="min_removable_streak_seconds",
        type=float,
        default=DEFAULT_MIN_REMOVABLE_STREAK_SECONDS,
        help="Minimum edge streak duration; requires at least 90 percent matching "
        f"samples in windows of this duration (default "
        f"{DEFAULT_MIN_REMOVABLE_STREAK_SECONDS} seconds). The legacy "
        "--min-black-streak-seconds name is also accepted",
    )
    parser.add_argument(
        "--report-signal-types",
        action="store_true",
        help="Classify every frame in the removed intervals to report BLUE/BLACK/STATIC "
        "(slower)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite all output files"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only report the detected ranges and ffmpeg commands",
    )
    args = parser.parse_args(argv)
    if args.black_entropy_threshold >= args.noise_entropy_threshold:
        parser.error("--black-entropy-threshold must be lower than --noise-entropy-threshold")
    if args.min_removable_streak_seconds < 0.0:
        parser.error("--min-removable-streak-seconds must be non-negative")

    video: Path = args.video
    if not video.is_file():
        parser.error(f"File not found: {video}")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg was not found on PATH")
    output = args.output or video.with_name(f"{video.stem}_trimmed.mkv")
    beginning_output = video.with_name(f"{video.stem}_begin.mkv")
    ending_output = video.with_name(f"{video.stem}_end.mkv")

    first_no, first_time, last_time = find_content_range(
        video,
        args.threshold,
        args.noise_entropy_threshold,
        args.black_entropy_threshold,
        args.min_removable_streak_seconds,
    )
    start_s, end_s = find_keyframe_bounds(video, first_time, last_time)
    import av

    with av.open(str(video)) as container:
        duration_s = float(container.duration) / av.time_base
    seek_s = first_time
    if args.min_removable_streak_seconds > 0.0:
        leading_edge_is_confirmed = False
        if first_time >= args.min_removable_streak_seconds:
            if duration_s >= args.min_removable_streak_seconds:
                leading_window_distance = calculate_removable_window_distance(
                    video,
                    0.0,
                    args.min_removable_streak_seconds,
                    args.threshold,
                    args.noise_entropy_threshold,
                    args.black_entropy_threshold,
                )
                leading_edge_is_confirmed = (
                    leading_window_distance
                    <= 1.0 - MIN_REMOVABLE_WINDOW_FRACTION
                )
        elif first_time > 0.0:
            blue_confirmation_duration = min(
                first_time, LEADING_BLUE_CONFIRMATION_SECONDS
            )
            blue_window_distance = calculate_removable_window_distance(
                video,
                0.0,
                blue_confirmation_duration,
                args.threshold,
                args.noise_entropy_threshold,
                args.black_entropy_threshold,
                blue_only=True,
            )
            leading_edge_is_confirmed = (
                blue_window_distance <= 1.0 - MIN_REMOVABLE_WINDOW_FRACTION
            )
        if not leading_edge_is_confirmed:
            start_s = 0.0
            seek_s = 0.0

    print(
        f"Content starts at frame #{first_no} ({first_time:.3f}s); "
        f"trailing search boundary at {last_time:.3f}s\n"
        f"Keep range: {seek_s:.3f}s .. "
        f"{'end of file' if end_s is None else f'{end_s:.3f}s'}"
    )

    if args.report_signal_types:
        beginning_types, ending_types = classify_removed_areas(
            video,
            seek_s,
            end_s,
            args.threshold,
            args.noise_entropy_threshold,
            args.black_entropy_threshold,
        )
    else:
        beginning_types = ending_types = None

    report = describe_removed_areas(
        video, seek_s, end_s, beginning_types, ending_types
    )
    print(report)
    print("\n", flush=True)

    command = build_ffmpeg_command(
        video, output, seek_s, start_s, end_s, args.overwrite
    )
    inspection_commands = build_inspection_clip_commands(
        video,
        beginning_output,
        ending_output,
        seek_s,
        end_s,
        duration_s,
        args.overwrite,
    )
    inspection_paths = {path.resolve() for path, _ in inspection_commands}
    if output.resolve() in inspection_paths:
        parser.error("--output must not use a *_begin.mkv or *_end.mkv inspection filename")

    print(" ".join(command))
    for inspection_path, inspection_command in inspection_commands:
        print(f"Inspection clip {inspection_path}:")
        print(" ".join(inspection_command))
    if args.dry_run:
        return 0

    result = _run_ffmpeg(command)
    try:
        if result != 0:
            return result
        out_first, out_last = video_pts_span(output)
        # Output times are shifted to start near 0; the span must cover the content.
        if out_last - out_first < last_time - first_time - 1e-3:
            print(f"ERROR: output video spans {out_last - out_first:.3f}s, shorter than "
                  f"the content ({last_time - first_time:.3f}s); cut is wrong.", file=sys.stderr)
            return 1
        print(f"Wrote {output} (video spans {out_last - out_first:.3f}s)")
        for inspection_path, inspection_command in inspection_commands:
            inspection_result = _run_ffmpeg(inspection_command)
            if inspection_result != 0:
                print(
                    f"ERROR: failed to write inspection clip {inspection_path}",
                    file=sys.stderr,
                )
                return inspection_result
            print(f"Wrote inspection clip {inspection_path}")
        return 0
    finally:
        print(report)

if __name__ == "__main__":
    sys.exit(main())
