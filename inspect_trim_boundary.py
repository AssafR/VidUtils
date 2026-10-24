"""Compare a trim's source, kept tail, and ending inspection clip by frame hash.

Usage:
    uv run inspect_trim_boundary.py input.mkv
    uv run inspect_trim_boundary.py input.mkv --cut-time 7186.984

By default, reruns the detector and reads ``{input}_trimmed.mkv`` and
``{input}_end.mkv``. Hashes are SHA-256 digests of decoded BGR24 frame pixels;
matching hashes identify the same decoded frame across stream-copied files.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from trim_padding import (
    DEFAULT_BLACK_BRIGHTNESS_THRESHOLD,
    DEFAULT_BLACK_ENTROPY_THRESHOLD,
    DEFAULT_BLUE_THRESHOLD,
    DEFAULT_NOISE_ENTROPY_THRESHOLD,
    find_content_range,
    find_keyframe_bounds,
    format_hms,
    is_black_frame,
)


@dataclass(frozen=True)
class FrameRecord:
    time_s: float
    digest: str
    mean_brightness: float
    entropy: float
    signal_types: tuple[str, ...]


@dataclass(frozen=True)
class BoundaryComparison:
    cut_time_s: float
    source_frames: tuple[FrameRecord, ...]
    kept_matches: tuple[tuple[int, int], ...]
    end_matches: tuple[tuple[int, int], ...]

    @property
    def kept_source_indices(self) -> set[int]:
        return {source_index for source_index, _ in self.kept_matches}

    @property
    def end_source_indices(self) -> set[int]:
        return {source_index for source_index, _ in self.end_matches}

    @property
    def shared_source_indices(self) -> set[int]:
        return self.kept_source_indices & self.end_source_indices


def _read_frame_records(
    video: Path, start_s: float, end_s: float
) -> tuple[FrameRecord, ...]:
    import av
    from save_frame_from_file import (
        calculate_blue_screen_distance,
        calculate_local_entropy,
    )

    records = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        origin = (stream.start_time or 0) * time_base
        if start_s > 0.0:
            container.seek(
                round((start_s + origin) / time_base),
                backward=True,
                any_frame=False,
                stream=stream,
            )
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_time = frame.pts * time_base - origin
            if frame_time < start_s:
                continue
            if frame_time >= end_s:
                break

            image = frame.to_ndarray(format="bgr24")
            entropy = calculate_local_entropy(image)
            mean_brightness = float(np.mean(image))
            blue_distance = calculate_blue_screen_distance(image)
            signal_types = []
            if blue_distance <= DEFAULT_BLUE_THRESHOLD:
                signal_types.append("BLUE")
            elif is_black_frame(
                image,
                entropy,
                DEFAULT_BLACK_ENTROPY_THRESHOLD,
                DEFAULT_BLACK_BRIGHTNESS_THRESHOLD,
            ):
                signal_types.append("BLACK")
            if entropy >= DEFAULT_NOISE_ENTROPY_THRESHOLD:
                signal_types.append("STATIC")

            digest = hashlib.sha256()
            digest.update(f"{image.shape}:{image.dtype}".encode("ascii"))
            digest.update(image.tobytes())
            records.append(
                FrameRecord(
                    frame_time,
                    digest.hexdigest(),
                    mean_brightness,
                    entropy,
                    tuple(signal_types),
                )
            )
    return tuple(records)


def _video_duration(video: Path) -> float:
    import av

    with av.open(str(video)) as container:
        return float(container.duration) / av.time_base


def _match_frame_sequences(
    source_frames: tuple[FrameRecord, ...],
    output_frames: tuple[FrameRecord, ...],
) -> tuple[tuple[int, int], ...]:
    source_hashes = [frame.digest for frame in source_frames]
    output_hashes = [frame.digest for frame in output_frames]
    matching_blocks = SequenceMatcher(
        None, source_hashes, output_hashes, autojunk=False
    ).get_matching_blocks()
    return tuple(
        (block.a + offset, block.b + offset)
        for block in matching_blocks
        for offset in range(block.size)
    )


def compare_boundary_frames(
    source: Path,
    kept_video: Path,
    end_clip: Path,
    cut_time_s: float,
    window_seconds: float = 2.0,
) -> BoundaryComparison:
    """Match the same decoded frames in the source, kept output, and end clip."""
    if window_seconds <= 0.0:
        raise ValueError("window_seconds must be positive")
    if cut_time_s < 0.0:
        raise ValueError("cut_time_s must be non-negative")

    source_start = max(0.0, cut_time_s - window_seconds)
    source_end = cut_time_s + window_seconds
    source_frames = _read_frame_records(source, source_start, source_end)
    if not source_frames:
        raise ValueError("No source frames were decoded in the requested window")

    output_window_s = 2.0 * window_seconds + 2.0
    kept_duration = _video_duration(kept_video)
    kept_frames = _read_frame_records(
        kept_video, max(0.0, kept_duration - output_window_s), kept_duration + 1.0
    )
    end_frames = _read_frame_records(end_clip, 0.0, output_window_s)

    kept_matches = _match_frame_sequences(source_frames, kept_frames)
    end_matches = _match_frame_sequences(source_frames, end_frames)
    if not kept_matches:
        raise ValueError(
            "No exact source-frame hashes matched the end of the kept output; "
            "check the cut time and output paths"
        )
    if not end_matches:
        raise ValueError(
            "No exact source-frame hashes matched the beginning of the end clip; "
            "check the cut time and output paths"
        )
    return BoundaryComparison(
        cut_time_s,
        source_frames,
        kept_matches,
        end_matches,
    )


def format_comparison(comparison: BoundaryComparison) -> str:
    kept_indices = comparison.kept_source_indices
    end_indices = comparison.end_source_indices
    shared_indices = comparison.shared_source_indices
    cut_time_s = comparison.cut_time_s

    lines = [
        f"Cut boundary: {format_hms(cut_time_s)} ({cut_time_s:.3f}s)",
        "Hash: SHA-256 of decoded BGR24 pixels; full hashes are compared, "
        "16-character prefixes are shown.",
        "Source time | Kept | End clip | Signals | Mean | Entropy | SHA-256",
    ]
    for source_index, frame in enumerate(comparison.source_frames):
        if abs(frame.time_s - cut_time_s) > 1.0:
            continue
        signals = ", ".join(frame.signal_types) if frame.signal_types else "CONTENT"
        lines.append(
            f"{frame.time_s:10.3f} | "
            f"{'yes' if source_index in kept_indices else 'no ':>4} | "
            f"{'yes' if source_index in end_indices else 'no ':>8} | "
            f"{signals:<12} | {frame.mean_brightness:5.1f} | "
            f"{frame.entropy:5.2f} | {frame.digest[:16]}"
        )

    if shared_indices:
        shared_times = [
            comparison.source_frames[index].time_s for index in shared_indices
        ]
        lines.append(
            f"Shared source frames: {len(shared_indices)} "
            f"({min(shared_times):.3f}s to {max(shared_times):.3f}s)."
        )
    else:
        lines.append("Shared source frames: none in the inspected window.")

    end_only_after_cut = [
        comparison.source_frames[index]
        for index in end_indices - kept_indices
        if comparison.source_frames[index].time_s >= cut_time_s
    ]
    black_count = sum("BLACK" in frame.signal_types for frame in end_only_after_cut)
    static_count = sum("STATIC" in frame.signal_types for frame in end_only_after_cut)
    blue_count = sum("BLUE" in frame.signal_types for frame in end_only_after_cut)
    lines.append(
        f"End-only source frames at/after the cut: {len(end_only_after_cut)} "
        f"(BLACK={black_count}, STATIC={static_count}, BLUE={blue_count})."
    )
    return "\n".join(lines)


def _resolve_cut_time(video: Path, supplied_cut_time: float | None) -> float:
    if supplied_cut_time is not None:
        return supplied_cut_time
    first_no, first_time, last_time = find_content_range(
        video, DEFAULT_BLUE_THRESHOLD
    )
    _, end_time = find_keyframe_bounds(video, first_time, last_time)
    if end_time is None:
        raise ValueError(
            f"No trailing cut boundary found after frame #{first_no}; "
            "specify --cut-time to inspect a manually selected boundary"
        )
    return end_time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video", type=Path, help="Original source video")
    parser.add_argument(
        "--cut-time",
        type=float,
        help="Cut time in source-stream seconds (defaults to a fresh detection)",
    )
    parser.add_argument(
        "--kept-output",
        type=Path,
        help="Trimmed output (default: {input}_trimmed.mkv)",
    )
    parser.add_argument(
        "--end-clip",
        type=Path,
        help="Ending inspection clip (default: {input}_end.mkv)",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=2.0,
        help="Source time sampled on either side of the cut (default 2 seconds)",
    )
    args = parser.parse_args(argv)
    if args.window_seconds <= 0.0:
        parser.error("--window-seconds must be positive")
    if not args.video.is_file():
        parser.error(f"File not found: {args.video}")

    kept_output = args.kept_output or args.video.with_name(
        f"{args.video.stem}_trimmed.mkv"
    )
    end_clip = args.end_clip or args.video.with_name(f"{args.video.stem}_end.mkv")
    for path in (kept_output, end_clip):
        if not path.is_file():
            parser.error(f"File not found: {path}")

    try:
        cut_time_s = _resolve_cut_time(args.video, args.cut_time)
        comparison = compare_boundary_frames(
            args.video,
            kept_output,
            end_clip,
            cut_time_s,
            args.window_seconds,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(format_comparison(comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
