from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image
from tqdm import tqdm

import save_frame_from_file
import trim_blue_ends
import vidfile_iterator
from trim_blue_ends import (
    DEFAULT_BLACK_ENTROPY_THRESHOLD,
    DEFAULT_BLACK_BRIGHTNESS_THRESHOLD,
    DEFAULT_BLUE_THRESHOLD,
    DEFAULT_NOISE_ENTROPY_THRESHOLD,
    calculate_removable_signal_distance,
    calculate_removable_window_distance,
    classify_removed_areas,
    classify_signal_types,
    describe_removed_areas,
    find_content_range,
    find_keyframe_bounds,
    scan_signal_types,
)
from save_frame_from_file import (
    calculate_blue_screen_distance,
    find_first_dissimilar_frame,
    find_last_dissimilar_frame,
    find_last_dissimilar_frame_time,
)
from vidfile_iterator import get_frame_iterator_from_file

BLUE_VIDEO_PATH = Path(__file__).with_name("vid_with_blue_HD.mp4")
BLUE_DISTANCE_THRESHOLD = 0.6
NOISE_VIDEO_PATH = Path(__file__).parent / "artifacts" / "vid_with_noise.mkv"
NOISE_AT_END_PATH = Path(__file__).parent / "artifacts" / "vid_noise_at_end.mkv"
DARK_TRANSITION_VIDEO_PATH = (
    Path(__file__).parent / "data" / "dark-transition-boundary.mkv"
)
LOW_ENTROPY_BLACK_VIDEO_PATH = (
    Path(__file__).parent / "data" / "low-entropy-black-ending.mkv"
)
LEADING_BLACK_OUTLIER_VIDEO_PATH = (
    Path(__file__).parent / "data" / "leading-black-outlier.mkv"
)
BLUE_BLACK_OPENING_VIDEO_PATH = (
    Path(__file__).parent / "data" / "blue-black-opening-content-streak.mkv"
)


class _FakeVideoFrame:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image

    def to_ndarray(self, *, format: str) -> np.ndarray:
        assert format == "bgr24"
        return self.image


def _mock_video_frames(monkeypatch, frames: list[tuple[int, np.ndarray]]) -> None:
    frame_iterator = (
        (0, frame_no, _FakeVideoFrame(image))
        for frame_no, image in frames
    )
    monkeypatch.setattr(
        save_frame_from_file,
        "get_frame_iterator_from_file",
        lambda _video_file: frame_iterator,
    )


def test_frame_iterator_does_not_print_startup_message(monkeypatch, capsys) -> None:
    import sys

    monkeypatch.setattr(sys, "argv", ["trim_blue_ends.py", "source.mkv"])
    monkeypatch.setattr(
        vidfile_iterator,
        "FileFrameIterator",
        lambda _filename: SimpleNamespace(frame_iterator=iter(())),
    )

    assert list(get_frame_iterator_from_file("source.mkv")) == []
    assert capsys.readouterr().out == ""


def _collect_blue_transition_samples(
    video_path: Path, threshold: float
) -> tuple[list[tuple[int, int]], dict[int, np.ndarray]]:
    transitions: list[tuple[int, int]] = []
    samples: dict[int, np.ndarray] = {}
    previous: tuple[int, np.ndarray, bool] | None = None

    frames = tqdm(
        get_frame_iterator_from_file(video_path),
        desc="Full scan for blue transitions",
        unit=" frames",
    )
    for _, frame_no, video_frame in frames:
        frame = video_frame.to_ndarray(format="bgr24")
        is_blue = calculate_blue_screen_distance(frame) <= threshold
        if previous is None:
            samples[frame_no] = frame
        else:
            previous_no, previous_frame, previous_is_blue = previous
            if is_blue != previous_is_blue:
                transitions.append((previous_no, frame_no))
                samples[previous_no] = previous_frame
                samples[frame_no] = frame
        previous = (frame_no, frame, is_blue)
    frames.close()

    return transitions, samples


def test_find_first_dissimilar_frame_accepts_image_path_and_uses_default_distance(
    tmp_path, monkeypatch
) -> None:
    reference = np.full((64, 64, 3), 10, dtype=np.uint8)
    different = np.full((64, 64, 3), 40, dtype=np.uint8)
    reference_path = tmp_path / "reference.png"
    assert cv2.imwrite(str(reference_path), reference)
    _mock_video_frames(monkeypatch, [(1, reference), (2, different)])

    frame_no, frame = find_first_dissimilar_frame(
        "video.mp4", reference_path, mse_threshold=100.0
    )

    assert frame_no == 2
    np.testing.assert_array_equal(frame, different)


def test_find_first_dissimilar_frame_accepts_pil_image_and_custom_distance(
    monkeypatch
) -> None:
    reference_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    reference_rgb[:, :] = [255, 0, 0]
    video_frame = np.full((4, 4, 3), 7, dtype=np.uint8)
    _mock_video_frames(monkeypatch, [(1, video_frame)])
    seen_reference: list[np.ndarray] = []

    def distance_method(frame: np.ndarray, reference: np.ndarray) -> float:
        np.testing.assert_array_equal(frame, video_frame)
        seen_reference.append(reference)
        return 5.0

    frame_no, frame = find_first_dissimilar_frame(
        "video.mp4",
        Image.fromarray(reference_rgb),
        mse_threshold=4.0,
        distance_method=distance_method,
    )

    assert frame_no == 1
    np.testing.assert_array_equal(frame, video_frame)
    expected_reference_bgr = np.zeros_like(reference_rgb)
    expected_reference_bgr[:, :] = [0, 0, 255]
    np.testing.assert_array_equal(seen_reference[0], expected_reference_bgr)


def test_calculate_blue_screen_distance_measures_non_blue_pixel_fraction() -> None:
    blue = np.full((10, 10, 3), (255, 0, 0), dtype=np.uint8)
    mixed = blue.copy()
    mixed[:5] = (0, 0, 255)
    no_blue = np.full((10, 10, 3), (0, 0, 255), dtype=np.uint8)

    assert calculate_blue_screen_distance(blue) == 0.0
    assert calculate_blue_screen_distance(mixed) == 0.5
    assert calculate_blue_screen_distance(no_blue) == 1.0


def test_default_blue_threshold_requires_a_high_blue_pixel_fraction() -> None:
    blue = np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8)
    mostly_blue = blue.copy()
    mostly_blue[:15] = (0, 0, 255)
    near_solid_blue = blue.copy()
    near_solid_blue[:5] = (0, 0, 255)

    assert DEFAULT_BLUE_THRESHOLD == pytest.approx(0.1)
    assert classify_signal_types(
        mostly_blue, DEFAULT_BLUE_THRESHOLD, 10.0, -1.0
    ) == set()
    assert classify_signal_types(
        near_solid_blue, DEFAULT_BLUE_THRESHOLD, 10.0, -1.0
    ) == {"BLUE"}


def test_removable_signal_classifier_recognizes_blue_static_black_and_content() -> None:
    blue = np.full((32, 32, 3), (255, 0, 0), dtype=np.uint8)
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    pattern = ((np.indices((64, 64)).sum(axis=0) // 8) % 2 * 100 + 70).astype(
        np.uint8
    )
    content = np.stack((pattern, pattern, pattern), axis=2)
    noise = np.random.default_rng(0).integers(
        0, 256, size=(128, 128, 3), dtype=np.uint8
    )

    def kinds(frame: np.ndarray) -> set[str]:
        return classify_signal_types(
            frame,
            BLUE_DISTANCE_THRESHOLD,
            DEFAULT_NOISE_ENTROPY_THRESHOLD,
            DEFAULT_BLACK_ENTROPY_THRESHOLD,
        )

    assert kinds(blue) == {"BLUE"}
    assert kinds(black) == {"BLACK"}
    assert kinds(noise) == {"STATIC"}
    assert kinds(content) == set()
    assert calculate_removable_signal_distance(black) <= BLUE_DISTANCE_THRESHOLD
    assert calculate_removable_signal_distance(noise) <= BLUE_DISTANCE_THRESHOLD
    assert calculate_removable_signal_distance(content) > BLUE_DISTANCE_THRESHOLD


def test_low_entropy_is_not_black_without_very_low_brightness() -> None:
    bright_flat_content = np.full((32, 32, 3), 30, dtype=np.uint8)
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    dark_frame_with_codec_residual = np.full((32, 32, 3), 2, dtype=np.uint8)

    assert DEFAULT_BLACK_BRIGHTNESS_THRESHOLD == pytest.approx(5.0)
    assert DEFAULT_BLACK_ENTROPY_THRESHOLD == pytest.approx(1.9)
    assert classify_signal_types(
        bright_flat_content,
        DEFAULT_BLUE_THRESHOLD,
        DEFAULT_NOISE_ENTROPY_THRESHOLD,
        DEFAULT_BLACK_ENTROPY_THRESHOLD,
    ) == set()
    assert (
        calculate_removable_signal_distance(bright_flat_content)
        > DEFAULT_BLUE_THRESHOLD
    )
    assert classify_signal_types(
        black,
        DEFAULT_BLUE_THRESHOLD,
        DEFAULT_NOISE_ENTROPY_THRESHOLD,
        DEFAULT_BLACK_ENTROPY_THRESHOLD,
    ) == {"BLACK"}
    assert trim_blue_ends.is_black_frame(
        dark_frame_with_codec_residual,
        entropy=1.85,
        black_entropy_threshold=DEFAULT_BLACK_ENTROPY_THRESHOLD,
        black_brightness_threshold=DEFAULT_BLACK_BRIGHTNESS_THRESHOLD,
    )
    assert not trim_blue_ends.is_black_frame(
        dark_frame_with_codec_residual,
        entropy=1.95,
        black_entropy_threshold=DEFAULT_BLACK_ENTROPY_THRESHOLD,
        black_brightness_threshold=DEFAULT_BLACK_BRIGHTNESS_THRESHOLD,
    )


def test_removable_window_distance_requires_90_percent_matching_frames(
    monkeypatch,
) -> None:
    import av

    stream = SimpleNamespace(time_base=0.5, start_time=0, average_rate=2)
    frames = [
        SimpleNamespace(
            pts=index,
            to_ndarray=lambda value=index, **_: np.full(
                (1, 1, 3), value, dtype=np.uint8
            ),
        )
        for index in range(20)
    ]

    class Container:
        streams = SimpleNamespace(video=[stream])

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def decode(self, _stream):
            return iter(frames)

    monkeypatch.setattr(av, "open", lambda *_: Container())
    monkeypatch.setattr(
        trim_blue_ends,
        "calculate_removable_signal_distance",
        lambda image, **_: 0.0 if image[0, 0, 0] < 18 else 1.0,
    )
    assert calculate_removable_window_distance(
        Path("source.mkv"), 0.0, 10.0, 0.6, 4.5, 1.5
    ) == pytest.approx(0.1)

    monkeypatch.setattr(
        trim_blue_ends,
        "calculate_removable_signal_distance",
        lambda image, **_: 0.0 if image[0, 0, 0] < 17 else 1.0,
    )
    assert calculate_removable_window_distance(
        Path("source.mkv"), 0.0, 10.0, 0.6, 4.5, 1.5
    ) == pytest.approx(0.15)

    monkeypatch.setattr(
        save_frame_from_file,
        "calculate_blue_screen_distance",
        lambda image: 0.0 if image[0, 0, 0] < 18 else 1.0,
    )
    assert calculate_removable_window_distance(
        Path("source.mkv"),
        0.0,
        10.0,
        0.6,
        4.5,
        1.5,
        blue_only=True,
    ) == pytest.approx(0.1)


@pytest.mark.skipif(
    not NOISE_AT_END_PATH.is_file(),
    reason="tests/artifacts/vid_noise_at_end.mkv is not available",
)
def test_time_bisection_uses_window_distance_for_every_probe() -> None:
    window_checks: list[float] = []
    probe_reports: list[float] = []

    def window_distance(time_s: float, _frame: np.ndarray) -> float:
        window_checks.append(time_s)
        return 1.0 if time_s <= 20.0 else 0.0

    last_time, _ = find_last_dissimilar_frame_time(
        NOISE_AT_END_PATH,
        Image.new("RGB", (1, 1), color="black"),
        known_dissimilar_time=0.0,
        known_similar_time=30.0,
        mse_threshold=0.1,
        distance_method=lambda *_: 1.0,
        on_probe=lambda _time, distance: probe_reports.append(distance),
        distance_method_at_time=window_distance,
    )

    assert last_time is not None
    assert last_time == pytest.approx(20.0, abs=0.2)
    assert len(window_checks) == len(probe_reports)
    assert len(window_checks) > 5


def test_describe_removed_areas_can_omit_signal_types(monkeypatch) -> None:
    import av
    from contextlib import nullcontext

    monkeypatch.setattr(
        av,
        "open",
        lambda *_: nullcontext(SimpleNamespace(duration=10 * av.time_base)),
    )
    report = describe_removed_areas(
        Path("source.mkv"), 1.0, 9.0, None, None
    )

    assert report == (
        "Found removable areas: beginning interval from 00:00:00.00 to 00:00:01.00; "
        "ending interval from 00:00:09.00 to 00:00:10.00"
    )


def test_main_skips_full_signal_classification_by_default(
    tmp_path, monkeypatch
) -> None:
    import av
    from contextlib import nullcontext

    video = tmp_path / "source.mkv"
    video.touch()
    monkeypatch.setattr(trim_blue_ends.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(trim_blue_ends, "find_content_range", lambda *args: (0, 0.0, 10.0))
    monkeypatch.setattr(trim_blue_ends, "find_keyframe_bounds", lambda *args: (0.0, None))
    monkeypatch.setattr(
        av,
        "open",
        lambda *_: nullcontext(SimpleNamespace(duration=10 * av.time_base)),
    )
    monkeypatch.setattr(
        trim_blue_ends,
        "classify_removed_areas",
        lambda *_: pytest.fail("full interval classification should be opt-in"),
    )
    monkeypatch.setattr(trim_blue_ends, "describe_removed_areas", lambda *args: "No removable areas")
    monkeypatch.setattr(trim_blue_ends, "build_ffmpeg_command", lambda *args: ["ffmpeg"])

    assert trim_blue_ends.main([str(video), "--dry-run"]) == 0


def test_ffmpeg_commands_keep_stats_at_warning_log_level() -> None:
    command = trim_blue_ends.build_ffmpeg_command(
        Path("source.mkv"), Path("output.mkv"), 0.0, 0.0, None, False
    )

    assert command[1:5] == [
        "-hide_banner",
        "-loglevel",
        "warning",
        "-stats",
    ]


def test_ffmpeg_run_is_surrounded_by_blank_lines(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        trim_blue_ends.subprocess,
        "run",
        lambda *_: SimpleNamespace(returncode=0),
    )

    assert trim_blue_ends._run_ffmpeg(["ffmpeg"]) == 0
    assert capsys.readouterr().out == "\n\n\n\n\n\n"


@pytest.mark.parametrize(
    ("blue_window_distance", "expected_keep_start"),
    [(0.0, 5.0), (1.0, 0.0)],
)
def test_short_leading_edge_is_removed_only_when_blue_is_confirmed(
    tmp_path, monkeypatch, capsys, blue_window_distance, expected_keep_start
) -> None:
    import av
    from contextlib import nullcontext

    video = tmp_path / "source.mkv"
    video.touch()
    blue_only_checks: list[bool] = []

    def window_distance(*args, **kwargs) -> float:
        blue_only_checks.append(kwargs.get("blue_only", False))
        return blue_window_distance

    monkeypatch.setattr(trim_blue_ends.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(
        trim_blue_ends, "find_content_range", lambda *args: (100, 5.0, 10.0)
    )
    monkeypatch.setattr(
        trim_blue_ends, "find_keyframe_bounds", lambda *args: (0.0, None)
    )
    monkeypatch.setattr(
        trim_blue_ends,
        "calculate_removable_window_distance",
        window_distance,
    )
    monkeypatch.setattr(trim_blue_ends, "describe_removed_areas", lambda *args: "report")
    monkeypatch.setattr(
        av,
        "open",
        lambda *_: nullcontext(SimpleNamespace(duration=10 * av.time_base)),
    )

    assert trim_blue_ends.main([str(video), "--dry-run"]) == 0
    assert blue_only_checks == [True]
    assert f"* Keep range: {expected_keep_start:.3f}s" in capsys.readouterr().out


def test_scan_signal_types_decodes_from_start_without_seeking(monkeypatch) -> None:
    import av

    stream = SimpleNamespace(time_base=1.0, start_time=100)
    frame = SimpleNamespace(
        pts=100,
        to_ndarray=lambda **_: np.zeros((1, 1, 3), dtype=np.uint8),
    )

    class Container:
        streams = SimpleNamespace(video=[stream])

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def seek(self, *_args, **_kwargs):
            raise AssertionError("the beginning scan must not seek past the first frames")

        def decode(self, _stream):
            return iter([frame])

    monkeypatch.setattr(av, "open", lambda *_: Container())
    monkeypatch.setattr(trim_blue_ends, "classify_signal_types", lambda *_: {"BLACK"})
    assert scan_signal_types(
        Path("source.mkv"), 0.0, 0.1, 0.6, 4.5, 1.5, "test scan"
    ) == {"BLACK"}


def test_find_first_dissimilar_frame_can_use_blue_screen_distance(
    monkeypatch
) -> None:
    blue = np.full((10, 10, 3), (255, 0, 0), dtype=np.uint8)
    mostly_non_blue = blue.copy()
    mostly_non_blue[:6] = (0, 0, 255)
    _mock_video_frames(monkeypatch, [(1, blue), (2, mostly_non_blue)])

    frame_no, frame = find_first_dissimilar_frame(
        "video.mp4",
        Image.fromarray(np.full((10, 10, 3), (0, 0, 255), dtype=np.uint8)),
        mse_threshold=0.5,
        distance_method=calculate_blue_screen_distance,
    )

    assert frame_no == 2
    np.testing.assert_array_equal(frame, mostly_non_blue)


@pytest.mark.skipif(
    not BLUE_VIDEO_PATH.is_file(),
    reason="tests/vid_with_blue_HD.mp4 is not available",
)
def test_blue_video_start_and_end_transitions() -> None:
    reference = Image.new("RGB", (1, 1), color="black")

    frame_no, frame = find_first_dissimilar_frame(
        BLUE_VIDEO_PATH,
        reference,
        mse_threshold=BLUE_DISTANCE_THRESHOLD,
        distance_method=calculate_blue_screen_distance,
    )

    assert frame_no == 93
    assert calculate_blue_screen_distance(frame) > BLUE_DISTANCE_THRESHOLD

    transitions, samples = _collect_blue_transition_samples(
        BLUE_VIDEO_PATH, BLUE_DISTANCE_THRESHOLD
    )
    assert transitions == [(92, 93), (2824, 2825)]

    diagnostic_dir = Path(__file__).parent / "artifacts" / "blue_transitions"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    sample_frames = [0, 92, 93, 2824, 2825]
    for sample_frame in sample_frames:
        output_path = diagnostic_dir / f"frame_{sample_frame:05}.png"
        assert cv2.imwrite(str(output_path), samples[sample_frame])

    print(f"Blue transition diagnostics saved to: {diagnostic_dir}")


@pytest.mark.skipif(
    not BLUE_VIDEO_PATH.is_file(),
    reason="tests/vid_with_blue_HD.mp4 is not available",
)
def test_bisection_finds_start_of_trailing_blue_run(request, capsys) -> None:
    reference = Image.new("RGB", (1, 1), color="black")
    probed: list[int] = []
    show_progress = request.config.get_verbosity() >= 0  # i.e. not run with -q

    with capsys.disabled():
        bar = tqdm(
            desc="Bisection search for trailing blue run",
            unit=" bisection probes",
            disable=not show_progress,
        )

        def on_probe(index: int, distance: float) -> None:
            probed.append(index)
            bar.update(1)
            kind = "not blue" if distance > BLUE_DISTANCE_THRESHOLD else "blue"
            bar.write(
                f"bisect probe {len(probed):2d}: frame {index:5d}  "
                f"distance={distance:.3f}  -> {kind}"
            )

        frame_no, frame = find_last_dissimilar_frame(
            BLUE_VIDEO_PATH,
            reference,
            known_dissimilar_frame=93,
            mse_threshold=BLUE_DISTANCE_THRESHOLD,
            distance_method=calculate_blue_screen_distance,
            on_probe=on_probe,
        )
        bar.close()

    assert frame_no == 2824
    assert calculate_blue_screen_distance(frame) > BLUE_DISTANCE_THRESHOLD
    assert len(probed) < 30


@pytest.mark.skipif(
    not NOISE_VIDEO_PATH.is_file(),
    reason="tests/artifacts/vid_with_noise.mkv is not available",
)
def test_short_noise_tail_does_not_qualify_as_a_ten_second_streak() -> None:
    first_no, first_time, last_time = find_content_range(
        NOISE_VIDEO_PATH, BLUE_DISTANCE_THRESHOLD
    )

    assert first_no == 141
    assert first_time == pytest.approx(5.64, abs=0.04)
    assert last_time == pytest.approx(26.328, abs=0.04)

    start_s, end_s = find_keyframe_bounds(NOISE_VIDEO_PATH, first_time, last_time)
    assert end_s is None
    beginning, ending = classify_removed_areas(
        NOISE_VIDEO_PATH,
        start_s,
        end_s,
        BLUE_DISTANCE_THRESHOLD,
        5.0,
    )
    assert beginning == {"BLUE", "BLACK", "STATIC"}
    assert ending == set()

    report = describe_removed_areas(
        NOISE_VIDEO_PATH, start_s, end_s, beginning, ending
    )
    assert "BLUE, BLACK, and STATIC at beginning" in report
    assert "STATIC at end" not in report


@pytest.mark.skipif(
    not NOISE_AT_END_PATH.is_file(),
    reason="tests/artifacts/vid_noise_at_end.mkv is not available",
)
def test_noise_at_end_video_uses_default_cutoff() -> None:
    first_no, first_time, last_time = find_content_range(
        NOISE_AT_END_PATH, BLUE_DISTANCE_THRESHOLD
    )

    assert first_no == 0
    assert first_time == pytest.approx(0.0, abs=0.04)
    assert last_time == pytest.approx(26.32, abs=0.2)

    start_s, end_s = find_keyframe_bounds(
        NOISE_AT_END_PATH, first_time, last_time
    )
    assert start_s == pytest.approx(0.0, abs=0.005)
    assert end_s is not None
    assert end_s > last_time
    beginning, ending = classify_removed_areas(
        NOISE_AT_END_PATH,
        start_s,
        end_s,
        BLUE_DISTANCE_THRESHOLD,
        DEFAULT_NOISE_ENTROPY_THRESHOLD,
    )
    assert beginning == set()
    assert ending == {"STATIC"}


def test_trailing_cut_preserves_content_until_actual_darkness() -> None:
    first_no, first_time, last_time = find_content_range(
        DARK_TRANSITION_VIDEO_PATH, DEFAULT_BLUE_THRESHOLD
    )

    assert first_no == 0
    assert first_time == pytest.approx(0.0, abs=0.04)
    assert last_time == pytest.approx(24.64, abs=0.35)

    start_s, end_s = find_keyframe_bounds(
        DARK_TRANSITION_VIDEO_PATH, first_time, last_time
    )
    assert start_s == pytest.approx(0.0, abs=0.04)
    assert end_s == pytest.approx(25.0, abs=0.05)


def test_dark_low_entropy_tail_is_removed_but_dim_content_is_kept() -> None:
    first_no, first_time, last_time = find_content_range(
        LOW_ENTROPY_BLACK_VIDEO_PATH, DEFAULT_BLUE_THRESHOLD
    )

    assert first_no == 0
    assert first_time == pytest.approx(0.0, abs=0.04)
    assert 19.0 <= last_time <= 24.0

    start_s, end_s = find_keyframe_bounds(
        LOW_ENTROPY_BLACK_VIDEO_PATH, first_time, last_time
    )
    assert start_s == pytest.approx(0.0, abs=0.04)
    assert end_s is not None
    assert 20.0 <= end_s <= 25.0


def test_leading_black_is_found_after_a_single_content_outlier() -> None:
    initial_window_distance = calculate_removable_window_distance(
        LEADING_BLACK_OUTLIER_VIDEO_PATH,
        0.0,
        10.0,
        DEFAULT_BLUE_THRESHOLD,
        DEFAULT_NOISE_ENTROPY_THRESHOLD,
        DEFAULT_BLACK_ENTROPY_THRESHOLD,
        black_brightness_threshold=DEFAULT_BLACK_BRIGHTNESS_THRESHOLD,
    )
    assert trim_blue_ends.is_confirmed_removable_window(initial_window_distance)

    first_no, first_time, _ = find_content_range(
        LEADING_BLACK_OUTLIER_VIDEO_PATH,
        DEFAULT_BLUE_THRESHOLD,
        minimum_streak_seconds=8.0,
    )

    assert first_no > 0
    assert first_time == pytest.approx(20.52, abs=0.5)


def test_leading_boundary_waits_for_sustained_informative_content() -> None:
    first_no, first_time, _ = find_content_range(
        BLUE_BLACK_OPENING_VIDEO_PATH, DEFAULT_BLUE_THRESHOLD
    )

    assert first_no > 0
    assert first_time == pytest.approx(61.0, abs=0.5)


def test_trim_report_is_repeated_after_ffmpeg(tmp_path, monkeypatch, capsys) -> None:
    import av
    from contextlib import nullcontext

    video = tmp_path / "source.mkv"
    video.touch()
    report = "Found removable areas: BLUE and STATIC (both)"
    built_commands: list[tuple[Path, float, float, float | None, bool]] = []

    def build_command(
        _video_file: Path,
        output_file: Path,
        seek_s: float,
        start_s: float,
        end_s: float | None,
        overwrite: bool,
    ) -> list[str]:
        built_commands.append((output_file, seek_s, start_s, end_s, overwrite))
        return ["ffmpeg", str(output_file)]

    monkeypatch.setattr(trim_blue_ends.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(trim_blue_ends, "find_content_range", lambda *args: (10, 1.0, 9.0))
    monkeypatch.setattr(trim_blue_ends, "find_keyframe_bounds", lambda *args: (0.5, 9.5))
    monkeypatch.setattr(
        trim_blue_ends,
        "classify_removed_areas",
        lambda *args: ({"BLUE", "STATIC"}, {"STATIC"}),
    )
    monkeypatch.setattr(trim_blue_ends, "describe_removed_areas", lambda *args: report)
    monkeypatch.setattr(trim_blue_ends, "build_ffmpeg_command", build_command)
    monkeypatch.setattr(
        av,
        "open",
        lambda *_: nullcontext(SimpleNamespace(duration=10 * av.time_base)),
    )
    monkeypatch.setattr(
        trim_blue_ends.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(trim_blue_ends, "video_pts_span", lambda *_: (0.0, 9.5))

    assert trim_blue_ends.main(
        [
            str(video),
            "--min-removable-streak-seconds",
            "0",
            "--report-signal-types",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert output.count(report) == 2
    assert f"* {report}\n\n\n* ffmpeg" in output
    assert [command[0].name for command in built_commands] == [
        "source_trimmed.mkv",
        "source_begin.mkv",
        "source_end.mkv",
    ]
    assert built_commands[1][1:4] == (0.0, 0.0, 1.0)
    assert built_commands[2][1:4] == (9.5, 9.5, None)
    assert output.count("Inspection clip ") == 2
    assert output.count("Wrote inspection clip ") == 2