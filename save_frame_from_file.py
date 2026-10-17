import os
import sys
from typing import Any, Callable, Iterator
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import WindowClosed, WindowManager, get_frames_estimate
from vidfile_iterator import (FileFrameIterator, iterate_frames_from_packet_stream, 
                              stream_frames_from_packet, decode_packet_to_frames,
                              get_frame_iterator_from_file, get_frames_from_iterator)
from utils import calculate_mse_between_frames, _compute_entropy, EtaTqdm


RESIZE_DIM_FOR_MSE = (64, 64)  # Resize dimensions for MSE calculation
BLUE_HSV_LOWER = np.array([90, 70, 40], dtype=np.uint8)
BLUE_HSV_UPPER = np.array([130, 255, 255], dtype=np.uint8)

# Weights for ``combined_entropy`` in CSV / ``find_last_proper_video_frame`` (must match each other).
FRAME_METRICS_COMBINED_LOCAL_WEIGHT = 0.7
FRAME_METRICS_COMBINED_TEMPORAL_WEIGHT = 0.3


def calculate_local_entropy(gray, blur_ksize=5):
    """Calculate entropy of the local high-frequency residual around the frame."""
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0).astype(np.uint8)
    residual = cv2.absdiff(gray, blur)#.astype(np.uint8)
    return _compute_entropy(residual)


def calculate_laplacian_variance(gray):
    """Calculate variance of Laplacian for sharpness/noise detection."""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def calculate_temporal_entropy(current_gray, previous_gray):
    """Calculate entropy of the difference from the previous frame (temporal noise detection)."""
    if previous_gray is None:
        return 0.0
    diff = cv2.absdiff(current_gray, previous_gray)
    return _compute_entropy(diff)


def calculate_blue_screen_distance(
    frame_bgr: np.ndarray, _reference_bgr: np.ndarray | None = None
) -> float:
    """Return the fraction of pixels not recognized as blue (0 = all blue, 1 = no blue).

    The blue range is defined in OpenCV HSV coordinates. Saturation and value
    minimums exclude gray and very dark pixels from counting as blue.
    """
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(frame_hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER)
    blue_fraction = np.count_nonzero(blue_mask) / blue_mask.size
    return float(1.0 - blue_fraction)

def calculate_edge_density(gray, threshold=50):
    """Calculate the proportion of edges/high-frequency content in the frame."""
    edges = cv2.Canny(gray, threshold//2, threshold)
    edge_pixels = np.count_nonzero(edges)
    total_pixels = edges.size
    return edge_pixels / total_pixels

def calculate_darkness_ratio(gray, pixel_threshold=40):
    gray = gray.astype(np.float32)

    mean = gray.mean()
    std = gray.std()
    dark_ratio = np.mean(gray < pixel_threshold)

    # high = likely black
    return dark_ratio / (1.0 + std / 10.0)

def calculate_histogram_entropy(gray):
    """Calculate entropy of the raw pixel value histogram (measures image variability)."""
    return _compute_entropy(gray)


def calculate_fft_highband_ratio(gray, cutoff_ratio=0.25):
    """
    Measure proportion of spectral energy in high frequencies.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image (2D).
        
    cutoff_ratio : float
        Radius threshold as fraction of image size.
        Frequencies beyond this radius are considered "high".

    Returns
    -------
    float
        High-frequency energy ratio.
    """

    # Convert to float

    gray_small  = cv2.resize(gray, (128, 96))
    gray = gray_small.astype(np.float32)

    # gray = gray.astype(np.float32)

    # Optional but recommended:
    # remove DC bias (average brightness)
    gray = gray - np.mean(gray)

    # 2D FFT
    fft = np.fft.fft2(gray)

    # Shift low frequencies to center
    fft_shifted = np.fft.fftshift(fft)

    # Power spectrum
    magnitude = np.abs(fft_shifted) ** 2

    h, w = gray.shape
    cy, cx = h // 2, w // 2

    # Distance-from-center map
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Radius threshold
    cutoff = min(h, w) * cutoff_ratio

    # High-frequency mask
    high_mask = distance >= cutoff

    # Energy calculations
    high_energy = magnitude[high_mask].sum()
    total_energy = magnitude.sum()

    # Avoid division by zero
    if total_energy == 0:
        return 0.0

    return high_energy / total_energy

def calculate_temporal_fft_highband_ratio(current_gray, previous_gray):
    """Calculate the high-frequency energy ratio of the difference from the previous frame (temporal noise detection)."""
    if previous_gray is None:
        return 0.0
    diff = cv2.absdiff(current_gray, previous_gray)
    return calculate_fft_highband_ratio(diff)





def is_proper_frame(frame_bgr, previous_bgr=None, mean_threshold=50.0, local_entropy_threshold=5.0, temporal_entropy_weight=0.3):
    """
    Check if a frame is 'proper' (not noise/black).
    Works with multi-channel frames (BGR) to capture richer entropy across color channels.
    A frame is considered proper if its mean brightness is above threshold and
    its combined local + temporal entropy is below threshold.
    """
    mean_val = np.mean(frame_bgr)  # Mean across all channels
    local_entropy_val = calculate_local_entropy(frame_bgr)
    temporal_entropy_val = calculate_temporal_entropy(frame_bgr, previous_bgr)
    combined_entropy = (1 - temporal_entropy_weight) * local_entropy_val + temporal_entropy_weight * temporal_entropy_val
    return is_proper_frame_params(mean_val, combined_entropy, mean_threshold, local_entropy_threshold)


functions_to_calculate_metrics_dict = {
    "timestamp": lambda **kwargs: kwargs["frame_no"] / kwargs["fps"] if kwargs["fps"] > 0 else 0.0,
    "frame": lambda **kwargs: kwargs["frame_no"],
    "mean": lambda **kwargs: np.mean(kwargs["frame_bgr"]),
    "local_entropy": lambda **kwargs: calculate_local_entropy(kwargs["frame_bgr"]),
    "darkness_ratio": lambda **kwargs: calculate_darkness_ratio(kwargs["frame_bgr"]),
    # "edge_density": lambda **kwargs: calculate_edge_density(kwargs["gray_for_edges"]),
    # "hist_entropy": lambda **kwargs: calculate_histogram_entropy(kwargs["gray_for_edges"]),
    # "temporal_entropy": lambda **kwargs: calculate_temporal_entropy(kwargs["current_gray"], kwargs["previous_gray"]),
    "laplacian_variance": lambda **kwargs: calculate_laplacian_variance(kwargs["gray_for_edges"]),
    "fft_highband_ratio": lambda **kwargs: calculate_fft_highband_ratio(kwargs["gray_for_edges"]),
    "temporal_fft_highband_ratio": lambda **kwargs: calculate_temporal_fft_highband_ratio(kwargs["current_gray"], kwargs["previous_gray"])
    }
    

def compute_frame_metrics_row(
    frame_bgr: np.ndarray,
    previous_bgr: np.ndarray | None,
    frame_no: int,
    fps: float,
) -> dict[str, float | int]:
    """
    One row of the same metrics written to ``debug_csv`` in ``find_last_proper_video_frame``.

    Use with ``iter_frame_metrics_rows`` to stream metrics without a CSV file (same pipeline as the notebook).
    """

    gray_for_edges = (
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if len(frame_bgr.shape) == 3 else frame_bgr
    )
    if previous_bgr is None:
        previous_gray = gray_for_edges
    else:
        previous_gray = (
            cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY) if len(previous_bgr.shape) == 3 else previous_bgr
        )


    all_params = {
        "frame_bgr": frame_bgr,
        "previous_bgr": previous_bgr,
        "current_gray": gray_for_edges,
        "previous_gray": previous_gray,
        "frame_no": frame_no,
        "fps": fps,
        "gray_for_edges": gray_for_edges,
    }

    metrics = {}
    for metric_name, metric_function in functions_to_calculate_metrics_dict.items():
        # print("All params: ", all_params)
        metrics[metric_name] = metric_function(**all_params)
    
    return metrics

    # timestamp = frame_no / fps if fps > 0 else 0.0
    # mean_val = float(np.mean(frame_bgr))
    # local_entropy_val = calculate_local_entropy(frame_bgr)
    # temporal_entropy_val = calculate_temporal_entropy(frame_bgr, previous_bgr)
    # combined_entropy = (
    #     FRAME_METRICS_COMBINED_LOCAL_WEIGHT * local_entropy_val
    #     + FRAME_METRICS_COMBINED_TEMPORAL_WEIGHT * temporal_entropy_val
    # )
    # # print('frame_bgr.shape: ', frame_bgr.shape)
    # hist_entropy = calculate_histogram_entropy(frame_bgr)

    # edge_density = calculate_edge_density(gray_for_edges)
    # laplacian_var = calculate_laplacian_variance(gray_for_edges)
    # return {
    #     "frame": frame_no,
    #     "timestamp": timestamp,
    #     "mean": mean_val,
    #     "local_entropy": local_entropy_val,
    #     "temporal_entropy": temporal_entropy_val,
    #     "combined_entropy": combined_entropy,
    #     "edge_density": edge_density,
    #     "hist_entropy": hist_entropy,
    #     "laplacian_variance": laplacian_var,
    # }

    # return {
    #     "frame": 0,
    #     "timestamp": 0,
    #     "mean": 0.0,
    #     "local_entropy": 0.0,
    #     "temporal_entropy": 0.0,
    #     "combined_entropy": 0.0,
    #     "edge_density": 0.0,
    #     "hist_entropy": 0.0,
    #     "laplacian_variance": 0.0,
    # }




def iter_frame_metrics_rows(
    image_iterator: Iterator[tuple[int, np.ndarray]],
    total_frames: int,
    fps: float,
    *,
    use_tqdm: bool = True,
) -> Iterator[dict[str, float | int]]:
    """
    Yield metric dicts in CSV column order for each ``(frame_no, frame_bgr)``.

    Parameters
    ----------
    image_iterator:
        Iterator of ``(frame_no, frame_bgr)`` pairs.
    total_frames:
        Estimated total frame count for progress reporting.
    fps:
        Frames per second used to derive the ``timestamp`` metric.
    use_tqdm:
        When ``True`` (default), wrap the iterator in ``EtaTqdm`` for a
        progress bar. Set to ``False`` when you want to control tqdm at a
        higher level (e.g. in ``locate_noise``).

    Notes
    -----
    This maintains ``previous_bgr`` exactly like
    ``find_last_proper_video_frame``.
    """
    previous_bgr = None
    iterable = (
        EtaTqdm(image_iterator, total=total_frames, desc="Processing frames")
        if use_tqdm
        else image_iterator
    )
    for frame_no, frame_bgr in iterable:
        metrics = compute_frame_metrics_row(frame_bgr, previous_bgr, frame_no, fps)
        yield metrics
        previous_bgr = frame_bgr


def is_proper_frame_params(frame_mean, frame_entropy, mean_threshold, entropy_threshold):
    """
    Check if a frame is 'proper' (not noise/black) based on mean and local entropy.
    A frame is considered proper if its mean brightness is above threshold and
    its local entropy is below threshold.
    """
    return frame_mean > mean_threshold and frame_entropy < entropy_threshold



def find_last_proper_video_frame(video_file, mean_threshold=0.0, local_entropy_threshold=5.0, debug_csv=None):
    """
    Find the last frame that is 'proper' before frames become noise/black.
    
    Args:
        video_file: Path to the video file
        mean_threshold: Minimum mean brightness for a proper frame
        local_entropy_threshold: Maximum local entropy for a proper frame
        debug_csv: Optional path to save frame metrics to CSV for analysis
    
    Returns:
        Tuple of (frame_no, frame_image) for the last proper frame, or (None, None) if none found
    """
    frame_iterator = get_frame_iterator_from_file(video_file)
    image_iterator = get_frames_from_iterator(frame_iterator)
    
    # Estimate total frames from video metadata (may be inaccurate)
    total_frames, fps = get_frames_estimate(video_file)
    
    last_proper_frame_no = None
    last_proper_frame_bgr = None
    previous_bgr = None
    
    print("Looking for the last proper frame before noise/black frames start...")
    print("Mean threshold:", mean_threshold)
    print("Local entropy threshold:", local_entropy_threshold)
    if debug_csv:
        print(f"Saving frame metrics to: {debug_csv}")

    metrics_log = []

    with EtaTqdm(image_iterator, total=total_frames, desc="Processing frames") as pbar:
        for frame_no, frame_bgr in pbar:
            row = compute_frame_metrics_row(frame_bgr, previous_bgr, frame_no, fps)
            mean_val = row["mean"]
            local_entropy_val = row["local_entropy"]
            temporal_entropy_val = row["temporal_entropy"]
            combined_entropy = row["combined_entropy"]
            edge_density = row["edge_density"]
            laplacian_var = row["laplacian_variance"]
            timestamp = row["timestamp"]

            pbar.set_postfix({
                    'frame': frame_no, 
                    'time': f"{timestamp:.2f}s", 
                    'mean': f"{mean_val:.1f}", 
                    'local_ent': f"{local_entropy_val:.2f}", 
                    'temp_ent': f"{temporal_entropy_val:.2f}", 
                    'combined': f"{combined_entropy:.2f}", 
                    'edges': f"{edge_density:.3f}", 
                    'lap_var': f"{laplacian_var:.1f}"}
                )
            

            if debug_csv:
                metrics_log.append(row)

            if is_proper_frame_params(mean_val, combined_entropy, mean_threshold, local_entropy_threshold):
                last_proper_frame_no = frame_no
                last_proper_frame_bgr = frame_bgr
            elif last_proper_frame_no is not None:
                # Frames have started becoming improper, return the last proper one
                if debug_csv:
                    _save_metrics_csv(debug_csv, metrics_log)
                return last_proper_frame_no, last_proper_frame_bgr
            
            previous_bgr = frame_bgr
    
    if debug_csv:
        _save_metrics_csv(debug_csv, metrics_log)
    # If all frames are proper or no improper frames found, return the last proper one
    return last_proper_frame_no, last_proper_frame_bgr


def _save_metrics_csv(filepath, metrics_list):
    """Save frame metrics to CSV for analysis and debugging."""
    import csv
    if not metrics_list:
        return
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_list[0].keys())
        writer.writeheader()
        writer.writerows(metrics_list)
    print(f"Metrics saved to {filepath}")




def find_first_dissimilar_frame(
    video_file,
    reference_image_path: str | os.PathLike[str] | Image.Image,
    min_frame: int = 0,
    mse_threshold: float = 1000.0,
    distance_method: Callable[[np.ndarray, np.ndarray], float] = calculate_mse_between_frames,
    with_time: bool = False,
):
    """
    Pipeline to find the first frame after min_frame that is not similar to the reference image.

    With ``with_time=True`` the result is ``(frame_no, frame_image, time_s)``
    (``(None, None, None)`` if not found), where ``time_s`` is the frame's real
    presentation time in seconds from the stream's time base.
    
    Args:
        video_file: Path to the video file
        reference_image_path: Path to a reference image or a PIL image
        min_frame: Minimum frame number to start checking from
        mse_threshold: Distance threshold above which images are considered dissimilar
        distance_method: Function comparing a BGR video frame and BGR reference image.
            Defaults to ``calculate_mse_between_frames``. For blue-screen
            detection, use ``calculate_blue_screen_distance`` and set
            ``mse_threshold`` between 0 and 1 (for example, 0.5 to classify
            frames with less than 50% recognized blue as dissimilar).
    
    Returns:
        Tuple of (frame_no, frame_image) for the first dissimilar frame, or (None, None) if none found
    """
    if isinstance(reference_image_path, Image.Image):
        ref_img = cv2.cvtColor(
            np.asarray(reference_image_path.convert("RGB")), cv2.COLOR_RGB2BGR
        )
    else:
        ref_path = os.fspath(reference_image_path)
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            raise ValueError(f"Could not load reference image: {ref_path}")
    
    frame_iterator = get_frame_iterator_from_file(video_file)
    for _, frame_no, frame in frame_iterator:
        if frame_no <= min_frame:
            continue

        frame_bgr = frame.to_ndarray(format="bgr24")
        distance = distance_method(frame_bgr, ref_img)
        if distance > mse_threshold:
            if with_time:
                return frame_no, frame_bgr, frame.time
            return frame_no, frame_bgr

    return (None, None, None) if with_time else (None, None)


def _load_reference_bgr(reference_image: str | os.PathLike[str] | Image.Image) -> np.ndarray:
    if isinstance(reference_image, Image.Image):
        return cv2.cvtColor(np.asarray(reference_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    ref_path = os.fspath(reference_image)
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        raise ValueError(f"Could not load reference image: {ref_path}")
    return ref_img


def find_last_dissimilar_frame(
    video_file,
    reference_image: str | os.PathLike[str] | Image.Image,
    known_dissimilar_frame: int = 0,
    mse_threshold: float = 1000.0,
    distance_method: Callable[[np.ndarray, np.ndarray], float] = calculate_mse_between_frames,
    on_probe: Callable[[int, float], None] | None = None,
):
    """
    Find the last dissimilar frame, i.e. the start of a trailing run of frames that
    resemble the reference, using bisection with random-access seeks instead of
    decoding the whole file.

    Assumes constant frame rate and that everything after the returned frame is
    similar to the reference (a single trailing run). Isolated similar frames
    between ``known_dissimilar_frame`` and the tail are NOT detected.

    Args:
        known_dissimilar_frame: Frame number already known to be dissimilar (for
            example the result of ``find_first_dissimilar_frame``). Bisection
            starts from here.
        on_probe: Optional callback `(frame_index, distance)` invoked for every
            probed frame, e.g. to display bisection progress.

    Returns:
        (frame_no, frame_bgr) of the last dissimilar frame. If the final frame is
        itself dissimilar there is no trailing run and the final frame is returned.
        (None, None) if ``known_dissimilar_frame`` is beyond the end of the video.
    """
    import av

    ref_img = _load_reference_bgr(reference_image)

    with av.open(os.fspath(video_file)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate)
        time_base = float(stream.time_base)
        start_pts = stream.start_time or 0

        frame_count = stream.frames
        if not frame_count:
            frame_count = round(float(container.duration) / av.time_base * rate)

        def frame_at(index: int) -> np.ndarray | None:
            target_pts = start_pts + round(index / rate / time_base)
            container.seek(target_pts, backward=True, any_frame=False, stream=stream)
            for frame in container.decode(stream):
                if frame.pts is not None and frame.pts >= target_pts - 1:
                    return frame.to_ndarray(format="bgr24")
            return None

        def is_dissimilar(index: int) -> tuple[bool, np.ndarray | None]:
            image = frame_at(index)
            if image is None:
                return False, None
            distance = distance_method(image, ref_img)
            if on_probe is not None:
                on_probe(index, distance)
            return distance > mse_threshold, image

        # Last decodable frame (the declared count can be off by a frame or two).
        hi = frame_count - 1
        while hi > known_dissimilar_frame:
            last_dissimilar, last_image = is_dissimilar(hi)
            if last_image is not None:
                break
            hi -= 1
        else:
            return None, None
        if last_dissimilar:
            return hi, last_image

        lo = known_dissimilar_frame
        lo_image = frame_at(lo)
        # Invariant: lo is dissimilar, hi is similar.
        while hi - lo > 1:
            mid = (lo + hi) // 2
            dissimilar, image = is_dissimilar(mid)
            if image is None:
                hi = mid
            elif dissimilar:
                lo, lo_image = mid, image
            else:
                hi = mid
        return lo, lo_image


def find_last_dissimilar_frame_time(
    video_file,
    reference_image: str | os.PathLike[str] | Image.Image,
    known_dissimilar_time: float,
    mse_threshold: float = 1000.0,
    distance_method: Callable[[np.ndarray, np.ndarray], float] = calculate_mse_between_frames,
    on_probe: Callable[[float, float], None] | None = None,
    distance_method_at_time: Callable[[float, np.ndarray], float] | None = None,
    known_similar_time: float | None = None,
):
    """
    Like ``find_last_dissimilar_frame`` but bisects on presentation time, so it
    needs neither a constant frame rate nor a frame index.

    Args:
        known_dissimilar_time: Presentation time (seconds, as ``frame.time``) of a
            frame already known to be dissimilar.
        on_probe: Optional callback `(time_s, distance)` for every probed frame.
        distance_method_at_time: Optional time-aware distance override used for
            endpoint and bisection probes, for example to validate a surrounding
            frame window.
        known_similar_time: Optional presentation time already known to be
            similar. When supplied, it is used as the bisection upper bound
            instead of discovering the final frame of the file.

    Returns:
        (time_s, frame_bgr) of the last dissimilar frame, using the same time
        base as ``frame.time``. (None, None) if the known frame is at or after
        the last decodable frame and no later frame exists.
    """
    import av

    ref_img = _load_reference_bgr(reference_image)

    with av.open(os.fspath(video_file)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        # Stop bisecting below half a nominal frame: no other frame fits there.
        min_gap = max(1, round(0.5 / float(stream.average_rate) / time_base))
        lo = round(known_dissimilar_time / time_base)

        def frame_from(target_pts: int):
            container.seek(target_pts, backward=True, any_frame=False, stream=stream)
            for frame in container.decode(stream):
                if frame.pts is not None and frame.pts >= target_pts:
                    return frame.pts, frame.to_ndarray(format="bgr24")
            return None, None

        def distance_at(pts: int, image: np.ndarray) -> float:
            if distance_method_at_time is not None:
                return distance_method_at_time(pts * time_base, image)
            return distance_method(image, ref_img)

        def probe(target_pts: int):
            pts, image = frame_from(target_pts)
            if image is None:
                return None, None, False
            distance = distance_at(pts, image)
            if on_probe is not None:
                on_probe(pts * time_base, distance)
            return pts, image, distance > mse_threshold

        if known_similar_time is not None:
            target = round(known_similar_time / time_base)
            last_pts, last_image = frame_from(target)
            if last_pts is None or last_image is None or last_pts <= lo:
                return None, None
            distance = distance_at(last_pts, last_image)
            if on_probe is not None:
                on_probe(last_pts * time_base, distance)
            if distance > mse_threshold:
                return None, None
        else:
            # Last frame of the file: seek slightly before the end (seeking
            # past it yields nothing on some containers, e.g. MPEG-TS) and
            # widen the margin until frames decode, then keep the final one.
            stream_end_s = (
                float(container.duration) / av.time_base
                + (stream.start_time or 0) * time_base
            )
            last_pts, last_image = None, None
            for margin_s in (1, 2, 4, 8, 16, 32, 64):
                target = max(lo, round((stream_end_s - margin_s) / time_base))
                container.seek(target, backward=True, any_frame=False, stream=stream)
                for frame in container.decode(stream):
                    if frame.pts is not None:
                        last_pts, last_image = frame.pts, frame.to_ndarray(format="bgr24")
                if last_image is not None or target == lo:
                    break
            if last_image is None or last_pts < lo:
                return None, None
            distance = distance_at(last_pts, last_image)
            if on_probe is not None:
                on_probe(last_pts * time_base, distance)
            if distance > mse_threshold:
                return last_pts * time_base, last_image
            if last_pts == lo:
                return None, None

        lo_pts, lo_image = frame_from(lo)
        hi = last_pts  # exclusive search ceiling; frames at/after it are similar
        # Invariant: lo is dissimilar; everything from hi on is similar.
        while hi - lo > min_gap:
            mid = (lo + hi) // 2
            pts, image, dissimilar = probe(mid)
            if image is None or pts >= hi:
                hi = mid
            elif dissimilar:
                lo, lo_image = pts, image
            else:
                hi = pts
        return lo * time_base, lo_image




# def get_image_iterator_from_file(filename=sys.argv[1]):
#     packet_iterator = get_packet_iterator_from_file(filename)
#     frame_iterator = iterate_frames_from_packet_stream(packet_iterator)
#     image_iterator = get_frames_from_iterator(frame_iterator)
#     return image_iterator



def read_save_frame_from_file(filename=sys.argv[1]):

    frame_iterator = get_frame_iterator_from_file(filename)
    image_iterator = get_frames_from_iterator(frame_iterator)

    with WindowManager('Frame Window'):
        packet_count = 0
        frame_count = 0

        for frame_no, frame_bgr in image_iterator:

                cv2.putText(frame_bgr, f"Frame {frame_no}/{frame_count}:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Frame Window', frame_bgr)

                # if frame_count >= 6:
                #     # Save the current frame to disk
                #     output_filename = f"saved_frame_{frame_count}.jpg"
                #     cv2.imwrite(output_filename, frame_bgr)
                #     print(f"Saved frame {frame_count} to {output_filename}")
                #     break  # Exit after saving the frame

                cv2.waitKey(1)

                if cv2.getWindowProperty('Frame Window', cv2.WND_PROP_VISIBLE) <= 0:
                    raise WindowClosed("Window was closed by user")
                frame_count += 1




if __name__ == "__main__":
    # read_save_frame_from_file() 

    #  frame_no, frame_bgr = find_first_dissimilar_frame(sys.argv[1], 'saved_frame_6.jpg', min_frame=6, mse_threshold=1000.0)
    #  if frame_no is not None:
    #     print(f"First dissimilar frame found: Frame {frame_no}")
    #     cv2.imshow('Dissimilar Frame', frame_bgr)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


    print("Looking for the last proper frame before noise/black frames start...")
    frame_no, frame_bgr = find_last_proper_video_frame(sys.argv[1], mean_threshold=0.0, local_entropy_threshold=500, debug_csv="frame_metrics.csv")
    if frame_no is not None:
        print(f"Last non-noise frame found: Frame {frame_no}")
        cv2.imshow('Dissimilar Frame', frame_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
