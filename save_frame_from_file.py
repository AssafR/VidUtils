import sys
from typing import Any, Iterator
import cv2
import numpy as np
from tqdm import tqdm
from utils import WindowClosed, WindowManager, get_frames_estimate
from vidfile_iterator import (FileFrameIterator, iterate_frames_from_packet_stream, 
                              stream_frames_from_packet, decode_packet_to_frames,
                              get_frame_iterator_from_file, get_frames_from_iterator)
from utils import calculate_mse_between_frames, _compute_entropy, EtaTqdm


RESIZE_DIM_FOR_MSE = (64, 64)  # Resize dimensions for MSE calculation

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


def calculate_edge_density(gray, threshold=50):
    """Calculate the proportion of edges/high-frequency content in the frame."""
    edges = cv2.Canny(gray, threshold//2, threshold)
    edge_pixels = np.count_nonzero(edges)
    total_pixels = edges.size
    return edge_pixels / total_pixels

def calculate_histogram_entropy(gray):
    """Calculate entropy of the raw pixel value histogram (measures image variability)."""
    return _compute_entropy(gray)


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
    # "temporal_entropy": lambda **kwargs: calculate_temporal_entropy(kwargs["current_gray"], kwargs["previous_gray"]),
    # "edge_density": lambda **kwargs: calculate_edge_density(kwargs["gray_for_edges"]),
    # "hist_entropy": lambda **kwargs: calculate_histogram_entropy(kwargs["gray_for_edges"]),
    "laplacian_variance": lambda **kwargs: calculate_laplacian_variance(kwargs["gray_for_edges"]),
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




def find_first_dissimilar_frame(video_file, reference_image_path, min_frame=0, mse_threshold=1000.0):
    """
    Pipeline to find the first frame after min_frame that is not similar to the reference image.
    
    Args:
        video_file: Path to the video file
        reference_image_path: Path to the reference image (e.g., 'saved_frame_6.jpg')
        min_frame: Minimum frame number to start checking from
        mse_threshold: MSE threshold below which images are considered similar
    
    Returns:
        Tuple of (frame_no, frame_image) for the first dissimilar frame, or (None, None) if none found
    """
    # Load reference image
    ref_img = cv2.imread(reference_image_path)
    if ref_img is None:
        raise ValueError(f"Could not load reference image: {reference_image_path}")
    
    image_iterator = get_image_iterator_from_file(video_file)
    
    for frame_no, frame_bgr in image_iterator:
        if frame_no <= min_frame:
            continue
        
        mse_value = calculate_mse_between_frames(frame_bgr, ref_img)
        # print(f"Frame {frame_no}: MSE={mse_value:.2f}")
        if mse_value > mse_threshold:  # Not similar
            return frame_no, frame_bgr
    
    return None, None  # No dissimilar frame found




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
