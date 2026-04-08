from fileinput import filename
import sys
import cv2
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy as scipy_entropy
from utils import WindowClosed, WindowManager
from vidfile_iterator import (FileFrameIterator, iterate_frames_from_packet_stream, 
                              stream_frames_from_packet, decode_packet_to_frames,
                             )

RESIZE_DIM_FOR_MSE = (64, 64)  # Resize dimensions for MSE calculation


def _compute_entropy(data):
    """
    Compute Shannon entropy of 2D or 3D image data (grayscale or multi-channel).
    For multi-channel, computes entropy per channel and returns the average.
    
    Args:
        data: 2D array (grayscale) or 3D array (multi-channel)
    
    Returns:
        Shannon entropy value
    """
    if data is None or data.size == 0:
        return 0.0
    
    # Handle multi-channel images
    if len(data.shape) == 3:
        # Compute entropy for each channel and average
        entropies = []
        for channel in range(data.shape[2]):
            channel_data = data[:, :, channel].flatten()
            hist = np.histogram(channel_data, bins=256, range=(0, 256))[0]
            hist = hist / hist.sum()
            hist = hist[hist > 0]  # Avoid log(0)
            entropies.append(scipy_entropy(hist, base=2))
        return np.mean(entropies)
    else:
        # Grayscale image
        flat_data = data.flatten()
        hist = np.histogram(flat_data, bins=256, range=(0, 256))[0]
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Avoid log(0)
        return scipy_entropy(hist, base=2)


def calculate_local_entropy(gray, blur_ksize=5):
    """Calculate entropy of the local high-frequency residual around the frame."""
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    residual = cv2.absdiff(gray, blur)
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
    image_iterator = get_image_iterator_from_file(video_file)
    
    # Estimate total frames from video metadata (may be inaccurate)
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else None
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 25.0  # Default FPS if unavailable
    cap.release()
    
    last_proper_frame_no = None
    last_proper_frame_bgr = None
    previous_bgr = None
    
    print("Looking for the last proper frame before noise/black frames start...")
    print("Mean threshold:", mean_threshold)
    print("Local entropy threshold:", local_entropy_threshold)
    if debug_csv:
        print(f"Saving frame metrics to: {debug_csv}")

    metrics_log = []

    with tqdm(image_iterator, total=total_frames, desc="Processing frames") as pbar:
        for frame_no, frame_bgr in pbar:
            timestamp = frame_no / fps if fps > 0 else 0

            mean_val = np.mean(frame_bgr)
            local_entropy_val = calculate_local_entropy(frame_bgr)
            temporal_entropy_val = calculate_temporal_entropy(frame_bgr, previous_bgr)
            combined_entropy = 0.7 * local_entropy_val + 0.3 * temporal_entropy_val
            
            # For edge density, convert to grayscale
            gray_for_edges = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if len(frame_bgr.shape) == 3 else frame_bgr
            edge_density = calculate_edge_density(gray_for_edges)
            hist_entropy = calculate_histogram_entropy(frame_bgr)
            laplacian_var = calculate_laplacian_variance(gray_for_edges)

            pbar.set_postfix({'frame': frame_no, 'time': f"{timestamp:.2f}s", 
                              'mean': f"{mean_val:.1f}", 
                              'local_ent': f"{local_entropy_val:.2f}", 'temp_ent': f"{temporal_entropy_val:.2f}", 
                              'combined': f"{combined_entropy:.2f}", 
                              'edges': f"{edge_density:.3f}", 'lap_var': f"{laplacian_var:.1f}"})
            

            if debug_csv:
                metrics_log.append({
                    'frame': frame_no,
                    'timestamp': timestamp,
                    'mean': mean_val,
                    'local_entropy': local_entropy_val,
                    'temporal_entropy': temporal_entropy_val,
                    'combined_entropy': combined_entropy,
                    'edge_density': edge_density,
                    'hist_entropy': hist_entropy,
                    'laplacian_variance': laplacian_var
                })

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


def mse(image1, image2):
    """Calculate Mean Squared Error between two images."""
    return np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)

def calculate_mse_between_frames(frame_bgr, ref_img):
        # Convert frame to grayscale
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Convert reference to grayscale for comparison
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    
    # Resize both images to the same size before extracting central region
        target_shape = ref_gray.shape
        if frame_gray.shape != target_shape:
            frame_gray = cv2.resize(frame_gray, (target_shape[1], target_shape[0]))
        if ref_gray.shape != target_shape:
            ref_gray = cv2.resize(ref_gray, (target_shape[1], target_shape[0]))

        # Take only the central region of the frame for comparison (optional)
        h, w = target_shape
        center_region = frame_gray[h//4:3*h//4, w//4:3*w//4]
        # Take only the central region of the reference image for comparison (optional)
        ref_center_region = ref_gray[h//4:3*h//4, w//4:3*w//4]
        # Take only the central region of the frame for comparison (optional)
        h, w = frame_gray.shape   
        center_region = frame_gray[h//4:3*h//4, w//4:3*w//4]
        # Take only the central region of the reference image for comparison (optional)
        ref_center_region = ref_gray[h//4:3*h//4, w//4:3*w//4]


        # Resize both images to a smaller size for faster MSE calculation (optional)
        frame_gray_small = cv2.resize(center_region, RESIZE_DIM_FOR_MSE)
        ref_gray_small = cv2.resize(ref_center_region, RESIZE_DIM_FOR_MSE)   

        
        # Calculate MSE
        mse_value = mse(ref_gray_small, frame_gray_small)

        return mse_value



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


def get_packet_iterator_from_file(filename):
    """Test the modern decoding approach in vidfile_iterator.py"""
    if len(sys.argv) < 2:
        print("Usage: python test_vidfile_iterator.py <video_file>")
        sys.exit(1)
    
    print(f"Reading file: {filename}")
    
    # Create the frame iterator
    frame_iterator = FileFrameIterator(filename)    
    return frame_iterator.packet_iterator


def get_frame_iterator_from_file(filename):
    """Test the modern decoding approach in vidfile_iterator.py"""
    if len(sys.argv) < 2:
        print("Usage: python test_vidfile_iterator.py <video_file>")
        sys.exit(1)
    
    print(f"Testing modern decoding in vidfile_iterator.py with: {filename}")
    
    # Create the frame iterator
    frame_iterator = FileFrameIterator(filename)    
    return frame_iterator.frame_iterator

def get_frames_from_iterator(frame_iterator, max_frames=10):
    """Helper function to get frames from the frame iterator"""
    frames = []
    frame_count = 0
    
    for frame_data in frame_iterator:
        packet_no, frame_no, frame = frame_data
        frame_img = frame.to_image()
        frame_np = np.array(frame_img)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)    
        yield frame_no, frame_bgr
    
    return 


def get_image_iterator_from_file(filename=sys.argv[1]):
    packet_iterator = get_packet_iterator_from_file(filename)
    frame_iterator = iterate_frames_from_packet_stream(packet_iterator)
    image_iterator = get_frames_from_iterator(frame_iterator)
    return image_iterator



def read_save_frame_from_file(filename=sys.argv[1]):

    image_iterator = get_image_iterator_from_file(filename)

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
