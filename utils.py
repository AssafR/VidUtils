import cv2
import numpy as np
from scipy.stats import entropy as scipy_entropy
import sys
from vidfile_iterator import FileFrameIterator
from tqdm import tqdm
from datetime import timedelta
from collections import OrderedDict

RESIZE_DIM_FOR_MSE = (64, 64)  # Resize dimensions for MSE calculation

class WindowManager:
    def __init__(self, window_name='Frame Window'):
        self.window_name = window_name

    def __enter__(self):
        cv2.namedWindow(self.window_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        while cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) > 0:
            if cv2.waitKey(1) != -1:
                break
        cv2.destroyAllWindows()


class WindowClosed(Exception):
    pass


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

import numpy as np

def _compute_entropy(data: np.ndarray) -> float:
    """
    Shannon entropy for 2D (H,W) or 3D (H,W,C) uint8 images.
    For multi-channel, computes entropy per channel and returns the average.
    """
    if data is None or data.size == 0:
        return 0.0

    if data.ndim == 2:  # grayscale -> treat as (H, W, 1)
        data = data[..., None]
    elif data.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")

    h, w, c = data.shape
    flat = data.reshape(-1, c)

    # Ensure integer type for bincount
    flat = flat.astype(np.int64, copy=False)

    entropies = np.empty(c, dtype=float)
    for ch in range(c):
        counts = np.bincount(flat[:, ch], minlength=256).astype(float)
        total = counts.sum()
        if total == 0:
            entropies[ch] = 0.0
            continue
        p = counts / total
        p = p[p > 0.0]
        entropies[ch] = -np.sum(p * np.log2(p))

    return float(entropies.mean())

def get_frames_estimate(video_file):
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else None
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 25.0  # Default FPS if unavailable
    cap.release()
    return total_frames, fps



class EtaTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eta_str: str | None = None
        self._user_postfix = OrderedDict()

    def _compute_eta(self):
        d = self.format_dict
        n = d.get("n")
        total = d.get("total")
        elapsed = d.get("elapsed")
        if n and total and elapsed and n > 0:
            remaining_seconds = elapsed * (total / n - 1)
            self._eta_str = str(timedelta(seconds=int(remaining_seconds)))
        else:
            self._eta_str = None

    def _combined_postfix(self):
        # if user explicitly set 'eta' or no ETA available, just use theirs
        if "eta" in self._user_postfix or self._eta_str is None:
            return self._user_postfix
        combined = OrderedDict( # Pushes the ETA to the front of the dictionary
            [("eta", self._eta_str), *self._user_postfix.items()]
        )
        return combined
        
    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        ordered_dict = ordered_dict or {}
        # preserves ordered_dict order, then appends kwargs in order
        self._user_postfix = OrderedDict(ordered_dict, **kwargs)
        return super().set_postfix(self._combined_postfix(), refresh=refresh)

    def __iter__(self):
        for item in super().__iter__():
            self._compute_eta()
            super().set_postfix(self._combined_postfix(), refresh=False)
            yield item