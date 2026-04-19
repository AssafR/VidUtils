import cv2
import sys
from save_frame_from_file import iter_frame_metrics_rows
from streaming_regime import (
    FinalRegimeBreakDetector,
    iter_metrics_with_jump_ratios,
    make_row_regime_break_detector,
)
from utils import get_frames_estimate
from vidfile_iterator import get_frame_iterator_from_file, get_frames_from_iterator
import pathlib
from tqdm import tqdm
from display_utils import display_thumbnails_from_frames
from display_utils import display_frame_window_around_index


def locate_noise(video_path):
    total_frames, fps = get_frames_estimate(video_path)

    frame_iterator = get_frame_iterator_from_file(video_path)
    frames = get_frames_from_iterator(frame_iterator)
    
    # sum=0
    # for frame_no, frame_bgr in EtaTqdm(image_iterator, total=frames, desc="Processing frames"):

    # 1) Compute frame-level metrics (no inner tqdm here).
    rows = iter_frame_metrics_rows(frames, total_frames, fps, use_tqdm=False)

    # 2) Enrich rows with short/long stats + jump ratio on laplacian_variance.
    enriched_rows = iter_metrics_with_jump_ratios(
        rows, 
        numerator_mode="percentile",
        percentile_q=0.75,
        short_name="lap_short", 
        long_name="lap_long", 
        jump_name="lap_jump_ratio"
        )

    # 3) Track the last upward crossing of a jump-ratio threshold on those rows.
    det = make_row_regime_break_detector(threshold=1.5, value_key="lap_jump_ratio", frame_key="frame")

    pbar = tqdm(enriched_rows, total=total_frames, desc="Jump ratio")
    for row in pbar:
        det.consume(row)
        pbar.set_description(f"Frame {row['frame']} - Jump Ratio: {row['lap_jump_ratio']:.3f}")
    return det.result()

if __name__ == "__main__":
    # read_save_frame_from_file() 

    #  frame_no, frame_bgr = find_first_dissimilar_frame(sys.argv[1], 'saved_frame_6.jpg', min_frame=6, mse_threshold=1000.0)
    #  if frame_no is not None:
    #     print(f"First dissimilar frame found: Frame {frame_no}")
    #     cv2.imshow('Dissimilar Frame', frame_bgr)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    print("sys.executable:", sys.executable)
    print("sys.version:", sys.version)
    print("cwd:", pathlib.Path().resolve())


    print("Looking for the last proper frame before noise/black frames start...")

    last_break = locate_noise(sys.argv[1])
    if last_break is not None:
        print(f"Last upward cross: {last_break.index}, {last_break.value:.3f}, {last_break.x:.3f}")
    else:
        print("No crossing under current rule")

    # Display the last 10 frames before the last upward cross in a grid using display_frame_window_around_index
    display_frame_window_around_index(sys.argv[1], last_break.index, k=48, thumbs_per_row=12)

    # display_frame_window_around_index(sys.argv[1], 270, k=24, thumbs_per_row=10)

