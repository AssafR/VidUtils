import cv2
import sys
from save_frame_from_file import iter_frame_metrics_rows
from streaming_regime import (
    notebook_aligned_jump_ratio_processor,
    FinalRegimeBreakDetector,
    run_jump_ratio_on_metric_rows,
)
from utils import get_frames_estimate
from vidfile_iterator import get_frame_iterator_from_file, get_frames_from_iterator
import pathlib

def locate_noise(video_path):
    total_frames, fps = get_frames_estimate(video_path)

    frame_iterator = get_frame_iterator_from_file(video_path)
    frames = get_frames_from_iterator(frame_iterator)
    
    # sum=0
    # for frame_no, frame_bgr in EtaTqdm(image_iterator, total=frames, desc="Processing frames"):

    proc = notebook_aligned_jump_ratio_processor(ewm_span=50, rolling_window=50, epsilon=1e-5)
    det = FinalRegimeBreakDetector(1.5)  # threshold on jump_ratio; tune for your data
    # frames = get_image_iterator_from_file(video_path)
    rows = iter_frame_metrics_rows(frames, total_frames, fps)
    # rows_df = pd.DataFrame(rows)
    steps, last_break = run_jump_ratio_on_metric_rows(rows, processor=proc, detector=det)
    return last_break

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
        print("Last upward cross:", last_break.index, last_break.value, last_break.x)
    else:
        print("No crossing under current rule")

