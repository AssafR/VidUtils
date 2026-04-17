import sys
from vidfile_iterator import (
    stream_frames_from_packet, 
    decode_packet_to_frames, flush_decoder, 
    get_packet_iterator_from_file, iterate_frames_from_packet_stream, 
    get_frames_from_iterator)
from utils import get_frames_estimate, EtaTqdm
import av
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from vidfile_iterator import get_frame_iterator_from_file

def test_modern_decoding():
    """Test the modern decoding approach in vidfile_iterator.py"""
    if len(sys.argv) < 2:
        print("Usage: python test_vidfile_iterator.py <video_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    print(f"Testing modern decoding in vidfile_iterator.py with: {filename}")

    frames,fps = get_frames_estimate(filename)
    
    # packet_iterator = get_packet_iterator_from_file(filename)
    # frame_iterator = iterate_frames_from_packet_stream(packet_iterator)

    frame_iterator = get_frame_iterator_from_file(filename)
    image_iterator = get_frames_from_iterator(frame_iterator)
    
    sum=0
    for frame_no, frame_bgr in EtaTqdm(image_iterator, total=frames, desc="Processing frames"):
        # print(f"Frame {frame_no}: {frame_bgr.shape}")
        frame_bgr_mean = np.mean(frame_bgr)      
        sum += frame_bgr_mean   

    average = sum / frames
    print(f"Average frame brightness: {average}")

if __name__ == "__main__":
    test_modern_decoding() 