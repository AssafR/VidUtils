import sys
from display_utils import display_thumbnails_stream 
from vidfile_iterator import *
import av
import av.container, av.packet, av.stream
import itertools



def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <video_file>")
        sys.exit(1)
    filename = sys.argv[1]
    print(f"Analyzing: {filename}")
    
    # Create the frame iterator
    frame_iterator = FileFrameIterator(filename)
    
    # Apply filter to only get packets smaller than 50000 bytes
    max_packet_size = 10000  # Increased from 2000 to get more frames

    # Use seek-and-decode approach with PTS
    print("Testing seek-and-decode approach with PTS:")
    frame_groups = list(group_packets_by_pts_and_decode(filename, frame_iterator.iterator, lambda x: filter_small_packets(x, max_packet_size)))
    
    print(f"\n=== DETAILED FRAME GROUP ANALYSIS ===")
    print(f"Total frame groups: {len(frame_groups)}")
    for i, frames in enumerate(frame_groups):
        print(f"Group {i}: {len(frames)} frames")
        if frames:
            print(f"  Frame PTS values: {[frame.pts for frame in frames[:5]]}...")  # Show first 5 frame PTS
        if i >= 10:  # Limit output
            print("... (more groups omitted)")
            break
    
    # Display thumbnails from the frame groups
    if frame_groups:
        from display_utils import display_thumbnails_from_frames
        display_thumbnails_from_frames(frame_groups)
    else:
        print("No frame groups to display")


if __name__ == "__main__":
    main()
