import sys
from display_utils import display_thumbnails_from_frames
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
    
    # Apply multiple filters in a chain
    print("Testing filter chaining approach:")
    print("Note: Using streaming approach that doesn't store packets in memory")
    
    # Example 1: Size-based filtering (small packets between 500-10000 bytes)
    filtered_stream_1 = chain_filters(
        frame_iterator.iterator,
        lambda x: filter_small_packets(x, 1000),  # Max 10KB
        lambda x: filter_large_packets(x, 0)     # Min 500B
    )
    
    # Example 2: Size + PTS range filtering
    filtered_stream_2 = chain_filters(
        frame_iterator.iterator,
        lambda x: filter_small_packets(x, 200),   # Max 8KB
        lambda x: filter_by_pts_range(x, 1000, 20000)  # PTS between 1000-20000
    )
    
    # Use the first filter chain for demonstration with streaming approach
    print("Using streaming approach (no memory storage of packets)...")
    frame_groups = list(group_packets_by_pts_and_decode_streaming(filename, filtered_stream_1, lambda x: True))  # No additional filtering needed
    
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
        # from display_utils import display_thumbnails_from_frames
        
        # print("\n=== Testing First 10 Frames Sampling ===")
        # display_thumbnails_from_frames(frame_groups, sampling_strategy="first")
        
        print("\n=== Testing Bookend Sampling (First 5 + Last 5) ===")
        display_thumbnails_from_frames(frame_groups, sampling_strategy="bookend")
        
        print("\n=== Testing Random Sampling (10 random frames) ===")
        display_thumbnails_from_frames(frame_groups, sampling_strategy="random")
    else:
        print("No frame groups to display")


if __name__ == "__main__":
    main()
