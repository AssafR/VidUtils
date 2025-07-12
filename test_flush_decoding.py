import sys
from vidfile_iterator import FileFrameIterator, decode_all_packets_with_flush, filter_small_packets
import av

def test_proper_decoding_with_flush():
    """Test the proper decoding approach with flushing"""
    if len(sys.argv) < 2:
        print("Usage: python test_flush_decoding.py <video_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    print(f"Testing proper decoding with flush: {filename}")
    
    # Create the frame iterator
    frame_iterator = FileFrameIterator(filename)
    
    # Apply filter to only get packets smaller than 50000 bytes
    max_packet_size = 50000
    filtered_iterator = filter(lambda x: filter_small_packets(x, max_packet_size), frame_iterator.iterator)
    
    print(f"Filtering packets smaller than {max_packet_size} bytes...")
    
    # Use the proper decoding function that handles flushing
    total_frames = 0
    packet_count = 0
    max_packets_to_test = 10  # Limit for testing
    
    for packet_no, frames in decode_all_packets_with_flush(filtered_iterator, max_packets_to_test):
        if packet_no == -1:
            # These are flushed frames
            print(f"\n=== FLUSHED FRAMES ===")
            for i, frame in enumerate(frames):
                print(f"  Flushed frame {i}: PTS={frame.pts}")
            total_frames += len(frames)
        else:
            # These are regular packet frames
            print(f"\n  Packet {packet_no}: {len(frames)} frames")
            for i, frame in enumerate(frames):
                print(f"    Frame {i}: PTS={frame.pts}")
            total_frames += len(frames)
            packet_count += 1
    
    print(f"\n=== SUMMARY ===")
    print(f"Processed {packet_count} packets")
    print(f"Total frames: {total_frames}")
    print(f"Average frames per packet: {total_frames / packet_count if packet_count > 0 else 0:.2f}")

if __name__ == "__main__":
    test_proper_decoding_with_flush() 