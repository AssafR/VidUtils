import sys
from vidfile_iterator import FileFrameIterator, stream_frames_from_packet, decode_packet_to_frames, flush_decoder
import av

def test_modern_decoding():
    """Test the modern decoding approach in vidfile_iterator.py"""
    if len(sys.argv) < 2:
        print("Usage: python test_vidfile_iterator.py <video_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    print(f"Testing modern decoding in vidfile_iterator.py with: {filename}")
    
    # Create the frame iterator
    frame_iterator = FileFrameIterator(filename)
    
    # Test the stream_frames_from_packet function
    print("\n=== Testing stream_frames_from_packet ===")
    packet_count = 0
    max_packets_to_test = 3
    
    for packet_data in frame_iterator.iterator:
        stream_frames_from_packet(packet_data)
        packet_count += 1
        if packet_count >= max_packets_to_test:
            break
    
    # Test the decode_packet_to_frames function
    print("\n=== Testing decode_packet_to_frames ===")
    frame_iterator = FileFrameIterator(filename)  # Reset iterator
    packet_count = 0
    
    for packet_data in frame_iterator.iterator:
        packet_no, packet = packet_data
        print(f"  Decoding packet {packet_no}...")
        frames = decode_packet_to_frames(packet_data)
        print(f"  Got {len(frames)} frames from packet {packet_no}")
        
        for i, frame in enumerate(frames):
            print(f"    Frame {i}: PTS={frame.pts}, time_base={frame.time_base}")
        
        packet_count += 1
        if packet_count >= max_packets_to_test:
            break
    
    # Test flushing the decoder
    print("\n=== Testing decoder flush ===")
    container = av.open(filename)
    video_stream = container.streams.video[0]
    codec_context = video_stream.codec_context
    
    # Process a few packets first
    frame_iterator = FileFrameIterator(filename)
    packet_count = 0
    for packet_data in frame_iterator.iterator:
        packet_no, packet = packet_data
        # Use decode instead of send for PyAV 14.4.0
        try:
            frames = list(codec_context.decode(packet))
            print(f"  Processed packet {packet_no}, got {len(frames)} frames")
        except Exception as e:
            print(f"  Error processing packet {packet_no}: {e}")
        packet_count += 1
        if packet_count >= 5:  # Process 5 packets
            break
    
    # Now flush to get remaining frames
    print("  Flushing decoder...")
    remaining_frames = flush_decoder(codec_context)
    print(f"  Got {len(remaining_frames)} remaining frames from flush")
    
    for i, frame in enumerate(remaining_frames):
        print(f"    Flushed frame {i}: PTS={frame.pts}")
    
    print(f"\nAll tests completed successfully!")

if __name__ == "__main__":
    test_modern_decoding() 