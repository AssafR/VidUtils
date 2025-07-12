import sys
from vidfile_iterator import FileFrameIterator, decode_packet_to_frames, flush_decoder
import av

def test_flush_timing():
    """Test to show when flushing is needed vs when it's not an issue"""
    if len(sys.argv) < 2:
        print("Usage: python test_flush_timing.py <video_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    print(f"Testing flush timing: {filename}")
    
    # Create the frame iterator
    frame_iterator = FileFrameIterator(filename)
    
    # Test 1: Process first 5 packets without flush
    print("\n=== TEST 1: First 5 packets (no flush needed) ===")
    packet_count = 0
    total_frames = 0
    codec_context = None
    
    for packet_data in frame_iterator.iterator:
        packet_no, packet = packet_data
        
        # Get codec context from first packet
        if codec_context is None:
            codec_context = packet.stream.codec_context
        
        frames = decode_packet_to_frames(packet_data)
        print(f"  Packet {packet_no}: {len(frames)} frames")
        total_frames += len(frames)
        
        packet_count += 1
        if packet_count >= 5:
            break
    
    print(f"  Total frames from first 5 packets: {total_frames}")
    
    # Test 2: Process next 5 packets without flush
    print("\n=== TEST 2: Next 5 packets (no flush needed) ===")
    packet_count = 0
    total_frames = 0
    
    for packet_data in frame_iterator.iterator:
        packet_no, packet = packet_data
        
        frames = decode_packet_to_frames(packet_data)
        print(f"  Packet {packet_no}: {len(frames)} frames")
        total_frames += len(frames)
        
        packet_count += 1
        if packet_count >= 5:
            break
    
    print(f"  Total frames from next 5 packets: {total_frames}")
    
    # Test 3: Now flush to see what was buffered
    print("\n=== TEST 3: Flushing after 10 packets ===")
    remaining_frames = flush_decoder(codec_context)
    print(f"  Flushed frames: {len(remaining_frames)}")
    for i, frame in enumerate(remaining_frames):
        print(f"    Flushed frame {i}: PTS={frame.pts}")
    
    # Test 4: Continue processing after flush
    print("\n=== TEST 4: Continue processing after flush ===")
    packet_count = 0
    total_frames = 0
    
    for packet_data in frame_iterator.iterator:
        packet_no, packet = packet_data
        
        frames = decode_packet_to_frames(packet_data)
        print(f"  Packet {packet_no}: {len(frames)} frames")
        total_frames += len(frames)
        
        packet_count += 1
        if packet_count >= 3:
            break
    
    print(f"  Total frames from next 3 packets: {total_frames}")
    
    # Test 5: Final flush
    print("\n=== TEST 5: Final flush ===")
    remaining_frames = flush_decoder(codec_context)
    print(f"  Final flushed frames: {len(remaining_frames)}")
    for i, frame in enumerate(remaining_frames):
        print(f"    Final flushed frame {i}: PTS={frame.pts}")

if __name__ == "__main__":
    test_flush_timing() 