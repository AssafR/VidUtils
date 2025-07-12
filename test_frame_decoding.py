#!/usr/bin/env python3
"""
Test script to demonstrate the difference between individual packet decoding
and flush-based decoding approaches.
"""

from vidfile_iterator import (
    FileFrameIterator, 
    decode_packet_to_frames, 
    decode_packet_to_frames_with_state,
    decode_all_packets_with_flush,
    packet_data_type
)
from typing import Iterator

def test_individual_packet_decoding(filename: str, max_packets: int = 10):
    """
    Test individual packet decoding - this often returns 0 frames.
    """
    print("Testing Individual Packet Decoding:")
    print("=" * 50)
    
    file_iterator = FileFrameIterator(filename)
    packet_count = 0
    total_frames = 0
    
    for packet_data in file_iterator.iterator:
        if packet_count >= max_packets:
            break
            
        packet_no, packet = packet_data
        frames = decode_packet_to_frames(packet_data)
        
        print(f"Packet {packet_no}: {len(frames)} frames")
        total_frames += len(frames)
        packet_count += 1
    
    print(f"\nTotal: {packet_count} packets, {total_frames} frames")
    print(f"Average frames per packet: {total_frames/packet_count:.2f}")

def test_individual_packet_decoding_with_state(filename: str, max_packets: int = 10):
    """
    Test individual packet decoding with state maintenance.
    """
    print("\nTesting Individual Packet Decoding with State:")
    print("=" * 50)
    
    file_iterator = FileFrameIterator(filename)
    packet_count = 0
    total_frames = 0
    codec_context = None
    
    for packet_data in file_iterator.iterator:
        if packet_count >= max_packets:
            break
            
        packet_no, packet = packet_data
        frames, codec_context = decode_packet_to_frames_with_state(packet_data, codec_context)
        
        print(f"Packet {packet_no}: {len(frames)} frames")
        total_frames += len(frames)
        packet_count += 1
    
    print(f"\nTotal: {packet_count} packets, {total_frames} frames")
    print(f"Average frames per packet: {total_frames/packet_count:.2f}")

def test_flush_based_decoding(filename: str, max_packets: int = 10):
    """
    Test flush-based decoding - this is the recommended approach.
    """
    print("\nTesting Flush-Based Decoding:")
    print("=" * 50)
    
    file_iterator = FileFrameIterator(filename)
    packet_count = 0
    total_frames = 0
    
    for packet_no, frames in decode_all_packets_with_flush(file_iterator.iterator, max_packets):
        print(f"Packet {packet_no}: {len(frames)} frames")
        total_frames += len(frames)
        packet_count += 1
    
    print(f"\nTotal: {packet_count} packets, {total_frames} frames")
    print(f"Average frames per packet: {total_frames/packet_count:.2f}")

def explain_why_individual_decoding_fails():
    """
    Explain why individual packet decoding often returns 0 frames.
    """
    print("\n" + "=" * 60)
    print("Why Individual Packet Decoding Often Returns 0 Frames:")
    print("=" * 60)
    
    print("""
1. **Decoder State Management**:
   - Video decoders maintain internal state
   - Each packet depends on previous packets for reference frames
   - Individual decoding resets or loses this state

2. **Inter-frame Compression**:
   - I-frames (keyframes): Can be decoded independently
   - P-frames: Depend on previous frames
   - B-frames: Depend on both previous and future frames
   - Individual decoding breaks these dependencies

3. **Packet vs Frame Relationship**:
   - Not every packet contains a complete frame
   - Some packets contain partial frame data
   - Some packets contain metadata or other non-frame data

4. **Codec-specific Behavior**:
   - Different codecs (H.264, H.265, etc.) have different packet structures
   - Some codecs require multiple packets to decode a single frame
   - Individual decoding doesn't handle these complexities

**Solution**: Use flush-based decoding (decode_all_packets_with_flush) 
which maintains proper decoder state and handles all frame dependencies.
    """)

if __name__ == "__main__":
    # You would need to provide a video file path here
    # filename = "your_video_file.mp4"
    
    print("Frame Decoding Comparison Test")
    print("=" * 60)
    print("Note: This test requires a video file. Please modify the filename variable.")
    
    explain_why_individual_decoding_fails()
    
    print("\nTo run the actual tests, uncomment and modify the filename:")
    print("# filename = 'your_video_file.mp4'")
    print("# test_individual_packet_decoding(filename, 10)")
    print("# test_individual_packet_decoding_with_state(filename, 10)")
    print("# test_flush_based_decoding(filename, 10)") 