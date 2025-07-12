#!/usr/bin/env python3
"""
Debug script to see what's happening with packet filtering and frame decoding.
"""

import sys
from vidfile_iterator import *

def debug_packet_filtering(filename: str):
    """Debug the packet filtering process step by step."""
    print("=== DEBUGGING PACKET FILTERING ===")
    
    # Create the frame iterator
    frame_iterator = FileFrameIterator(filename)
    
    # Test the first filter
    max_packet_size = 2000
    print(f"Filtering packets smaller than {max_packet_size} bytes...")
    
    # Count packets in original stream
    original_count = 0
    for packet_data in frame_iterator.iterator:
        original_count += 1
        if original_count >= 20:  # Only check first 20
            break
    
    print(f"Original stream has at least {original_count} packets")
    
    # Test first filter
    frame_iterator = FileFrameIterator(filename)  # Reset
    filtered_iterator = filter_stream_preserve_consecutivity(
        frame_iterator.iterator, 
        lambda x: filter_small_packets(x, max_packet_size)
    )
    
    first_filter_count = 0
    for it, pd_it in enumerate(filtered_iterator):
        group_packets = list(pd_it)
        print(f"First filter - Group {it}: {len(group_packets)} packets")
        if group_packets:
            packet_no, packet = group_packets[0]
            print(f"  First packet: {packet_no}, size: {packet.size}")
        first_filter_count += 1
        if first_filter_count >= 5:
            break
    
    print(f"First filter created {first_filter_count} groups")
    
    # Test second filter
    frame_iterator = FileFrameIterator(filename)  # Reset
    filtered_iterator = filter_stream_preserve_consecutivity(
        frame_iterator.iterator, 
        lambda x: filter_small_packets(x, max_packet_size)
    )
    filtered_iterator_2 = filter_stream_preserve_consecutivity(
        filtered_iterator, 
        lambda x: x[0] < 100
    )
    
    second_filter_count = 0
    for it, pd_it in enumerate(filtered_iterator_2):
        group_packets = list(pd_it)
        print(f"Second filter - Group {it}: {len(group_packets)} packets")
        if group_packets:
            packet_no, packet = group_packets[0]
            print(f"  First packet: {packet_no}, size: {packet.size}")
        second_filter_count += 1
        if second_filter_count >= 5:
            break
    
    print(f"Second filter created {second_filter_count} groups")

def debug_frame_decoding(filename: str):
    """Debug frame decoding with a simple packet."""
    print("\n=== DEBUGGING FRAME DECODING ===")
    
    # Get a simple packet and try to decode it
    frame_iterator = FileFrameIterator(filename)
    
    for packet_data in frame_iterator.iterator:
        packet_no, packet = packet_data
        print(f"Testing packet {packet_no}, size: {packet.size}")
        
        # Try different decoding approaches
        print("  Trying decode_packet_to_frames:")
        frames1 = decode_packet_to_frames(packet_data)
        print(f"    Result: {len(frames1)} frames")
        
        print("  Trying decode_packet_to_frames_with_state:")
        frames2, _ = decode_packet_to_frames_with_state(packet_data)
        print(f"    Result: {len(frames2)} frames")
        
        print("  Trying decode_all_packets_with_flush:")
        for p_no, p_frames in decode_all_packets_with_flush([packet_data]):
            print(f"    Packet {p_no}: {len(p_frames)} frames")
        
        # Only test first few packets
        if packet_no >= 5:
            break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_test.py <video_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    debug_packet_filtering(filename)
    debug_frame_decoding(filename) 