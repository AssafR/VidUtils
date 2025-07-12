#!/usr/bin/env python3
"""
Test script to demonstrate the filter_stream_preserve_consecutivity function.
"""

from vidfile_iterator import filter_stream_preserve_consecutivity, packet_data_type
from typing import Iterator
import av

def create_test_packet_stream() -> Iterator[packet_data_type]:
    """
    Create a test packet stream with some gaps in packet numbers.
    This simulates a real packet stream where some packets might be missing.
    """
    # Simulate packets with some gaps
    packet_numbers = [0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 15, 16, 17, 18, 20]
    
    for packet_no in packet_numbers:
        # Create a mock packet with size based on packet number
        # (in real usage, this would be actual av.packet.Packet objects)
        mock_packet = type('MockPacket', (), {
            'size': packet_no * 100,  # Size increases with packet number
            'stream': type('MockStream', (), {'codec_context': None})()
        })()
        
        yield (packet_no, mock_packet)

def test_consecutive_filter():
    """
    Test the filter_stream_preserve_consecutivity function.
    """
    print("Testing filter_stream_preserve_consecutivity function...")
    print("=" * 60)
    
    # Create test packet stream
    test_stream = create_test_packet_stream()
    
    # Define a filter function: keep packets with size < 1000
    def size_filter(packet_data: packet_data_type) -> bool:
        packet_no, packet = packet_data
        return packet.size < 1000
    
    # Apply the filter while preserving consecutivity
    filtered_streams = filter_stream_preserve_consecutivity(test_stream, size_filter)
    
    # Process each consecutive group
    for group_idx, consecutive_packets in enumerate(filtered_streams):
        print(f"\nConsecutive Group {group_idx + 1}:")
        print("-" * 30)
        
        packet_numbers = []
        for packet_no, packet in consecutive_packets:
            packet_numbers.append(packet_no)
            print(f"  Packet {packet_no}: size={packet.size}")
        
        print(f"  Total packets in group: {len(packet_numbers)}")
        print(f"  Packet numbers: {packet_numbers}")
        
        # Verify consecutivity
        if len(packet_numbers) > 1:
            is_consecutive = all(
                packet_numbers[i] + 1 == packet_numbers[i + 1] 
                for i in range(len(packet_numbers) - 1)
            )
            print(f"  Consecutive: {is_consecutive}")

def test_with_real_filter_example():
    """
    Show how to use the function with the existing filter_small_packets function.
    """
    print("\n" + "=" * 60)
    print("Example with existing filter_small_packets function:")
    print("=" * 60)
    
    # Create test packet stream
    test_stream = create_test_packet_stream()
    
    # Use the existing filter function
    from vidfile_iterator import filter_small_packets
    
    # Apply the filter while preserving consecutivity
    filtered_streams = filter_stream_preserve_consecutivity(test_stream, filter_small_packets)
    
    # Process each consecutive group
    for group_idx, consecutive_packets in enumerate(filtered_streams):
        print(f"\nConsecutive Group {group_idx + 1} (size < 1000):")
        print("-" * 40)
        
        packet_numbers = []
        for packet_no, packet in consecutive_packets:
            packet_numbers.append(packet_no)
            print(f"  Packet {packet_no}: size={packet.size}")
        
        print(f"  Total packets in group: {len(packet_numbers)}")
        print(f"  Packet numbers: {packet_numbers}")

if __name__ == "__main__":
    test_consecutive_filter()
    test_with_real_filter_example() 