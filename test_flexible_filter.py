#!/usr/bin/env python3
"""
Test script to verify filter_stream_preserve_consecutivity works with both
single iterators and iterator of iterators.
"""

from vidfile_iterator import filter_stream_preserve_consecutivity, packet_data_type
from typing import Iterator, List

def create_test_packet_stream() -> Iterator[packet_data_type]:
    """Create a test packet stream with some gaps."""
    packet_numbers = [0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 15, 16, 17, 18, 20]
    
    for packet_no in packet_numbers:
        mock_packet = type('MockPacket', (), {
            'size': packet_no * 100,
            'stream': type('MockStream', (), {'codec_context': None})()
        })()
        yield (packet_no, mock_packet)

def create_multiple_test_streams() -> List[Iterator[packet_data_type]]:
    """Create multiple test packet streams."""
    def stream1():
        packet_numbers = [0, 1, 2, 5, 6, 7]
        for packet_no in packet_numbers:
            mock_packet = type('MockPacket', (), {
                'size': packet_no * 100,
                'stream': type('MockStream', (), {'codec_context': None})()
            })()
            yield (packet_no, mock_packet)
    
    def stream2():
        packet_numbers = [10, 11, 12, 13, 15, 16, 17, 18, 20]
        for packet_no in packet_numbers:
            mock_packet = type('MockPacket', (), {
                'size': packet_no * 100,
                'stream': type('MockStream', (), {'codec_context': None})()
            })()
            yield (packet_no, mock_packet)
    
    return [stream1(), stream2()]

def test_single_iterator():
    """Test with a single packet iterator."""
    print("Testing with single iterator:")
    print("-" * 40)
    
    # Create single test stream
    test_stream = create_test_packet_stream()
    
    # Define filter function
    def size_filter(packet_data: packet_data_type) -> bool:
        packet_no, packet = packet_data
        return packet.size < 1000
    
    # Apply filter
    filtered_streams = filter_stream_preserve_consecutivity(test_stream, size_filter)
    
    # Process results
    for group_idx, consecutive_packets in enumerate(filtered_streams):
        print(f"Group {group_idx + 1}:")
        packet_numbers = []
        for packet_no, packet in consecutive_packets:
            packet_numbers.append(packet_no)
            print(f"  Packet {packet_no}: size={packet.size}")
        print(f"  Packet numbers: {packet_numbers}")
        print()

def test_iterator_of_iterators():
    """Test with an iterator of iterators."""
    print("Testing with iterator of iterators:")
    print("-" * 40)
    
    # Create multiple test streams
    test_streams = create_multiple_test_streams()
    
    # Define filter function
    def size_filter(packet_data: packet_data_type) -> bool:
        packet_no, packet = packet_data
        return packet.size < 1000
    
    # Apply filter
    filtered_streams = filter_stream_preserve_consecutivity(test_streams, size_filter)
    
    # Process results
    for group_idx, consecutive_packets in enumerate(filtered_streams):
        print(f"Group {group_idx + 1}:")
        packet_numbers = []
        for packet_no, packet in consecutive_packets:
            packet_numbers.append(packet_no)
            print(f"  Packet {packet_no}: size={packet.size}")
        print(f"  Packet numbers: {packet_numbers}")
        print()

def test_list_of_iterators():
    """Test with a list containing iterators."""
    print("Testing with list of iterators:")
    print("-" * 40)
    
    # Create multiple test streams as a list
    test_streams = create_multiple_test_streams()
    
    # Define filter function
    def size_filter(packet_data: packet_data_type) -> bool:
        packet_no, packet = packet_data
        return packet.size < 1000
    
    # Apply filter
    filtered_streams = filter_stream_preserve_consecutivity(test_streams, size_filter)
    
    # Process results
    for group_idx, consecutive_packets in enumerate(filtered_streams):
        print(f"Group {group_idx + 1}:")
        packet_numbers = []
        for packet_no, packet in consecutive_packets:
            packet_numbers.append(packet_no)
            print(f"  Packet {packet_no}: size={packet.size}")
        print(f"  Packet numbers: {packet_numbers}")
        print()

def test_empty_streams():
    """Test with empty streams."""
    print("Testing with empty streams:")
    print("-" * 40)
    
    # Empty single iterator
    empty_single = iter([])
    filtered_single = list(filter_stream_preserve_consecutivity(empty_single, lambda p: True))
    print(f"Empty single iterator: {len(filtered_single)} groups")
    
    # Empty iterator of iterators
    empty_multiple = iter([])
    filtered_multiple = list(filter_stream_preserve_consecutivity(empty_multiple, lambda p: True))
    print(f"Empty iterator of iterators: {len(filtered_multiple)} groups")
    
    # List with empty iterators
    empty_list = [iter([]), iter([])]
    filtered_list = list(filter_stream_preserve_consecutivity(empty_list, lambda p: True))
    print(f"List with empty iterators: {len(filtered_list)} groups")

if __name__ == "__main__":
    print("Testing filter_stream_preserve_consecutivity flexibility")
    print("=" * 60)
    
    test_single_iterator()
    test_iterator_of_iterators()
    test_list_of_iterators()
    test_empty_streams()
    
    print("All tests completed!") 