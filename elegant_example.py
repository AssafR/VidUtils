#!/usr/bin/env python3
"""
Example demonstrating the elegant approach to handling both single iterators
and iterator of iterators in filter_stream_preserve_consecutivity.
"""

from vidfile_iterator import filter_stream_preserve_consecutivity, packet_data_type
from typing import Iterator

def create_simple_stream() -> Iterator[packet_data_type]:
    """Create a simple packet stream."""
    for i in range(5):
        mock_packet = type('MockPacket', (), {
            'size': i * 200,
            'stream': type('MockStream', (), {'codec_context': None})()
        })()
        yield (i, mock_packet)

def demonstrate_elegance():
    """
    Demonstrate how the elegant approach works with both input types.
    """
    print("Elegant Approach Demonstration")
    print("=" * 50)
    
    # Create a simple stream
    single_stream = create_simple_stream()
    
    # Define a simple filter
    def size_filter(packet_data: packet_data_type) -> bool:
        packet_no, packet = packet_data
        return packet.size < 600  # Keep packets with size < 600
    
    print("1. Single Iterator Input:")
    print("-" * 30)
    print("filter_stream_preserve_consecutivity(single_stream, size_filter)")
    
    # This works because the function normalizes the input internally
    filtered1 = filter_stream_preserve_consecutivity(single_stream, size_filter)
    
    for group_idx, consecutive_packets in enumerate(filtered1):
        packet_numbers = [p[0] for p in consecutive_packets]
        print(f"   Group {group_idx + 1}: {packet_numbers}")
    
    print("\n2. Iterator of Iterators Input:")
    print("-" * 30)
    print("filter_stream_preserve_consecutivity([stream1, stream2], size_filter)")
    
    # Create multiple streams
    stream1 = create_simple_stream()
    stream2 = create_simple_stream()
    
    # This also works with the same function
    filtered2 = filter_stream_preserve_consecutivity([stream1, stream2], size_filter)
    
    for group_idx, consecutive_packets in enumerate(filtered2):
        packet_numbers = [p[0] for p in consecutive_packets]
        print(f"   Group {group_idx + 1}: {packet_numbers}")
    
    print("\n3. List of Iterators Input:")
    print("-" * 30)
    print("filter_stream_preserve_consecutivity(list_of_streams, size_filter)")
    
    # Create a list of streams
    streams_list = [create_simple_stream(), create_simple_stream()]
    
    # This works too
    filtered3 = filter_stream_preserve_consecutivity(streams_list, size_filter)
    
    for group_idx, consecutive_packets in enumerate(filtered3):
        packet_numbers = [p[0] for p in consecutive_packets]
        print(f"   Group {group_idx + 1}: {packet_numbers}")

def show_normalization_logic():
    """
    Show the simple normalization logic that makes this elegant.
    """
    print("\n" + "=" * 50)
    print("Normalization Logic (simplified):")
    print("=" * 50)
    
    print("""
def normalize_input(stream):
    try:
        first_item = next(iter(stream))
        
        # If first_item looks like packet_data (tuple with int first element)
        if isinstance(first_item, tuple) and len(first_item) == 2 and isinstance(first_item[0], int):
            return [stream]  # Wrap single iterator in list
        else:
            # Already an iterator of iterators, reconstruct
            def reconstruct():
                yield first_item
                yield from stream
            return reconstruct()
            
    except StopIteration:
        return []  # Empty stream
    except TypeError:
        return [stream]  # Not iterable, treat as single iterator

# Then simply process each stream:
normalized_streams = normalize_input(packet_stream)
for stream in normalized_streams:
    yield from process_single_stream(stream)
    """)

if __name__ == "__main__":
    demonstrate_elegance()
    show_normalization_logic()
    
    print("\n" + "=" * 50)
    print("Benefits of this approach:")
    print("=" * 50)
    print("1. Single function handles both input types")
    print("2. No complex type detection or multiple code paths")
    print("3. Easy to understand and maintain")
    print("4. Consistent behavior regardless of input type")
    print("5. Minimal overhead for normalization") 