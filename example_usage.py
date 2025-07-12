#!/usr/bin/env python3
"""
Example usage of filter_stream_preserve_consecutivity with video processing.
"""

from vidfile_iterator import (
    FileFrameIterator, 
    filter_stream_preserve_consecutivity, 
    filter_small_packets,
    decode_packet_to_frames,
    decode_all_packets_with_flush
)
from typing import Iterator

def process_consecutive_packets_efficiently(filename: str, max_packet_size: int = 1000):
    """
    Example: Process video packets efficiently by filtering small packets
    while preserving consecutivity for optimal processing.
    """
    print(f"Processing video file: {filename}")
    print("=" * 50)
    
    # Create file iterator
    file_iterator = FileFrameIterator(filename)
    
    # Filter packets while preserving consecutivity
    filtered_streams = filter_stream_preserve_consecutivity(
        file_iterator.iterator, 
        lambda p: filter_small_packets(p, max_packet_size)
    )
    
    total_consecutive_groups = 0
    total_packets_processed = 0
    
    # Process each consecutive group
    for group_idx, consecutive_packets in enumerate(filtered_streams):
        print(f"\nProcessing Consecutive Group {group_idx + 1}:")
        print("-" * 40)
        
        group_packets = list(consecutive_packets)  # Convert to list for processing
        packet_numbers = [p[0] for p in group_packets]
        
        print(f"  Packet numbers: {packet_numbers}")
        print(f"  Group size: {len(group_packets)} packets")
        
        # Process this consecutive group efficiently
        # You can use decode_all_packets_with_flush for optimal processing
        for packet_no, packet in group_packets:
            frames = decode_packet_to_frames((packet_no, packet))
            print(f"    Packet {packet_no}: decoded {len(frames)} frames")
            total_packets_processed += 1
        
        total_consecutive_groups += 1
    
    print(f"\nSummary:")
    print(f"  Total consecutive groups: {total_consecutive_groups}")
    print(f"  Total packets processed: {total_packets_processed}")

def create_custom_filter_example():
    """
    Example: Create custom filters for different use cases.
    """
    print("\n" + "=" * 60)
    print("Custom Filter Examples:")
    print("=" * 60)
    
    # Example 1: Filter packets by even packet numbers
    def even_packet_filter(packet_data):
        packet_no, packet = packet_data
        return packet_no % 2 == 0
    
    # Example 2: Filter packets by size range
    def size_range_filter(packet_data, min_size=500, max_size=2000):
        packet_no, packet = packet_data
        return min_size <= packet.size <= max_size
    
    # Example 3: Filter packets by position in stream (first 100 packets)
    def first_100_filter(packet_data):
        packet_no, packet = packet_data
        return packet_no < 100
    
    print("Available custom filters:")
    print("1. even_packet_filter: Keep only even-numbered packets")
    print("2. size_range_filter: Keep packets within size range")
    print("3. first_100_filter: Keep only first 100 packets")
    
    return {
        'even_packets': even_packet_filter,
        'size_range': size_range_filter,
        'first_100': first_100_filter
    }

def advanced_processing_example():
    """
    Example: Advanced processing with multiple filter stages.
    """
    print("\n" + "=" * 60)
    print("Advanced Processing Example:")
    print("=" * 60)
    
    # This shows how you might chain multiple processing steps
    # while maintaining consecutivity at each stage
    
    def create_processing_pipeline():
        """
        Create a processing pipeline that maintains consecutivity.
        """
        def stage1_filter(packet_data):
            # First stage: filter by size
            return filter_small_packets(packet_data, 1500)
        
        def stage2_filter(packet_data):
            # Second stage: additional filtering (e.g., by packet number)
            packet_no, packet = packet_data
            return packet_no % 3 == 0  # Every third packet
        
        return stage1_filter, stage2_filter
    
    stage1_filter, stage2_filter = create_processing_pipeline()
    
    print("Pipeline stages:")
    print("1. Stage 1: Filter packets by size < 1500")
    print("2. Stage 2: Keep every 3rd packet from stage 1")
    print("\nNote: Each stage preserves consecutivity independently")

if __name__ == "__main__":
    # Show examples (without actually processing a video file)
    print("filter_stream_preserve_consecutivity Usage Examples")
    print("=" * 60)
    
    # Show custom filter examples
    custom_filters = create_custom_filter_example()
    
    # Show advanced processing example
    advanced_processing_example()
    
    print("\n" + "=" * 60)
    print("To use with actual video file:")
    print("=" * 60)
    print("process_consecutive_packets_efficiently('your_video.mp4', max_packet_size=1000)")
    
    print("\nKey benefits:")
    print("1. Efficient processing of consecutive packets")
    print("2. Maintains packet order within each group")
    print("3. Allows for optimized batch processing")
    print("4. Preserves temporal relationships in video data") 