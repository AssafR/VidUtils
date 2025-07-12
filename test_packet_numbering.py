#!/usr/bin/env python3
"""
Test script to verify that packet numbers are preserved correctly
through the filter_stream_preserve_consecutivity function.
"""

from vidfile_iterator import FileFrameIterator, filter_stream_preserve_consecutivity, filter_small_packets
import sys

def test_packet_numbering_preservation(filename: str):
    """
    Test that packet numbers are preserved through filtering.
    """
    print("Testing Packet Numbering Preservation:")
    print("=" * 50)
    
    # Create the frame iterator
    frame_iterator = FileFrameIterator(filename)
    
    # Apply filter to only get packets smaller than 2000 bytes
    max_packet_size = 2000
    filtered_iterator = filter_stream_preserve_consecutivity(
        frame_iterator.iterator, 
        lambda x: filter_small_packets(x, max_packet_size)
    )
    
    print(f"Original packet numbers should be preserved:")
    print("-" * 40)
    
    for iterator_idx, pd_it in enumerate(filtered_iterator):
        print(f"\nConsecutive Group {iterator_idx}:")
        packet_numbers = []
        
        for packet_data in pd_it:  # No enumerate() - preserve original numbers
            packet_no, packet = packet_data
            packet_numbers.append(packet_no)
            print(f"  Packet {packet_no} (size: {packet.size})")
        
        print(f"  Group packet numbers: {packet_numbers}")
        
        # Verify consecutivity
        if len(packet_numbers) > 1:
            is_consecutive = all(
                packet_numbers[i] + 1 == packet_numbers[i + 1] 
                for i in range(len(packet_numbers) - 1)
            )
            print(f"  Consecutive: {is_consecutive}")
        
        # Only process first few groups for demonstration
        if iterator_idx >= 3:
            break

def demonstrate_the_problem():
    """
    Demonstrate what happens when you use enumerate() incorrectly.
    """
    print("\n" + "=" * 50)
    print("Demonstrating the Problem with enumerate():")
    print("=" * 50)
    
    print("""
INCORRECT (what you had):
    for packet_data in enumerate(pd_it):
        packet_no, packet = packet_data  # packet_no is 0, 1, 2, 3...
        
CORRECT (what you should have):
    for packet_data in pd_it:
        packet_no, packet = packet_data  # packet_no is original number
    """)
    
    print("The enumerate() function creates new indices starting from 0,")
    print("which overwrites the original packet numbers from the video file.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_packet_numbering.py <video_file>")
        print("This will show that packet numbers are preserved correctly.")
        demonstrate_the_problem()
        sys.exit(1)
    
    filename = sys.argv[1]
    test_packet_numbering_preservation(filename) 