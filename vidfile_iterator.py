import av
import av.container, av.packet, av.stream
import numpy as np
from typing import Callable, List, Optional, Union, Iterator, Tuple
from typing_extensions import TypeAlias
from fractions import Fraction
from itertools import groupby


# Note: If you're seeking/jumping: Always flush after seeking to clear stale state

    
# Define types for later, packet_data and packet_data_iterator are used to store the packet data and the iterator respectively
# Define a new type called packet_data that is a tuple of an integer and an av.packet.Packet
packet_data_type: TypeAlias = Tuple[int, av.packet.Packet]
packet_data_iterator: TypeAlias = Iterator[packet_data_type]
packet_data_iterator_iterator: TypeAlias = Iterator[packet_data_iterator]
frame_list_type: TypeAlias = List[av.VideoFrame]
frame_data_type: TypeAlias = Tuple[int, int, av.VideoFrame]
frame_data_iterator: TypeAlias = Iterator[frame_data_type]

def _create_consecutive_iterator(start_packet_data: packet_data_type, stream_iterator: packet_data_iterator, filter_func: Callable[[packet_data_type], bool]) -> packet_data_iterator:
    """
    Create an iterator for consecutive packets starting from start_packet_data.
    
    Args:
        start_packet_data: The first packet in the consecutive sequence
        stream_iterator: Iterator of remaining packets to check
        filter_func: Function that takes packet_data and returns bool
    
    Yields:
        Consecutive packet_data tuples that match the filter condition
    """
    # Yield the first packet
    yield start_packet_data
    
    packet_no, packet = start_packet_data
    last_packet_no = packet_no
    
    # Continue yielding consecutive packets from the stream
    for next_packet_data in stream_iterator:
        next_packet_no, next_packet = next_packet_data
        
        # Check if this packet matches the filter condition
        if not filter_func(next_packet_data):
            # Packet doesn't match filter, stop this consecutive group
            break
        
        # Check if this packet is consecutive
        if next_packet_no != last_packet_no + 1:
            # Non-consecutive packet, stop this consecutive group
            break
        
        # Consecutive packet that matches filter, yield it
        yield next_packet_data
        last_packet_no = next_packet_no

def _process_single_stream(stream: packet_data_iterator, filter_func: Callable[[packet_data_type], bool]) -> packet_data_iterator_iterator:
    """
    Process a single packet stream while preserving consecutivity.
    
    Args:
        stream: Iterator of packet_data tuples
        filter_func: Function that takes packet_data and returns bool
    
    Yields:
        Iterator of consecutive packet_data tuples that match the filter condition
    """
    # Create an iterator from the stream for peeking
    stream_iterator = iter(stream)
    
    try:
        while True:
            # Get the next packet
            packet_data = next(stream_iterator)
            packet_no, packet = packet_data
            
            # Check if this packet matches the filter condition
            if filter_func(packet_data):
                # Create a streaming iterator for this consecutive group
                consecutive_iterator = _create_consecutive_iterator(packet_data, stream_iterator, filter_func)
                yield consecutive_iterator
                
                # The consecutive_iterator has consumed all consecutive packets
                # We need to continue with the next non-consecutive packet
                # This is handled by the _create_consecutive_iterator function
                # which stops when it encounters a non-consecutive or non-matching packet
            # If packet doesn't match filter, just continue to next packet
            
    except StopIteration:
        # Stream exhausted
        pass

def _normalize_input(stream) -> packet_data_iterator_iterator:
    """
    Convert single iterator to iterator of iterators if needed.
    
    Args:
        stream: Either a single packet iterator or iterator of packet iterators
    
    Returns:
        Iterator of packet iterators
    """
    try:
        # Try to get the first item
        first_item = next(iter(stream))
        
        # Check if first_item is a packet_data tuple (has packet_no as first element)
        if isinstance(first_item, tuple) and len(first_item) == 2 and isinstance(first_item[0], int):
            # It's a single packet stream, wrap it in a list
            return [stream]
        else:
            # It's already an iterator of iterators, reconstruct the stream
            def reconstruct_stream():
                yield first_item
                yield from stream
            return reconstruct_stream()
            
    except StopIteration:
        # Empty stream, return empty iterator
        return []
    except TypeError:
        # stream is not iterable, treat it as a single iterator
        return [stream]

def filter_stream_preserve_consecutivity(
    packet_stream: Union[packet_data_iterator, packet_data_iterator_iterator], 
    filter_func: Callable[[packet_data_type], bool]
) -> packet_data_iterator_iterator:
    """
    Filter a packet stream while preserving consecutivity.
    
    This function takes either a single stream of packets or an iterator of packet streams,
    along with a filter function, then returns an iterator of iterators where each inner 
    iterator contains consecutive packets that match the filter condition. When there's a 
    break in consecutivity (non-consecutive packet numbers), a new iterator is started.
    
    Args:
        packet_stream: Either:
            - Iterator of packet_data tuples (packet_no, packet), or
            - Iterator of iterators of packet_data tuples
        filter_func: Function that takes packet_data and returns bool
    
    Yields:
        Iterator of consecutive packet_data tuples that match the filter condition
        
    Example:
        # Single stream
        filtered_streams = filter_stream_preserve_consecutivity(
            packet_stream, 
            lambda p: p[1].size < 1000
        )
        
        # Iterator of iterators
        filtered_streams = filter_stream_preserve_consecutivity(
            [packet_stream1, packet_stream2], 
            lambda p: p[1].size < 1000
        )
        
        for consecutive_packets in filtered_streams:
            # Each consecutive_packets iterator contains packets with consecutive numbers
            for packet_no, packet in consecutive_packets:
                print(f"Processing consecutive packet {packet_no}")
    """
    # Normalize input: wrap single iterator in a list to make it an iterator of iterators
    normalized_streams = _normalize_input(packet_stream)
    
    # Process each stream in the normalized input
    for stream in normalized_streams:
        yield from _process_single_stream(stream, filter_func)

def group_packets_starting_with_keyframe(packet_stream, filter_func):
    """
    Groups filtered packets so that each group starts with a keyframe.
    Only yields groups that start with a keyframe and contain at least one packet.
    """
    group = []
    for packet_data in packet_stream:
        packet_no, packet = packet_data
        if filter_func(packet_data):
            if hasattr(packet, 'is_keyframe') and packet.is_keyframe:
                if group:
                    yield iter(group)
                group = [packet_data]
            else:
                if group:
                    group.append(packet_data)
                # else: skip packets before first keyframe
    if group:
        yield iter(group)

class FileFrameIterator:
    """
    Factory class to create frame iterators from a video file.
    """
    def __init__(self, filename: str):
        self.filename = filename
        self.container: av.container.InputContainer = av.open(filename, mode='r')  # type: ignore[assignment]
        self.container_stream = self.container.streams.video[0]
        self.time_base = self.container_stream.time_base if self.container_stream.time_base else 1.0 / 25.0
        self.container_stream.thread_type = "AUTO"
        self.container_stream.thread_count = 1  # Set to 1 to avoid threading issues with frame extraction
        self.iterator = self.frame_iterator()

    def frame_iterator(self) -> packet_data_iterator:
        for packet_no, packet in enumerate(self.container.demux(self.container_stream)):
            # print("Processing packet number:", packet_no)
            packet:packet_data = (packet_no, packet)
            yield packet

def chain_filters(packet_stream, *filter_functions):
    """
    Chain multiple filters together using iterators.
    A packet must pass ALL filters to be included.
    
    Args:
        packet_stream: Iterator of (packet_no, packet) tuples
        *filter_functions: Variable number of filter functions
    
    Yields:
        Packets that pass all filters
    """
    for packet_data in packet_stream:
        # Check if packet passes all filters
        if all(filter_func(packet_data) for filter_func in filter_functions):
            yield packet_data


def filter_small_packets(packet_data: packet_data_type, max_size=1000):
    """
    Filter function that returns True if packet size is smaller than max_size.
    
    Args:
        packet_data: Tuple of (packet_no, packet)
        max_size: Maximum packet size in bytes (default: 1000)
    
    Returns:
        bool: True if packet.size < max_size, False otherwise
    """
    packet_no, packet = packet_data
    return packet.size < max_size


def filter_large_packets(packet_data: packet_data_type, min_size=500):
    """
    Filter function that returns True if packet size is larger than min_size.
    
    Args:
        packet_data: Tuple of (packet_no, packet)
        min_size: Minimum packet size in bytes (default: 500)
    
    Returns:
        bool: True if packet.size > min_size, False otherwise
    """
    packet_no, packet = packet_data
    return packet.size > min_size


def filter_by_pts_range(packet_data: packet_data_type, min_pts=0, max_pts=None):
    """
    Filter function that returns True if packet PTS is within the specified range.
    
    Args:
        packet_data: Tuple of (packet_no, packet)
        min_pts: Minimum PTS value (default: 0)
        max_pts: Maximum PTS value (default: None for no upper limit)
    
    Returns:
        bool: True if packet.pts is within range, False otherwise
    """
    packet_no, packet = packet_data
    if packet.pts is None:
        return False
    if max_pts is None:
        return packet.pts >= min_pts
    return min_pts <= packet.pts <= max_pts

def stream_frames_from_packet(packet_data: packet_data_type):
    """
    Decode packets using PyAV version 14.4.0 compatible API.
    Uses codec_context.decode() method for older PyAV versions.
    
    Args:
        packet_data: Tuple of (packet_no, packet)
    """
    packet_no, packet = packet_data
    
    # Get the codec context from the stream
    codec_context = packet.stream.codec_context
    
    # Decode the packet using the older API
    try:
        frames = codec_context.decode(packet)
        frame_count = 0
        for frame in frames:
            print(f"      Packet number: {packet_no}, Frame number: {frame_count}, Frame PTS: {frame.pts}")
            frame_count += 1
        
        if frame_count == 0:
            print(f"      No frames decoded from this packet (iterator was empty from start)")
            
    except Exception as e:
        print(f"      Error decoding packet: {e}")

def decode_packet_to_frame_iterator(packet_data: packet_data_type) -> frame_data_iterator:
    """
    Decode a packet to a frame iterator using PyAV version 14.4.0 compatible API.
    """
    packet_no, packet = packet_data
    frames = []
    for frame_no, frame in enumerate(decode_packet_to_frames(packet_data)):
        yield (packet_no, frame_no, frame)

def decode_packet_stream_to_frame_stream(packet_data_iterator: packet_data_iterator) -> frame_data_iterator:
    """
    Decode a packet stream to a frame stream using PyAV version 14.4.0 compatible API.
    """
    for packet_data in packet_data_iterator:
        yield from decode_packet_to_frame_iterator(packet_data)

def decode_packet_to_frames(packet_data: packet_data_type) -> frame_list_type:
    """
    Decode a packet to a list of frames using PyAV version 14.4.0 compatible API.
    Note: This function doesn't handle decoder flushing - use decode_all_packets_with_flush instead.
    
    Args:
        packet_data: Tuple of (packet_no, packet)
    
    Returns:
        List of decoded video frames
    """
    packet_no, packet = packet_data
    frames = []
    
    # Get the codec context from the stream
    codec_context = packet.stream.codec_context
    
    # Decode the packet using the older API
    try:
        decoded_frames = codec_context.decode(packet)
        frames = list(decoded_frames)
    except Exception as e:
        # Don't print every error - only print occasionally to avoid spam
        if packet_no % 10 == 0:  # Print every 10th error
            print(f"Error decoding packet {packet_no}: {e}")
    
    return frames

def decode_packet_to_frames_with_state(packet_data: packet_data_type, codec_context: Optional[av.CodecContext] = None) -> Tuple[frame_list_type, av.CodecContext]:
    """
    Decode a packet to a list of frames while maintaining decoder state.
    This is better than decode_packet_to_frames for individual packet decoding.
    
    Args:
        packet_data: Tuple of (packet_no, packet)
        codec_context: Optional codec context to reuse (for state maintenance)
    
    Returns:
        Tuple of (frames_list, codec_context) for continued use
    """
    packet_no, packet = packet_data
    frames = []
    
    # Get or create the codec context
    if codec_context is None:
        codec_context = packet.stream.codec_context
    
    # Decode the packet using the older API
    try:
        decoded_frames = codec_context.decode(packet)
        frames = list(decoded_frames)
    except Exception as e:
        print(f"Error decoding packet {packet_no}: {e}")
    
    return frames, codec_context

def decode_all_packets_with_flush(packet_iterator, max_packets=None):
    """
    Decode all packets with proper flushing at the end.
    This is the recommended approach for getting all frames.
    
    Args:
        packet_iterator: Iterator of packet_data tuples
        max_packets: Maximum number of packets to process (None for all)
    
    Yields:
        Tuple of (packet_no, frames_list) for each packet
    """
    packet_count = 0
    codec_context = None
    
    try:
        for packet_data in packet_iterator:
            packet_no, packet = packet_data
            
            # Get codec context from first packet
            if codec_context is None:
                codec_context = packet.stream.codec_context
            
            # Decode this packet
            frames = decode_packet_to_frames(packet_data)
            yield (packet_no, frames)
            
            packet_count += 1
            if max_packets and packet_count >= max_packets:
                break
        
        # Flush the decoder to get remaining frames
        if codec_context:
            print(f"Flushing decoder after {packet_count} packets...")
            remaining_frames = flush_decoder(codec_context)
            if remaining_frames:
                print(f"Got {len(remaining_frames)} remaining frames from flush")
                yield (-1, remaining_frames)  # Use -1 to indicate flushed frames
                
    except Exception as e:
        print(f"Error in decode_all_packets_with_flush: {e}")
        # Don't re-raise - just stop processing this iterator

def flush_decoder(codec_context: av.CodecContext) -> frame_list_type:
    """
    Flush the decoder to get any remaining frames.
    For PyAV version 14.4.0, this sends None to the decoder.
    
    Args:
        codec_context: The codec context to flush
    
    Returns:
        List of remaining decoded frames
    """
    frames = []
    
    try:
        # Send None to flush the decoder
        decoded_frames = codec_context.decode(None)
        frames = list(decoded_frames)
    except Exception as e:
        print(f"Error during decoder flush: {e}")
    
    return frames

def decode_group_with_seek(filename: str, group_packets: List[packet_data_type]) -> frame_list_type:
    """
    Decode a group of packets by seeking to the nearest previous keyframe.
    This allows decoding packets that don't start with a keyframe.
    
    Args:
        filename: Path to the video file
        group_packets: List of (packet_no, packet) tuples to decode
    
    Returns:
        List of decoded frames corresponding to the packets in the group
    """
    if not group_packets:
        return []
    
    # Get the PTS of the first packet in the group
    first_packet_no, first_packet = group_packets[0]
    if first_packet.pts is None:
        print(f"Warning: First packet {first_packet_no} has no PTS, skipping group")
        return []
    
    # Create a set of packet numbers we want to keep
    target_packet_numbers = {packet_no for packet_no, _ in group_packets}
    
    # Open the file and seek to the nearest previous keyframe
    container = av.open(filename, mode='r')
    video_stream = container.streams.video[0]
    
    try:
        # Seek to the nearest previous keyframe before the first packet
        container.seek(first_packet.pts, any_frame=False, backward=True, stream=video_stream)
        
        frames = []
        codec_context = video_stream.codec_context
        
        # Demux and decode from the seek position
        for packet in container.demux(video_stream):
            if packet.pts is None:
                continue
            
            # Check if this packet corresponds to one in our target group
            # We need to map the demuxed packet back to our packet numbering
            # For now, we'll use a simple approach: check if PTS matches
            packet_matches = False
            for target_packet_no, target_packet in group_packets:
                if target_packet.pts == packet.pts:
                    packet_matches = True
                    break
            
            if packet_matches:
                # Decode this packet
                try:
                    decoded_frames = codec_context.decode(packet)
                    frames.extend(list(decoded_frames))
                    print(f"Decoded packet with PTS {packet.pts}: {len(list(decoded_frames))} frames")
                except Exception as e:
                    print(f"Error decoding packet with PTS {packet.pts}: {e}")
            
            # Stop if we've passed the last packet in our group
            last_packet_no, last_packet = group_packets[-1]
            if last_packet.pts is not None and packet.pts is not None and packet.pts > last_packet.pts:
                break
        
        # Flush the decoder to get any remaining frames
        try:
            remaining_frames = codec_context.decode(None)
            frames.extend(list(remaining_frames))
            if remaining_frames:
                print(f"Got {len(list(remaining_frames))} remaining frames from flush")
        except Exception as e:
            print(f"Error during decoder flush: {e}")
        
        return frames
        
    finally:
        container.close()

def group_packets_by_pts_and_decode(filename: str, packet_stream, filter_func) -> Iterator[frame_list_type]:
    """
    Group packets by filter criteria and decode each group using seek-and-decode.
    This allows filtering out keyframes while still being able to decode the remaining packets.
    Note: This uses a single-pass approach that loads packets into memory for simplicity and performance.
    
    Args:
        filename: Path to the video file
        packet_stream: Iterator of (packet_no, packet) tuples
        filter_func: Function that takes (packet_no, packet) and returns bool
    
    Yields:
        List of decoded frames for each group
    """
    group = []
    
    for packet_data in packet_stream:
        packet_no, packet = packet_data
        
        if filter_func(packet_data):
            group.append(packet_data)
        else:
            # Packet doesn't match filter, process current group if it exists
            if group:
                print(f"Processing group with {len(group)} packets")
                frames = decode_group_with_seek(filename, group)
                if frames:
                    yield frames
                group = []
    
    # Process the last group if it exists
    if group:
        print(f"Processing final group with {len(group)} packets")
        frames = decode_group_with_seek(filename, group)
        if frames:
            yield frames

def group_packets_by_pts_and_decode_streaming(filename: str, packet_stream, filter_func) -> Iterator[frame_list_type]:
    """
    Group packets by filter criteria and decode each group using seek-and-decode.
    This is a true streaming approach that doesn't store packets in memory.
    
    Args:
        filename: Path to the video file
        packet_stream: Iterator of (packet_no, packet) tuples
        filter_func: Function that takes (packet_no, packet) and returns bool
    
    Yields:
        List of decoded frames for each group
    """
    # First pass: identify group boundaries and store minimal info needed for seeking
    group_boundaries = []
    current_group_start = None
    current_group_end = None
    
    for packet_data in packet_stream:
        packet_no, packet = packet_data
        
        if filter_func(packet_data):
            # Packet matches filter
            if current_group_start is None:
                # Start of a new group
                current_group_start = (packet_no, packet.pts)
            current_group_end = (packet_no, packet.pts)
        else:
            # Packet doesn't match filter, end current group if it exists
            if current_group_start is not None:
                group_boundaries.append((current_group_start, current_group_end))
                current_group_start = None
                current_group_end = None
    
    # Handle the last group if it exists
    if current_group_start is not None:
        group_boundaries.append((current_group_start, current_group_end))
    
    # Second pass: process each group individually using seek-and-decode
    for i, (start_info, end_info) in enumerate(group_boundaries):
        start_packet_no, start_pts = start_info
        end_packet_no, end_pts = end_info
        
        print(f"Processing group {i+1}/{len(group_boundaries)}: packets {start_packet_no}-{end_packet_no}, PTS {start_pts}-{end_pts}")
        
        # Decode this group using seek-and-decode
        frames = decode_group_by_pts_range(filename, start_pts, end_pts)
        if frames:
            yield frames

def decode_group_by_pts_range(filename: str, start_pts: int, end_pts: int) -> frame_list_type:
    """
    Decode frames for packets within a PTS range by seeking to the nearest previous keyframe.
    
    Args:
        filename: Path to the video file
        start_pts: Start PTS (inclusive)
        end_pts: End PTS (inclusive)
    
    Returns:
        List of decoded frames in the PTS range
    """
    # Open the file and seek to the nearest previous keyframe
    container = av.open(filename, mode='r')
    video_stream = container.streams.video[0]
    
    try:
        # Seek to the nearest previous keyframe before the start PTS
        container.seek(start_pts, any_frame=False, backward=True, stream=video_stream)
        
        frames = []
        codec_context = video_stream.codec_context
        
        # Demux and decode from the seek position
        for packet in container.demux(video_stream):
            if packet.pts is None:
                continue
            
            # Check if this packet is within our target PTS range
            if start_pts <= packet.pts <= end_pts:
                # Decode this packet
                try:
                    decoded_frames = codec_context.decode(packet)
                    frames.extend(list(decoded_frames))
                    print(f"Decoded packet with PTS {packet.pts}: {len(list(decoded_frames))} frames")
                except Exception as e:
                    print(f"Error decoding packet with PTS {packet.pts}: {e}")
            
            # Stop if we've passed the end PTS
            if packet.pts > end_pts:
                break
        
        # Flush the decoder to get any remaining frames
        try:
            remaining_frames = codec_context.decode(None)
            frames.extend(list(remaining_frames))
            if remaining_frames:
                print(f"Got {len(list(remaining_frames))} remaining frames from flush")
        except Exception as e:
            print(f"Error during decoder flush: {e}")
        
        return frames
        
    finally:
        container.close()
