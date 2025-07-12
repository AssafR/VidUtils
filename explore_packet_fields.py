import sys
from vidfile_iterator import FileFrameIterator
import av
import av.container, av.packet, av.stream

def explore_packet_fields(packet_data, actual_fps=None):
    """
    Explore and display all available fields in an av.packet.Packet object.
    """
    packet_no, packet = packet_data
    
    print(f"\n=== Packet {packet_no} Analysis ===")
    print(f"Packet type: {type(packet)}")
    print(f"Packet size: {packet.size}")
    
    # Get all attributes of the packet
    print("\n--- All Packet Attributes ---")
    for attr in dir(packet):
        if not attr.startswith('_'):  # Skip private attributes
            try:
                value = getattr(packet, attr)
                if not callable(value):  # Skip methods, only show properties
                    print(f"  {attr}: {value}")
            except Exception as e:
                print(f"  {attr}: Error accessing - {e}")
    
    # Check for specific frame-related attributes
    print("\n--- Frame-Related Attributes ---")
    frame_attrs = ['pts', 'dts', 'duration', 'time_base', 'stream_index', 'flags', 'side_data']
    for attr in frame_attrs:
        try:
            value = getattr(packet, attr, None)
            if value is not None:
                print(f"  {attr}: {value}")
        except Exception as e:
            print(f"  {attr}: Error - {e}")
    
    # Test frame estimation with actual FPS
    print("\n--- Frame Estimation Test ---")
    if actual_fps:
        estimated_frames = estimate_frames_in_packet(packet, actual_fps)
        if estimated_frames>1:
            print(" ***** More than 1 frame in packet *****")
        print(f"  Estimated frames in packet (using actual FPS {actual_fps}): {estimated_frames}")
    else:
        estimated_frames = estimate_frames_in_packet(packet)
        print(f"  Estimated frames in packet (using default 30 FPS): {estimated_frames}")
    
    # Check if packet has any frame count information
    print("\n--- Frame Count Investigation ---")
    try:
        # Some packets might have frame count in side_data
        if hasattr(packet, 'side_data') and packet.side_data:
            print(f"  Side data available: {len(packet.side_data)} items")
            for i, data in enumerate(packet.side_data):
                print(f"    Side data {i}: {data}")
        
        # Check if there are any methods that might give frame info
        frame_methods = [attr for attr in dir(packet) if 'frame' in attr.lower() and callable(getattr(packet, attr))]
        if frame_methods:
            print(f"  Frame-related methods: {frame_methods}")
            
    except Exception as e:
        print(f"  Error checking frame info: {e}")

def estimate_frames_in_packet(packet, fps=30.0):
    """Estimate frame count using packet duration and time_base"""
    if packet.duration and packet.time_base:
        # Convert time_base to float for calculation
        time_base_float = float(packet.time_base)
        duration_seconds = packet.duration * time_base_float
        estimated_frames = duration_seconds * fps
        
        # Debug output
        print(f"  DEBUG: packet.duration = {packet.duration}")
        print(f"  DEBUG: packet.time_base = {packet.time_base} (type: {type(packet.time_base)})")
        print(f"  DEBUG: time_base_float = {time_base_float}")
        print(f"  DEBUG: duration_seconds = {duration_seconds}")
        print(f"  DEBUG: fps = {fps}")
        print(f"  DEBUG: estimated_frames = {estimated_frames}")
        print(f"  DEBUG: rounded_frames = {round(estimated_frames)}")
        
        return round(estimated_frames)
    else:
        print(f"  DEBUG: Missing duration or time_base")
        print(f"  DEBUG: packet.duration = {packet.duration}")
        print(f"  DEBUG: packet.time_base = {packet.time_base}")
    return None

def analyze_packet_flags(packet):
    """Analyze packet flags for frame-related hints"""
    flags = []
    if hasattr(packet, 'flags'):
        if packet.flags & 0x0001:  # AV_PKT_FLAG_KEY
            flags.append("Keyframe")
        if packet.flags & 0x0002:  # AV_PKT_FLAG_CORRUPT
            flags.append("Corrupt")
        if packet.flags & 0x0004:  # AV_PKT_FLAG_DISCARD
            flags.append("Discard")
    return flags

def main():
    if len(sys.argv) < 2:
        print("Usage: python explore_packet_fields.py <video_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    print(f"Exploring packet fields in: {filename}")
    
    # Create the frame iterator
    frame_iterator = FileFrameIterator(filename)
    
    # Get actual FPS from the stream
    container = av.open(filename)
    video_stream = container.streams.video[0]
    actual_fps = float(video_stream.average_rate) if video_stream.average_rate else None
    print(f"Actual video FPS: {actual_fps}")
    print(f"Video stream info: {video_stream}")
    
    # Analyze first few packets
    packet_count = 0
    max_packets_to_analyze = 15
    
    for packet_data in frame_iterator.iterator:
        explore_packet_fields(packet_data, actual_fps)
        packet_count += 1
        if packet_count >= max_packets_to_analyze:
            break
    
    print(f"\nAnalyzed {packet_count} packets.")

if __name__ == "__main__":
    main() 