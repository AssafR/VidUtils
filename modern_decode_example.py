import sys
from vidfile_iterator import FileFrameIterator
import av
import av.container, av.packet, av.stream

def decode_packet_modern(packet_data):
    """
    Modern approach to decode packets using PyAV's send/receive API.
    This replaces the deprecated packet.decode() method.
    """
    packet_no, packet = packet_data
    
    # Get the codec context from the stream
    codec_context = packet.stream.codec_context
    
    print(f"  --- Packet number: {packet_no}, Packet size: {packet.size}")
    
    # Send the packet to the decoder
    try:
        codec_context.send(packet)
    except Exception as e:
        print(f"      Error sending packet to decoder: {e}")
        return
    
    # Receive all frames from the decoder
    frame_count = 0
    while True:
        try:
            frame = codec_context.receive()
            print(f"      Frame {frame_count}: PTS={frame.pts}, time_base={frame.time_base}")
            frame_count += 1
        except av.EOFError:
            # No more frames available from this packet
            break
        except Exception as e:
            print(f"      Error receiving frame: {e}")
            break
    
    if frame_count == 0:
        print(f"      No frames decoded from this packet")

def decode_packet_modern_with_flush(packet_data, codec_context):
    """
    Alternative approach that handles the codec context separately.
    This is useful when you want to flush the decoder at the end.
    """
    packet_no, packet = packet_data
    
    print(f"  --- Packet number: {packet_no}, Packet size: {packet.size}")
    
    # Send the packet to the decoder
    try:
        codec_context.send(packet)
    except Exception as e:
        print(f"      Error sending packet to decoder: {e}")
        return
    
    # Receive all frames from the decoder
    frame_count = 0
    while True:
        try:
            frame = codec_context.receive()
            print(f"      Frame {frame_count}: PTS={frame.pts}, time_base={frame.time_base}")
            frame_count += 1
        except av.EOFError:
            # No more frames available from this packet
            break
        except Exception as e:
            print(f"      Error receiving frame: {e}")
            break
    
    if frame_count == 0:
        print(f"      No frames decoded from this packet")
    
    return frame_count

def main():
    if len(sys.argv) < 2:
        print("Usage: python modern_decode_example.py <video_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    print(f"Analyzing: {filename}")
    
    # Open the container and get the video stream
    container = av.open(filename)
    video_stream = container.streams.video[0]
    codec_context = video_stream.codec_context
    
    print(f"Video codec: {codec_context.name}")
    print(f"Video FPS: {video_stream.average_rate}")
    
    # Create the frame iterator
    frame_iterator = FileFrameIterator(filename)
    
    # Process first few packets with modern decoding
    packet_count = 0
    max_packets_to_analyze = 5
    
    print(f"\n=== Modern Decoding Approach ===")
    for packet_data in frame_iterator.iterator:
        decode_packet_modern(packet_data)
        packet_count += 1
        if packet_count >= max_packets_to_analyze:
            break
    
    # Flush the decoder to get any remaining frames
    print(f"\n=== Flushing Decoder ===")
    try:
        codec_context.send(None)  # Send None to flush
        frame_count = 0
        while True:
            try:
                frame = codec_context.receive()
                print(f"Flushed frame {frame_count}: PTS={frame.pts}")
                frame_count += 1
            except av.EOFError:
                break
        if frame_count > 0:
            print(f"Flushed {frame_count} remaining frames")
    except Exception as e:
        print(f"Error during flush: {e}")
    
    print(f"\nAnalyzed {packet_count} packets using modern decoding API.")

if __name__ == "__main__":
    main() 