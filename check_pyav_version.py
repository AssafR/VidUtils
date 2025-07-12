import sys
import av
from vidfile_iterator import FileFrameIterator

def check_codec_context_methods():
    """Check what methods are available on the codec context"""
    if len(sys.argv) < 2:
        print("Usage: python check_pyav_version.py <video_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    print(f"PyAV version: {av.__version__}")
    
    # Open container and get codec context
    container = av.open(filename)
    video_stream = container.streams.video[0]
    codec_context = video_stream.codec_context
    
    print(f"Codec context type: {type(codec_context)}")
    print(f"Codec name: {codec_context.name}")
    
    # Check for send/receive methods
    print(f"\nHas 'send' method: {hasattr(codec_context, 'send')}")
    print(f"Has 'receive' method: {hasattr(codec_context, 'receive')}")
    
    # Check for decode method
    print(f"Has 'decode' method: {hasattr(codec_context, 'decode')}")
    
    # List all methods that contain 'send', 'receive', or 'decode'
    methods = [attr for attr in dir(codec_context) if any(x in attr.lower() for x in ['send', 'receive', 'decode']) and callable(getattr(codec_context, attr))]
    print(f"\nRelevant methods: {methods}")
    
    # Check packet methods
    frame_iterator = FileFrameIterator(filename)
    packet_data = next(frame_iterator.iterator)
    packet_no, packet = packet_data
    
    print(f"\nPacket type: {type(packet)}")
    print(f"Has 'decode' method on packet: {hasattr(packet, 'decode')}")
    
    # Try the old packet.decode() method
    print(f"\nTesting packet.decode() method...")
    try:
        frames = list(packet.decode())
        print(f"Success! Got {len(frames)} frames from packet.decode()")
        for i, frame in enumerate(frames):
            print(f"  Frame {i}: PTS={frame.pts}")
    except Exception as e:
        print(f"Error with packet.decode(): {e}")

if __name__ == "__main__":
    check_codec_context_methods() 