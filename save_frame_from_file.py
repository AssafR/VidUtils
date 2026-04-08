import sys
import cv2
import numpy as np
from utils import WindowClosed, WindowManager
from vidfile_iterator import FileFrameIterator, stream_frames_from_packet, decode_packet_to_frames


def read_save_frame_from_file(frame_no=2):

    """Test the modern decoding approach in vidfile_iterator.py"""
    if len(sys.argv) < 2:
        print("Usage: python test_vidfile_iterator.py <video_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    print(f"Testing modern decoding in vidfile_iterator.py with: {filename}")
    max_packets_to_test = 300


    # Create the frame iterator
    # Test the decode_packet_to_frames function

    print("\n=== Testing decode_packet_to_frames ===")
    frame_iterator = FileFrameIterator(filename)  # Reset iterator
    with WindowManager('Frame Window'):
        packet_count = 0
        frame_count = 0

        for packet_data in frame_iterator.packet_iterator:
            packet_no, packet = packet_data
            print(f"  Decoding packet {packet_no}...")
            frames = decode_packet_to_frames(packet_data)
            print(f"  Got {len(frames)} frames from packet {packet_no}")
            
            for i, frame in enumerate(frames):
                print(f"    Frame {i}_{frame_count}: PTS={frame.pts}, time_base={frame.time_base}")

                frame_img = frame.to_image()
                frame_np = np.array(frame_img)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

                # if frame_count >= 6:
                #     # Save the current frame to disk
                #     output_filename = f"saved_frame_{frame_count}.jpg"
                #     cv2.imwrite(output_filename, frame_bgr)
                #     print(f"Saved frame {frame_count} to {output_filename}")
                #     return  # Exit after saving the frame


                cv2.putText(frame_bgr, f"Frame {frame_count}:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Frame Window', frame_bgr)
                cv2.waitKey(1)
                if cv2.getWindowProperty('Frame Window', cv2.WND_PROP_VISIBLE) <= 0:
                    raise WindowClosed("Window was closed by user")
                frame_count += 1


            packet_count += 1
            if packet_count >= max_packets_to_test:
                break





if __name__ == "__main__":
    read_save_frame_from_file() 