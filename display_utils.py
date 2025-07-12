import av
import matplotlib.pyplot as plt
from typing import List
from vidfile_iterator import decode_all_packets_with_flush, packet_data_iterator_iterator,decode_packet_to_frames_with_state, frame_list_type

# def display_thumbnails_regions(title: str, filename: str, regions: List[Region], thumbs_per_row: int = 10):
#     container = av.open(filename, mode='r')
#     stream = container.streams.video[0]
#     time_base = stream.time_base if stream.time_base else 1.0 / 25.0
#     stream.thread_type = "AUTO"

#     all_thumbs = []
#     labels = []

#     for region in regions:
#         start, end = region.start, region.end
#         score_text = f" ({region.score:.3f})" if region.score is not None else ""
#         container.seek(int(start / time_base), stream=stream)
#         thumbnails = []
#         for frame in container.decode(stream):
#             pts_sec = float(frame.pts * time_base)
#             if pts_sec > end:
#                 break
#             if pts_sec < start:
#                 continue
#             thumbnails.append(frame.to_image())
#             if len(thumbnails) >= thumbs_per_row:
#                 break
#         all_thumbs.append(thumbnails)
#         labels.append(f"{start:.2f}s – {end:.2f}s{score_text}")

#     rows = len(all_thumbs)
#     fig, axes = plt.subplots(rows, thumbs_per_row, figsize=(thumbs_per_row * 1.5, rows * 1.5))
#     fig.suptitle(title, fontsize=16)

#     for i, row_thumbs in enumerate(all_thumbs):
#         for j in range(thumbs_per_row):
#             ax = axes[i, j] if rows > 1 else axes[j]
#             ax.axis('off')
#             if j == 0:
#                 ax.set_title(labels[i], fontsize=10, loc='left')
#             if j < len(row_thumbs):
#                 ax.imshow(row_thumbs[j])

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)
#     plt.show()



def display_thumbnails_stream(pd_it_it: packet_data_iterator_iterator, thumbs_per_row: int = 10):
    all_thumbs = []
    labels = []

    for iterator_index, pd_it in enumerate(pd_it_it):
        if len(all_thumbs) > 10: # Stop after 10 rows
            break

        row_complete = False # Every iterator pd_it starts a new row
        thumbnails_row = []
        codec_context = None  # Reset codec context for each iterator group

        # Use the flush-based approach for better frame decoding within each group
        for packet_no, frames in decode_all_packets_with_flush(pd_it):
            print(f'\n-----\n----Starting with iterator {iterator_index}, packet {packet_no}')
            if len(frames) == 0:
                print(f"   No frames decoded from packet {packet_no}")
                continue
            print(f"   Decoded {len(frames)} frames from packet {packet_no}")

            thumbnails_row.extend(frame.to_image() for frame in frames)
            print(f"Adding {len(frames)} frames of packet {packet_no} to row {len(all_thumbs)}, current row length is {len(thumbnails_row)}")

            # Finished all frames in the packet
            if len(thumbnails_row) > thumbs_per_row:
                print(f"Cropping row {len(all_thumbs)} from {len(thumbnails_row)} frames to {thumbs_per_row} frames")
                thumbnails_row = thumbnails_row[:thumbs_per_row]
                row_complete = True
                break # Stop reading packets from this iterator
        
        # Exhausted all packets in the iterator pd_it , or finished a row
        if len(thumbnails_row) > 0 or row_complete:
            labels.append(f"iterator {iterator_index} , last packet {packet_no}")
            all_thumbs.append(thumbnails_row)
            thumbnails_row = []
            row_complete = False
            print(f"Early stop: Finished iterator {iterator_index} and row {len(all_thumbs)} with {len(thumbnails_row)} frames")
            continue # Continue with the next iterator
        if len(all_thumbs) > 4:
            break
        print(f"Finished iterator {iterator_index} and row {len(all_thumbs)} with {len(thumbnails_row)} frames")


    rows = len(all_thumbs)
    if rows>0:
        fig, axes = plt.subplots(rows, thumbs_per_row, figsize=(thumbs_per_row * 1.5, rows * 1.5))
        title = "Thumbnails"
        fig.suptitle(title, fontsize=16)

        for i, row_thumbs in enumerate(all_thumbs):
            for j in range(thumbs_per_row):
                ax = axes[i, j] if rows > 1 else axes[j]
                ax.axis('off')
                if j == 0:
                    ax.set_title(labels[i], fontsize=10, loc='left')
                if j < len(row_thumbs):
                    ax.imshow(row_thumbs[j])

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    else:
        print("No thumbnails to display")


def display_thumbnails_stream_with_global_state(pd_it_it: packet_data_iterator_iterator, thumbs_per_row: int = 10):
    """
    Alternative version that maintains decoder state across all iterators.
    This might work better for some video files.
    """
    all_thumbs = []
    labels = []
    global_codec_context = None  # Maintain state across all iterators

    for iterator_index, pd_it in enumerate(pd_it_it):
        if len(all_thumbs) > 10: # Stop after 10 rows
            break

        row_complete = False # Every iterator pd_it starts a new row
        thumbnails_row = []

        # Use state-aware decoding for each packet in the group
        for packet_data in pd_it:  # No enumerate() - preserve original packet numbers
            packet_no, packet = packet_data
            print(f'\n-----\n----Starting with iterator {iterator_index}, packet {packet_no}')
            
            frames, global_codec_context = decode_packet_to_frames_with_state(packet_data, global_codec_context)
            if len(frames) == 0:
                print(f"   No frames decoded from packet {packet_no}")
                continue
            print(f"   Decoded {len(frames)} frames from packet {packet_no}")

            thumbnails_row.extend(frame.to_image() for frame in frames)
            print(f"Adding {len(frames)} frames of packet {packet_no} to row {len(all_thumbs)}, current row length is {len(thumbnails_row)}")

            # Finished all frames in the packet
            if len(thumbnails_row) > thumbs_per_row:
                print(f"Cropping row {len(all_thumbs)} from {len(thumbnails_row)} frames to {thumbs_per_row} frames")
                thumbnails_row = thumbnails_row[:thumbs_per_row]
                row_complete = True
                break # Stop reading packets from this iterator
        
        # Exhausted all packets in the iterator pd_it , or finished a row
        if len(thumbnails_row) > 0 or row_complete:
            labels.append(f"iterator {iterator_index} , last packet {packet_no}")
            all_thumbs.append(thumbnails_row)
            thumbnails_row = []
            row_complete = False
            print(f"Early stop: Finished iterator {iterator_index} and row {len(all_thumbs)} with {len(thumbnails_row)} frames")
            continue # Continue with the next iterator
        if len(all_thumbs) > 4:
            break
        print(f"Finished iterator {iterator_index} and row {len(all_thumbs)} with {len(thumbnails_row)} frames")

    # Display thumbnails
    rows = len(all_thumbs)
    if rows>0:
        fig, axes = plt.subplots(rows, thumbs_per_row, figsize=(thumbs_per_row * 1.5, rows * 1.5))
        title = "Thumbnails (Global State)"
        fig.suptitle(title, fontsize=16)

        for i, row_thumbs in enumerate(all_thumbs):
            for j in range(thumbs_per_row):
                ax = axes[i, j] if rows > 1 else axes[j]
                ax.axis('off')
                if j == 0:
                    ax.set_title(labels[i], fontsize=10, loc='left')
                if j < len(row_thumbs):
                    ax.imshow(row_thumbs[j])

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    else:
        print("No thumbnails to display")


def display_thumbnails_stream_fresh_context(pd_it_it: packet_data_iterator_iterator, thumbs_per_row: int = 10):
    """
    Version that creates fresh decoder contexts for each iterator group.
    This avoids state conflicts between different groups.
    """
    all_thumbs = []
    labels = []

    for iterator_index, pd_it in enumerate(pd_it_it): # Get all iterators
        print(f"\n=== Processing iterator group {iterator_index} ===")
        
        if len(all_thumbs) >= 4: # Stop after 4 rows
            print(f"Reached maximum rows ({len(all_thumbs)}), stopping.")
            break

        thumbnails_row = []

        # Create a fresh decoder context for this group
        try:
            # Get the first packet to create a fresh codec context
            first_packet_data = next(pd_it)
            packet_no, packet = first_packet_data
            
            # Create a fresh codec context
            fresh_codec_context = packet.stream.codec_context
            
            # Process the first packet
            frames, _ = decode_packet_to_frames_with_state(first_packet_data, fresh_codec_context)
            if len(frames) > 0:
                thumbnails_row.extend(frame.to_image() for frame in frames)
                print(f"First packet {packet_no}: {len(frames)} frames, row length: {len(thumbnails_row)}")
            
            # Process remaining packets in the group
            for packet_data in pd_it:
                packet_no, packet = packet_data
                frames, _ = decode_packet_to_frames_with_state(packet_data, fresh_codec_context)
                
                if len(frames) > 0:
                    thumbnails_row.extend(frame.to_image() for frame in frames)
                    print(f"Packet {packet_no}: {len(frames)} frames, row length: {len(thumbnails_row)}")
                
                # Check if we have enough frames
                if len(thumbnails_row) >= thumbs_per_row:
                    thumbnails_row = thumbnails_row[:thumbs_per_row]
                    print(f"Reached target frames ({thumbs_per_row}), stopping this group.")
                    break
            
        except StopIteration:
            # Empty iterator group
            print(f"Iterator group {iterator_index} is empty.")
            continue
        except Exception as e:
            print(f"Error processing iterator {iterator_index}: {e}")
            continue
        
        # Add the row if we have any frames
        if len(thumbnails_row) > 0:
            labels.append(f"iterator {iterator_index} , last packet {packet_no}")
            all_thumbs.append(thumbnails_row)
            print(f"✓ Added row {len(all_thumbs)} with {len(thumbnails_row)} frames")
        else:
            print(f"✗ No frames in iterator group {iterator_index}, skipping.")
        
        print(f"Total rows so far: {len(all_thumbs)}")

    # Display thumbnails
    rows = len(all_thumbs)
    if rows > 0:
        fig, axes = plt.subplots(rows, thumbs_per_row, figsize=(thumbs_per_row * 1.5, rows * 1.5))
        title = "Thumbnails (Fresh Context)"
        fig.suptitle(title, fontsize=16)

        for i, row_thumbs in enumerate(all_thumbs):
            for j in range(thumbs_per_row):
                ax = axes[i, j] if rows > 1 else axes[j]
                ax.axis('off')
                if j == 0:
                    ax.set_title(labels[i], fontsize=10, loc='left')
                if j < len(row_thumbs):
                    ax.imshow(row_thumbs[j])

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    else:
        print("No thumbnails to display")


def display_thumbnails_stream_continuous(pd_it_it: packet_data_iterator_iterator, thumbs_per_row: int = 10):
    """
    Version that maintains continuous decoder state across all iterator groups.
    This should decode frames much better by preserving decoder state.
    """
    all_thumbs = []
    labels = []
    global_codec_context = None  # Maintain state across all groups

    for iterator_index, pd_it in enumerate(pd_it_it):
        print(f"\n=== Processing iterator group {iterator_index} ===")
        
        if len(all_thumbs) >= 4: # Stop after 4 rows
            print(f"Reached maximum rows ({len(all_thumbs)}), stopping.")
            break

        thumbnails_row = []

        # Process all packets in this group with continuous decoder state
        for packet_data in pd_it:
            packet_no, packet = packet_data
            
            # Use state-aware decoding with global context
            frames, global_codec_context = decode_packet_to_frames_with_state(packet_data, global_codec_context)
            
            if len(frames) > 0:
                thumbnails_row.extend(frame.to_image() for frame in frames)
                print(f"Packet {packet_no}: {len(frames)} frames, row length: {len(thumbnails_row)}")
            else:
                print(f"Packet {packet_no}: 0 frames")
            
            # Check if we have enough frames
            if len(thumbnails_row) >= thumbs_per_row:
                thumbnails_row = thumbnails_row[:thumbs_per_row]
                print(f"Reached target frames ({thumbs_per_row}), stopping this group.")
                break
        
        # Add the row if we have any frames
        if len(thumbnails_row) > 0:
            labels.append(f"iterator {iterator_index} , last packet {packet_no}")
            all_thumbs.append(thumbnails_row)
            print(f"✓ Added row {len(all_thumbs)} with {len(thumbnails_row)} frames")
        else:
            print(f"✗ No frames in iterator group {iterator_index}, skipping.")
        
        print(f"Total rows so far: {len(all_thumbs)}")

    # Display thumbnails
    rows = len(all_thumbs)
    if rows > 0:
        fig, axes = plt.subplots(rows, thumbs_per_row, figsize=(thumbs_per_row * 1.5, rows * 1.5))
        title = "Thumbnails (Continuous State)"
        fig.suptitle(title, fontsize=16)

        for i, row_thumbs in enumerate(all_thumbs):
            for j in range(thumbs_per_row):
                ax = axes[i, j] if rows > 1 else axes[j]
                ax.axis('off')
                if j == 0:
                    ax.set_title(labels[i], fontsize=10, loc='left')
                if j < len(row_thumbs):
                    ax.imshow(row_thumbs[j])

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    else:
        print("No thumbnails to display")


def display_thumbnails_stream_flattened(pd_it_it: packet_data_iterator_iterator, thumbs_per_row: int = 10):
    """
    Version that flattens all iterator groups into a single stream.
    This provides the best decoder state management by processing packets sequentially.
    """
    all_thumbs = []
    labels = []
    global_codec_context = None
    current_row = []
    current_group_start = None

    # Flatten all iterator groups into a single stream
    for iterator_index, pd_it in enumerate(pd_it_it):
        print(f"\n=== Processing iterator group {iterator_index} ===")
        
        if len(all_thumbs) >= 4: # Stop after 4 rows
            print(f"Reached maximum rows ({len(all_thumbs)}), stopping.")
            break

        # Process all packets in this group
        for packet_data in pd_it:
            packet_no, packet = packet_data
            
            # Track the start of this group
            if current_group_start is None:
                current_group_start = packet_no
            
            # Use state-aware decoding with global context
            frames, global_codec_context = decode_packet_to_frames_with_state(packet_data, global_codec_context)
            
            if len(frames) > 0:
                current_row.extend(frame.to_image() for frame in frames)
                print(f"Packet {packet_no}: {len(frames)} frames, row length: {len(current_row)}")
            else:
                print(f"Packet {packet_no}: 0 frames")
            
            # Check if we have enough frames for a row
            if len(current_row) >= thumbs_per_row:
                # Complete the current row
                row_thumbnails = current_row[:thumbs_per_row]
                labels.append(f"group {iterator_index}, packets {current_group_start}-{packet_no}")
                all_thumbs.append(row_thumbnails)
                print(f"✓ Added row {len(all_thumbs)} with {len(row_thumbnails)} frames")
                
                # Reset for next row
                current_row = current_row[thumbs_per_row:]  # Keep any remaining frames
                current_group_start = None
                
                # Check if we have enough rows
                if len(all_thumbs) >= 4:
                    print(f"Reached maximum rows ({len(all_thumbs)}), stopping.")
                    break
        
        # If we didn't complete a row in this group, but have some frames, add them
        if len(current_row) > 0 and current_group_start is not None:
            labels.append(f"group {iterator_index}, packets {current_group_start}-{packet_no}")
            all_thumbs.append(current_row)
            print(f"✓ Added partial row {len(all_thumbs)} with {len(current_row)} frames")
            current_row = []
            current_group_start = None
        
        print(f"Total rows so far: {len(all_thumbs)}")

    # Display thumbnails
    rows = len(all_thumbs)
    if rows > 0:
        fig, axes = plt.subplots(rows, thumbs_per_row, figsize=(thumbs_per_row * 1.5, rows * 1.5))
        title = "Thumbnails (Flattened Stream)"
        fig.suptitle(title, fontsize=16)

        for i, row_thumbs in enumerate(all_thumbs):
            for j in range(thumbs_per_row):
                ax = axes[i, j] if rows > 1 else axes[j]
                ax.axis('off')
                if j == 0:
                    ax.set_title(labels[i], fontsize=10, loc='left')
                if j < len(row_thumbs):
                    ax.imshow(row_thumbs[j])

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    else:
        print("No thumbnails to display")


def display_thumbnails_stream_simple(pd_it_it: packet_data_iterator_iterator, thumbs_per_row: int = 10):
    """
    Simple version that uses the proven decode_all_packets_with_flush approach.
    This should work reliably.
    """
    all_thumbs = []
    labels = []

    for iterator_index, pd_it in enumerate(pd_it_it):
        print(f"\n=== Processing iterator group {iterator_index} ===")
        
        if len(all_thumbs) >= 4: # Stop after 4 rows
            print(f"Reached maximum rows ({len(all_thumbs)}), stopping.")
            break

        thumbnails_row = []

        # Use the proven flush-based approach for this group
        for packet_no, frames in decode_all_packets_with_flush(pd_it):
            if len(frames) > 0:
                thumbnails_row.extend(frame.to_image() for frame in frames)
                print(f"Packet {packet_no}: {len(frames)} frames, row length: {len(thumbnails_row)}")
            else:
                print(f"Packet {packet_no}: 0 frames")
            
            # Check if we have enough frames
            if len(thumbnails_row) >= thumbs_per_row:
                thumbnails_row = thumbnails_row[:thumbs_per_row]
                print(f"Reached target frames ({thumbs_per_row}), stopping this group.")
                break
        
        # Add the row if we have any frames
        if len(thumbnails_row) > 0:
            labels.append(f"iterator {iterator_index} , last packet {packet_no}")
            all_thumbs.append(thumbnails_row)
            print(f"✓ Added row {len(all_thumbs)} with {len(thumbnails_row)} frames")
        else:
            print(f"✗ No frames in iterator group {iterator_index}, skipping.")
        
        print(f"Total rows so far: {len(all_thumbs)}")

    # Display thumbnails
    rows = len(all_thumbs)
    if rows > 0:
        fig, axes = plt.subplots(rows, thumbs_per_row, figsize=(thumbs_per_row * 1.5, rows * 1.5))
        title = "Thumbnails (Simple Flush-based)"
        fig.suptitle(title, fontsize=16)

        for i, row_thumbs in enumerate(all_thumbs):
            for j in range(thumbs_per_row):
                ax = axes[i, j] if rows > 1 else axes[j]
                ax.axis('off')
                if j == 0:
                    ax.set_title(labels[i], fontsize=10, loc='left')
                if j < len(row_thumbs):
                    ax.imshow(row_thumbs[j])

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    else:
        print("No thumbnails to display")


def display_thumbnails_stream_working(pd_it_it: packet_data_iterator_iterator, thumbs_per_row: int = 10):
    """
    Working version that processes each iterator group as a continuous stream.
    Based on debug findings: packet 0 works, others fail due to decoder state issues.
    """
    all_thumbs = []
    labels = []

    for iterator_index, pd_it in enumerate(pd_it_it):
        print(f"\n=== Processing iterator group {iterator_index} ===")
        
        if len(all_thumbs) >= 4: # Stop after 4 rows
            print(f"Reached maximum rows ({len(all_thumbs)}), stopping.")
            break

        thumbnails_row = []

        # Convert iterator group to list to avoid consuming it
        group_packets = list(pd_it)
        print(f"Group has {len(group_packets)} packets")
        
        if len(group_packets) == 0:
            print("Empty group, skipping.")
            continue
        
        # Process the entire group as a continuous stream
        try:
            for packet_no, frames in decode_all_packets_with_flush(group_packets):
                if packet_no == -1:
                    # These are flushed frames
                    print(f"Flushed frames: {len(frames)}")
                    thumbnails_row.extend(frame.to_image() for frame in frames)
                else:
                    # Regular packet frames
                    if len(frames) > 0:
                        thumbnails_row.extend(frame.to_image() for frame in frames)
                        print(f"Packet {packet_no}: {len(frames)} frames, row length: {len(thumbnails_row)}")
                    else:
                        print(f"Packet {packet_no}: 0 frames")
                
                # Check if we have enough frames
                if len(thumbnails_row) >= thumbs_per_row:
                    thumbnails_row = thumbnails_row[:thumbs_per_row]
                    print(f"Reached target frames ({thumbs_per_row}), stopping this group.")
                    break
                    
        except Exception as e:
            print(f"Error processing group {iterator_index}: {e}")
            continue
        
        # Add the row if we have any frames
        if len(thumbnails_row) > 0:
            labels.append(f"iterator {iterator_index} , packets {len(group_packets)}")
            all_thumbs.append(thumbnails_row)
            print(f"✓ Added row {len(all_thumbs)} with {len(thumbnails_row)} frames")
        else:
            print(f"✗ No frames in iterator group {iterator_index}, skipping.")
        
        print(f"Total rows so far: {len(all_thumbs)}")

    # Display thumbnails
    rows = len(all_thumbs)
    if rows > 0:
        fig, axes = plt.subplots(rows, thumbs_per_row, figsize=(thumbs_per_row * 1.5, rows * 1.5))
        title = "Thumbnails (Working Version)"
        fig.suptitle(title, fontsize=16)

        for i, row_thumbs in enumerate(all_thumbs):
            for j in range(thumbs_per_row):
                ax = axes[i, j] if rows > 1 else axes[j]
                ax.axis('off')
                if j == 0:
                    ax.set_title(labels[i], fontsize=10, loc='left')
                if j < len(row_thumbs):
                    ax.imshow(row_thumbs[j])

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    else:
        print("No thumbnails to display")


def display_thumbnails_from_frames(frame_groups: List[frame_list_type], thumbs_per_row: int = 10, sampling_strategy: str = "first"):
    """
    Display thumbnails from a list of frame groups.
    Each group is a list of frames that were decoded together.
    
    Args:
        frame_groups: List of frame lists, where each inner list contains frames from one group
        thumbs_per_row: Maximum number of thumbnails per row
        sampling_strategy: "first" for first N frames, "bookend" for first N/2 + last N/2 frames
    """
    all_thumbs = []
    labels = []

    for group_index, frames in enumerate(frame_groups):
        if len(all_thumbs) >= 4:  # Stop after 4 rows
            print(f"Reached maximum rows ({len(all_thumbs)}), stopping.")
            break
        
        if not frames:
            print(f"Group {group_index}: No frames, skipping.")
            continue
        
        # Select frames based on sampling strategy
        if sampling_strategy == "bookend" and len(frames) > thumbs_per_row:
            # Take first half and last half
            half = thumbs_per_row // 2
            chunk = frames[:half] + frames[-half:]
        else:
            # Take first N frames (default behavior)
            chunk = frames[:thumbs_per_row]
        
        # Convert frames to thumbnails
        thumbs = []
        for frame in chunk:
            try:
                rgb_frame = frame.to_ndarray(format='rgb24')
                thumb = rgb_frame[::4, ::4]  # Simple downsampling
                thumbs.append(thumb)
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
        if thumbs:
            all_thumbs.append(thumbs)
            # Label shows the full group PTS/timecode range
            first_frame = frames[0]
            last_frame = frames[-1]
            first_pts = first_frame.pts if first_frame.pts is not None else 0
            last_pts = last_frame.pts if last_frame.pts is not None else 0
            try:
                time_base = first_frame.time_base
                first_time = first_pts * time_base if time_base else 0
                last_time = last_pts * time_base if time_base else 0
                timecode_str = f" ({first_time:.3f}s-{last_time:.3f}s)"
            except:
                timecode_str = ""
            
            # Add sampling info to label
            if sampling_strategy == "bookend" and len(frames) > thumbs_per_row:
                sample_info = f" [first{thumbs_per_row//2}+last{thumbs_per_row//2}]"
            else:
                sample_info = f" [first{len(chunk)}]"
            
            labels.append(f"group {group_index}, PTS {first_pts}-{last_pts}{timecode_str}{sample_info}")
    
    if not all_thumbs:
        print("No thumbnails to display")
        return
    
    # Display thumbnails
    rows = len(all_thumbs)
    if rows > 0:
        fig, axes = plt.subplots(rows, thumbs_per_row, figsize=(thumbs_per_row * 1.5, rows * 1.5))
        title = "Thumbnails (Seek-and-Decode with PTS)"
        fig.suptitle(title, fontsize=16)

        for i, row_thumbs in enumerate(all_thumbs):
            for j in range(thumbs_per_row):
                ax = axes[i, j] if rows > 1 else axes[j]
                ax.axis('off')
                if j == 0:
                    ax.set_title(labels[i], fontsize=10, loc='left')
                if j < len(row_thumbs):
                    ax.imshow(row_thumbs[j])

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    else:
        print("No thumbnails to display")
