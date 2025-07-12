import av
import matplotlib.pyplot as plt
from typing import List, Iterator
from vidfile_iterator import decode_all_packets_with_flush, packet_data_iterator_iterator,decode_packet_to_frames_with_state, frame_list_type
import random


def reservoir_sample_frames(frame_iterator: Iterator, sample_size: int) -> List:
    """
    Reservoir sampling for frames - maintains streaming model.
    Randomly samples N frames from a stream of frames.
    
    Args:
        frame_iterator: Iterator yielding frames
        sample_size: Number of frames to sample
    
    Returns:
        List of randomly sampled frames (sorted by PTS)
    """
    sampled_frames = []
    frame_count = 0
    
    for frame in frame_iterator:
        frame_count += 1
        
        if len(sampled_frames) < sample_size:
            # Fill the reservoir initially
            sampled_frames.append(frame)
        else:
            # Randomly replace with probability sample_size/frame_count
            if random.random() < sample_size / frame_count:
                # Replace a random frame in the reservoir
                replace_index = random.randint(0, sample_size - 1)
                sampled_frames[replace_index] = frame
    
    # Sort by PTS to maintain temporal order in display
    sampled_frames.sort(key=lambda f: f.pts if f.pts is not None else 0)
    return sampled_frames


def display_thumbnail_grid(all_thumbs: List[List], labels: List[str], thumbs_per_row: int = 10, title: str = "Thumbnails"):
    """
    Display a grid of thumbnails using matplotlib.
    
    Args:
        all_thumbs: List of thumbnail rows, where each row is a list of thumbnail arrays
        labels: List of labels for each row
        thumbs_per_row: Maximum number of thumbnails per row
        title: Title for the plot
    """
    if not all_thumbs:
        print("No thumbnails to display")
        return
    
    rows = len(all_thumbs)
    if rows > 0:
        fig, axes = plt.subplots(rows, thumbs_per_row, figsize=(thumbs_per_row * 1.5, rows * 1.5))
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


def convert_frames_to_thumbnails(frames: List, downsample_factor: int = 4) -> List:
    """
    Convert a list of frames to thumbnail arrays.
    
    Args:
        frames: List of video frames to convert
        downsample_factor: Factor to downsample thumbnails (default: 4)
    
    Returns:
        List of thumbnail arrays (numpy arrays)
    """
    thumbs = []
    for frame in frames:
        try:
            # Convert frame to RGB and resize
            rgb_frame = frame.to_ndarray(format='rgb24')
            # Resize to thumbnail size using downsampling
            thumb = rgb_frame[::downsample_factor, ::downsample_factor]
            thumbs.append(thumb)
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue
    return thumbs


def process_frame_group_for_display(frames: List, chunk: List, group_index: int, sampling_strategy: str, thumbs_per_row: int):
    """
    Process a frame group for display: convert frames to thumbnails and create the label.
    
    Args:
        frames: Complete list of frames in the group
        chunk: Selected frames for display (based on sampling strategy)
        group_index: Index of the group
        sampling_strategy: Sampling strategy used ("first", "bookend", "random")
        thumbs_per_row: Maximum thumbnails per row
    
    Returns:
        Tuple of (thumbnails_list, label_string) or (None, None) if no thumbnails
    """
    # Convert frames to thumbnails
    thumbs = convert_frames_to_thumbnails(chunk)
    
    if not thumbs:
        return None, None
    
    # Extract PTS and timecode information from the complete group
    first_frame = frames[0]
    last_frame = frames[-1]
    first_pts = first_frame.pts if first_frame.pts is not None else 0
    last_pts = last_frame.pts if last_frame.pts is not None else 0
    
    # Convert PTS to timecode if possible
    try:
        time_base = first_frame.time_base
        first_time = first_pts * time_base if time_base else 0
        last_time = last_pts * time_base if time_base else 0
        timecode_str = f" ({first_time:.3f}s-{last_time:.3f}s)"
    except:
        timecode_str = ""
    
    # Create sampling strategy description for the label
    if sampling_strategy == "bookend" and len(frames) > thumbs_per_row:
        sample_info = f" [first{thumbs_per_row//2}+last{thumbs_per_row//2}]"
    elif sampling_strategy == "random" and len(frames) > thumbs_per_row:
        sample_info = f" [random{len(chunk)}]"
    else:
        sample_info = f" [first{len(chunk)}]"
    
    # Create the complete label
    label = f"group {group_index}, PTS {first_pts}-{last_pts}{timecode_str}{sample_info}"
    
    return thumbs, label


def display_thumbnails_from_frames(frame_groups: List[frame_list_type], thumbs_per_row: int = 10, sampling_strategy: str = "first"):
    """
    Display thumbnails from a list of frame groups.
    Each group is a list of frames that were decoded together.
    
    Args:
        frame_groups: List of frame lists, where each inner list contains frames from one group
        thumbs_per_row: Maximum number of thumbnails per row
        sampling_strategy: "first" for first N frames, "bookend" for first N/2 + last N/2 frames, "random" for random N frames
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
        elif sampling_strategy == "random" and len(frames) > thumbs_per_row:
            # Use reservoir sampling for true streaming random sampling
            chunk = reservoir_sample_frames(iter(frames), thumbs_per_row)
        else:
            # Take first N frames (default behavior)
            chunk = frames[:thumbs_per_row]
        
        # Process the frame group for display (convert to thumbnails and create label)
        thumbs, label = process_frame_group_for_display(frames, chunk, group_index, sampling_strategy, thumbs_per_row)
        
        if thumbs and label:
            all_thumbs.append(thumbs)
            labels.append(label)
    
    # Display thumbnails using the extracted function
    title = f"Thumbnails ({sampling_strategy.capitalize()} Sampling)"
    display_thumbnail_grid(all_thumbs, labels, thumbs_per_row, title)
