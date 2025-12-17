import cv2
import numpy as np
import tempfile
import os

def process_video_for_traffic_speed(video_path, target_seq_len=12):
    """
    Extracts a proxy for traffic speed from a video file using dense optical flow.
    
    How it works:
    1. Reads the video frame by frame.
    2. Compares consecutive frames to see how much pixels moved (Optical Flow).
    3. Faster movement = Higher traffic speed.
    4. Averages this movement over time to get a sequence of data points.
    
    Args:
        video_path (str): Path to the video file.
        target_seq_len (int): How many data points to generate (e.g., 12 points = 1 hour).
        
    Returns:
        np.array: Extracted speed signal.
    """
    print(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error opening video file")
    
    # Read first frame
    ret, frame1 = cap.read()
    if not ret:
        return np.zeros(target_seq_len)
        
    # --- Optimization: Resize for Speed ---
    # Analyzing specific movement of every pixel in a large video (e.g., 1920x1080) is very slow.
    # We resize the video to a smaller width (320px). This makes processing 10-20x faster
    # while still capturing the general movement of cars.
    height, width = frame1.shape[:2]
    new_width = 320
    new_height = int(height * (new_width / width))
    
    frame1 = cv2.resize(frame1, (new_width, new_height))
    
    # Convert to grayscale because color doesn't help with movement detection
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    frame_speeds = []
    frame_count = 0
    skip_frames = 2  # Process every 3rd frame (skip 2) to speed it up further
    
    print(f"Video processing optimized: Resizing to {new_width}x{new_height}, skipping {skip_frames} frames.")
    
    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
            
        frame_count += 1
        # Skip frames to save time
        if frame_count % (skip_frames + 1) != 0:
            continue
            
        # Resize current frame to match the first one
        frame2 = cv2.resize(frame2, (new_width, new_height))
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # --- Optical Flow Calculation ---
        # unique math method (Farneback) that calculates the direction and speed
        # of every moving pixel between two frames.
        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Convert (x, y) movement to Magnitude (Speed) and Angle (Direction)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Average the speed of all pixels in this frame.
        # Ideally, we would mask out the sky/trees, but for this demo, the average works well enough.
        avg_speed = np.mean(mag)
        frame_speeds.append(avg_speed)
        
        prvs = next_frame
        
    cap.release()
    
    if not frame_speeds:
        return np.zeros(target_seq_len)
        
    # --- Resampling ---
    # The video might have 300 frames, but our AI model needs exactly 'target_seq_len' inputs (12).
    # We split the video data into 12 chunks and average each chunk.
    frame_speeds = np.array(frame_speeds)
    
    if len(frame_speeds) < target_seq_len:
        # If video is too short, stretch the data points
        indices = np.linspace(0, len(frame_speeds)-1, target_seq_len)
        resampled = np.interp(indices, np.arange(len(frame_speeds)), frame_speeds)
    else:
        # Bin averaging: specific frames -> 1 time step
        chunks = np.array_split(frame_speeds, target_seq_len)
        resampled = np.array([np.mean(chunk) for chunk in chunks])
        
    # --- Normalization ---
    # Optical flow gives speed in "pixels per frame".
    # Real traffic data is in "miles per hour" (e.g., 20-65 mph).
    # We map the pixel speed to a realistic traffic range (5 mph to 65 mph) so the AI understands it.
    if np.max(resampled) > 0:
        # Scale to 0-1 range first
        resampled_norm = (resampled - np.min(resampled)) / (np.max(resampled) - np.min(resampled) + 1e-5)
        # Then scale to 5-65 range
        resampled_scaled = resampled_norm * 60 + 5 
    else:
        resampled_scaled = resampled
        
    return resampled_scaled
