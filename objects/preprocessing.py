import cv2
import numpy as np

def preprocess_frame(frame, target_size=(84, 84)):
    """
    Preprocess a game frame for the model.
    
    Args:
        frame: The input frame from the game
        target_size: The desired size of the processed frame (height, width)
    
    Returns:
        Processed frame as a numpy array
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Resize to target size
    resized = cv2.resize(gray, target_size)
    
    # Normalize pixel values to [0, 1]
    normalized = resized / 255.0
    
    return normalized

def prepare_batch(frames, sequence_length=4):
    """
    Prepare a batch of frames for the model.
    
    Args:
        frames: List of preprocessed frames
        sequence_length: Number of frames to include in each sequence
    
    Returns:
        Batch of sequences ready for model input
    """
    sequences = []
    for i in range(len(frames) - sequence_length + 1):
        sequence = frames[i:i + sequence_length]
        sequences.append(sequence)
    
    return np.array(sequences) 