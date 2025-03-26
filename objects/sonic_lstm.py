import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv2D, Flatten, MaxPooling2D
from .preprocessing import preprocess_frame, prepare_batch

class SonicLSTM:
    def __init__(self, input_shape=(84, 84, 3), sequence_length=4):
        self.sequence_length = sequence_length
        self.input_shape = input_shape
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential([
            # CNN layers to process game frames
            Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=self.input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (4, 4), strides=2, activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), strides=1, activation='relu'),
            Flatten(),
            
            # LSTM layer for temporal dependencies
            LSTM(512, return_sequences=True),
            LSTM(256),
            
            # Output layer for actions
            Dense(12, activation='softmax')  # 12 possible actions in Sonic
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model
    
    def process_frame(self, frame):
        """Process a single frame using the preprocessing module."""
        return preprocess_frame(frame)
    
    def process_batch(self, frames):
        """Process a batch of frames using the preprocessing module."""
        return prepare_batch(frames, self.sequence_length) 