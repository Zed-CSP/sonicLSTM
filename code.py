import gym
import retro
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv2D, Flatten, MaxPooling2D
import cv2

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
    
    def preprocess_frame(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84))
        # Normalize
        normalized = resized / 255.0
        return normalized

def main():
    # Initialize the Sonic environment
    env = retro.make(game='SonicTheHedgehog-Genesis')
    
    # Create the LSTM agent
    agent = SonicLSTM()
    
    # Training loop
    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Preprocess the current frame
            processed_state = agent.preprocess_frame(state)
            
            # Get action from model
            action = agent.model.predict(processed_state.reshape(1, 1, 84, 84, 3))
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            state = next_state
            
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

if __name__ == "__main__":
    main()
