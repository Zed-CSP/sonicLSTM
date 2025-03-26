# Sonic LSTM

An LSTM-based AI agent that learns to play Sonic the Hedgehog using deep reinforcement learning.

## Project Structure

```
sonicLSTM/
├── docs/                # Documentation
│   └── bayesian_optimization.md
├── objects/              # Core model and preprocessing components
│   ├── sonic_lstm.py    # LSTM model architecture
│   └── preprocessing.py # Frame preprocessing utilities
├── scripts/             # Executable scripts
│   └── train.py        # Training loop implementation
├── trained_models/     # Directory for storing trained model weights
└── requirements.txt    # Project dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up the Sonic ROM:
   - Place your Sonic the Hedgehog ROM file in the appropriate Retro Gym directory
   - The ROM should be named `SonicTheHedgehog-Genesis`

## Usage

To train the agent:
```bash
python code.py
```

## Model Architecture

The agent uses a hybrid architecture combining CNN and LSTM layers to process both spatial and temporal information:

### CNN Layers
- Input: 84x84x3 grayscale frames
- Layer 1: Conv2D(32, 8x8, stride=4) + ReLU + MaxPool(2x2)
- Layer 2: Conv2D(64, 4x4, stride=2) + ReLU + MaxPool(2x2)
- Layer 3: Conv2D(64, 3x3, stride=1) + ReLU
- Flatten layer to prepare for LSTM input

### LSTM Layers
- First LSTM: 512 units with return_sequences=True
- Second LSTM: 256 units
- Output: 12 units (softmax) for action prediction

### Training Configuration
- Optimizer: Adam
- Loss Function: Categorical Cross-entropy
- Batch Size: 32
- Learning Rate: 0.001

## Hyperparameter Tuning

### Search Space
```python
hyperparameters = {
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64],
    'lstm_units': [(256, 128), (512, 256), (1024, 512)],
    'sequence_length': [3, 4, 5],
    'cnn_filters': [(32, 64, 64), (64, 128, 128), (128, 256, 256)],
    'dropout_rate': [0.1, 0.2, 0.3]
}
```

### Tuning Process
1. **Grid Search**:
   - Test all combinations of hyperparameters
   - Evaluate each configuration over 50 episodes
   - Select best performing configuration

2. **Bayesian Optimization**:
   - Detailed implementation and results available in [Bayesian Optimization Documentation](docs/bayesian_optimization.md)
   - Uses Gaussian Process for parameter optimization
   - Focus on most promising regions of search space
   - Early stopping for poorly performing configurations

3. **Best Parameters Found**:
   - Learning Rate: 0.001
   - Batch Size: 32
   - LSTM Units: (512, 256)
   - Sequence Length: 4
   - CNN Filters: (32, 64, 64)
   - Dropout Rate: 0.2

### Parameter Sensitivity Analysis
- Learning rate: Most sensitive parameter
- Sequence length: Moderate impact on performance
- Batch size: Less sensitive but affects training stability
- LSTM units: Significant impact on model capacity

## Visualization and Monitoring

### Training Metrics Visualization
```python
# Example plotting code
import matplotlib.pyplot as plt

def plot_training_metrics(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot episode rewards
    ax1.plot(history['episode_rewards'])
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    
    # Plot ring collection rate
    ax2.plot(history['ring_rate'])
    ax2.set_title('Ring Collection Rate')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Rings/Minute')
```

### Real-time Monitoring
- TensorBoard integration for live metrics
- Frame-by-frame visualization of agent behavior
- Action distribution heatmaps
- Reward decomposition plots

### Performance Analysis
- Learning curves showing convergence
- Reward distribution histograms
- Action frequency analysis
- State value estimation plots

## Reward Structure

The agent receives rewards based on multiple factors:

### Primary Rewards
- Forward Progress: +1 for moving right
- Ring Collection: +10 per ring
- Score Points: +5 per 1000 points
- Zone Completion: +50 for completing a zone

### Penalties
- Death: -50
- Standing Still: -0.1 per frame
- Moving Left: -0.5 per frame
- Time Penalty: -0.1 per second

### Reward Shaping
- Rewards are normalized to [-1, 1] range
- Exponential decay applied to long-term rewards
- Reward clipping to prevent extreme values

## Model Evaluation

### Training Metrics
- Episode Reward: Total reward per episode
- Average Score: Mean game score achieved
- Zone Progress: Number of zones completed
- Ring Collection Rate: Rings collected per minute
- Survival Time: Average time per episode

### Evaluation Protocol
1. Run 100 evaluation episodes
2. Calculate mean and standard deviation of metrics
3. Compare against baseline (random policy)
4. Visualize learning curves

### Performance Benchmarks
- Minimum acceptable score: 5000 points
- Target ring collection: 50 rings per episode
- Expected zone completion: 1 zone per 10 episodes
- Average survival time: 3 minutes

## Preprocessing

Each game frame is preprocessed by:
1. Converting to grayscale
2. Resizing to 84x84 pixels
3. Normalizing pixel values to [0, 1]

## Training

The agent is trained through episodes where it:
1. Processes the current game state
2. Predicts the next action
3. Receives reward feedback
4. Updates its policy based on the experience
