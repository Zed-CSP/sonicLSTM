import retro
from objects.sonic_lstm import SonicLSTM

def train_agent(episodes=1000):
    # Initialize the Sonic environment
    env = retro.make(game='SonicTheHedgehog-Genesis')
    
    # Create the LSTM agent
    agent = SonicLSTM()
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Process the current frame
            processed_state = agent.process_frame(state)
            
            # Get action from model
            action = agent.model.predict(processed_state.reshape(1, 1, 84, 84, 3))
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            state = next_state
            
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

if __name__ == "__main__":
    train_agent() 