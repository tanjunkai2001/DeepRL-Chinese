"""
This script implements a Deep Q-Network (DQN) agent to solve the CartPole-v0 environment from OpenAI Gym.
"""

import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- Hyperparameters ---
BATCH_SIZE = 128            # Number of transitions sampled from the replay buffer
GAMMA = 0.99                # Discount factor for future rewards
EPS_START = 0.9             # Initial value of epsilon in the epsilon-greedy policy
EPS_END = 0.05              # Final value of epsilon
EPS_DECAY = 1000            # Controls the rate of epsilon decay
TARGET_UPDATE = 10          # How often to update the target network (in episodes)
REPLAY_MEMORY_SIZE = 10000  # Size of the replay memory buffer
LR = 1e-4                   # Learning rate for the optimizer
NUM_EPISODES = 600          # Total number of episodes to train for

# --- Replay Memory ---
# A named tuple to represent a single transition in our environment
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    """A cyclic buffer of bounded size that holds the transitions observed recently."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a random batch of transitions for training"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- DQN Model ---
class DQN(nn.Module):
    """
    The Deep Q-Network model. It is a simple feedforward neural network that takes the
    state of the environment as input and outputs the expected Q-values for each possible action.
    """
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# --- Main Training Components ---
# Set up the environment
env = gym.make("CartPole-v0")

# Set up the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the number of actions and observations from the environment
n_actions = env.action_space.n
state, _ = env.reset()
n_observations = len(state)

# Initialize the policy network and the target network
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Set the target network to evaluation mode

# Initialize the optimizer
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

# Initialize the replay memory
memory = ReplayMemory(REPLAY_MEMORY_SIZE)

steps_done = 0

def select_action(state):
    """Selects an action using an epsilon-greedy policy."""
    global steps_done
    sample = random.random()
    # Calculate epsilon based on the current step
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    # Exploitation: choose the best action from the policy network
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    # Exploration: choose a random action
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

episode_durations = []

def plot_durations(show_result=False):
    """A helper function to plot the duration of episodes during training."""
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def optimize_model():
    """Performs a single step of the optimization process."""
    if len(memory) < BATCH_SIZE:
        return  # Not enough memory for a batch, so we skip optimization
        
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for details).
    # This converts batch-array of Transitions to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        
    # Compute the expected Q values: R + γ * max_a' Q_target(s', a')
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# --- Training Loop ---
for i_episode in range(NUM_EPISODES):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    for t in range(1000): # Limit episode length
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
            
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()

# --- Visualization of the Trained Agent ---
print("\n--- Starting visualization of the trained agent ---")
env = gym.make("CartPole-v0", render_mode="human")
for i in range(5): # Run 5 episodes for visualization
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in range(500): # Max steps per episode for visualization
        env.render()
        
        # Select action based on the trained policy network (no exploration)
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1)
        
        observation, reward, terminated, truncated, _ = env.step(action.item())
        
        if terminated or truncated:
            print(f"Episode {i+1} finished after {t+1} timesteps")
            break
            
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

env.close()