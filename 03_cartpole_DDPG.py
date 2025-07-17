"""
This script implements a Deep Deterministic Policy Gradient (DDPG) agent 
to solve the Pendulum-v1 environment from OpenAI Gym.
"""
import gym
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
TAU = 0.005                 # Soft update parameter for target networks
LR_ACTOR = 1e-4             # Learning rate for the actor network
LR_CRITIC = 1e-3            # Learning rate for the critic network
REPLAY_MEMORY_SIZE = 100000 # Size of the replay memory buffer
NUM_EPISODES = 200          # Total number of episodes to train for
WEIGHT_DECAY = 1e-2         # L2 weight decay for the critic optimizer

# --- Environment Setup ---
# Using Pendulum-v1 for its continuous action space
env = gym.make("Pendulum-v1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Ornstein-Uhlenbeck Noise ---
# A noise process for exploration in continuous action spaces.
class OUNoise:
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state, dtype=torch.float32).to(device)

# --- Replay Memory ---
# Same as in the DQN implementation, stores (state, action, reward, next_state) tuples.
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- Actor-Critic Network Definitions ---
class Actor(nn.Module):
    def __init__(self, n_observations, n_actions, action_scale):
        super(Actor, self).__init__()
        self.action_scale = torch.tensor(action_scale, dtype=torch.float32).to(device)
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # Use tanh to bound the action output to [-1, 1], then scale to the env's action range
        return torch.tanh(self.layer3(x)) * self.action_scale

class Critic(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Critic, self).__init__()
        # Q-network: state and action go in, Q-value comes out
        self.layer1 = nn.Linear(n_observations + n_actions, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, state, action):
        # Concatenate state and action along the feature dimension
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DDPGAgent:
    def __init__(self, n_observations, n_actions, action_scale):
        # Actor Networks
        self.actor = Actor(n_observations, n_actions, action_scale).to(device)
        self.actor_target = Actor(n_observations, n_actions, action_scale).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        # Critic Networks
        self.critic = Critic(n_observations, n_actions).to(device)
        self.critic_target = Critic(n_observations, n_actions).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.memory = ReplayMemory(REPLAY_MEMORY_SIZE)
        self.noise = OUNoise(n_actions)

    def select_action(self, state):
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)
        self.actor.train()
        action += self.noise.sample()
        # Clip the action to be within the environment's valid range
        return torch.clamp(action, -self.actor.action_scale, self.actor.action_scale)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        # --- Update Critic ---
        # 1. Get next actions from the target actor
        with torch.no_grad():
            next_actions = self.actor_target(next_state_batch)
            # 2. Get Q-values for next states from the target critic
            target_q_values = self.critic_target(next_state_batch, next_actions)
            # 3. Compute the Bellman target for Q-values: y = r + γ * Q'(s', μ'(s'))
            y = reward_batch + GAMMA * target_q_values

        # 4. Get current Q-values from the main critic
        current_q_values = self.critic(state_batch, action_batch)

        # 5. Compute critic loss and update the critic network
        critic_loss = F.mse_loss(current_q_values, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update Actor ---
        # 1. Get action predictions from the main actor
        predicted_actions = self.actor(state_batch)
        # 2. Compute actor loss: We want to maximize Q(s, μ(s)), so we minimize -Q(s, μ(s))
        actor_loss = -self.critic(state_batch, predicted_actions).mean()

        # 3. Update the actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft Update Target Networks ---
        self.soft_update(self.critic_target, self.critic, TAU)
        self.soft_update(self.actor_target, self.actor, TAU)

    def soft_update(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

# --- Main Training Loop ---
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
action_scale = env.action_space.high[0] # The action space is symmetric, e.g., [-2, 2]

agent = DDPGAgent(n_observations, n_actions, action_scale)
episode_rewards = []

print("--- Starting Training ---")
for i_episode in range(NUM_EPISODES):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    agent.noise.reset()
    total_reward = 0

    for t in range(env._max_episode_steps):
        action = agent.select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.cpu().detach().numpy().flatten())
        
        total_reward += reward
        done = terminated or truncated

        reward = torch.tensor([reward], dtype=torch.float32, device=device).view(1, 1)
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        agent.memory.push(state, action, reward, next_state)
        state = next_state
        
        agent.optimize_model()

        if done:
            break
            
    episode_rewards.append(total_reward)
    if (i_episode + 1) % 10 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {i_episode+1}/{NUM_EPISODES} | Total Reward: {total_reward:.2f} | Avg Reward (last 100): {avg_reward:.2f}")

print("--- Training Complete ---")

# Plotting the results
plt.figure()
plt.plot(episode_rewards)
plt.title('Episode Rewards over Time')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.show()

# --- Visualization of the Trained Agent (Final Test) ---
print("\n--- Starting visualization of the trained agent ---")
test_env = gym.make("Pendulum-v1", render_mode="human")
for i in range(5): # Run 5 episodes for visualization
    state, info = test_env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    episode_reward = 0
    for t in range(test_env._max_episode_steps):
        test_env.render()
        
        # Select action deterministically from the trained actor (no noise)
        with torch.no_grad():
            action = agent.actor(state)
        
        observation, reward, terminated, truncated, _ = test_env.step(action.cpu().detach().numpy().flatten())
        episode_reward += reward
        
        if terminated or truncated:
            break
            
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    print(f"Test Episode {i+1} | Total Reward: {episode_reward:.2f}")

test_env.close()