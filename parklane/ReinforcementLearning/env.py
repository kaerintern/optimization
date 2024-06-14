import gym
import numpy as np
import pandas as pd
import warnings
from gym import spaces
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

warnings.filterwarnings("ignore")

# Load dataset
dataset = pd.read_csv(r"/Users/admin/Desktop/optimization/parklane/ReinforcementLearning/sample_train.csv")

# Reward function
def calculate_reward(ch_sysef, lift, ct):
    alpha = 100
    beta = 0.1
    gamma = 0.1
    reward = -(alpha * ch_sysef + beta * lift + gamma * ct)
    
    return reward

# Environment
class BuildingEnv(gym.Env):
    def __init__(self):
        super(BuildingEnv, self).__init__()

        self.action_space = spaces.Discrete(21)
        self.observation_space = spaces.Box(
            low=np.array([27, 22, 3, 0.4], dtype=np.float32), 
            high=np.array([30, 26, 25, 0.65], dtype=np.float32)
        )

        self.dataset = dataset
        self.current_index = 0
        self.reset()
        
    def step(self, action):
        self._take_action(action)
        reward = self._calculate_reward()
        self.current_index = (self.current_index + 1) % len(self.dataset)
        self.state = self._get_state()
        done = self.current_index == 0
        return self.state, reward, done, {}
    
    def reset(self):
        self.current_index = 0
        self.state = self._get_initial_state()  
        return self.state
    
    def _take_action(self, action):
        cwst_changes = np.linspace(-1, 1, num=21)
        cwst_change = cwst_changes[action]
        self.state[0] += cwst_change
        self.state[0] = np.clip(self.state[0], 27, 30)

    def _calculate_reward(self):
        _, lift, ct_tot_kw, ch_sysef = self.state
        reward = calculate_reward(ch_sysef, lift, ct_tot_kw)
        return reward
    
    def _get_state(self):
        row = self.dataset.iloc[self.current_index]
        return np.array([row['h_cwst'], row['lift'], row['ct_tot_kw'], row['ch_sysef']])

    def _get_initial_state(self):
        row = dataset.iloc[0]
        return np.array([row['h_cwst'], row['lift'], row['ct_tot_kw'], row['ch_sysef']])

# DQN Agent
class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.010
        self.epsilon_decay = 0.995
        self.memory_max = 2000

        self.policy_model = self._build_model()
        self.target_model = self._build_model()

        self.policy_model_optimizer = optim.Adam(self.policy_model.parameters(), lr=0.001)
        self.policy_criterion = nn.MSELoss()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def update_target_model(self):
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()
    
    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_max:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.policy_model(state)
        return torch.argmax(act_values[0]).item()
    
    def replay(self, batch_size):
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        for i in minibatch:

            state, action, reward, next_state, done = self.memory[i]
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            reward = torch.FloatTensor([reward])

            with torch.no_grad():
                target = self.policy_model(state)
                if done:
                    target[0][0][action] = reward
                else:
                    t = self.target_model(next_state)[0][0].max()
                    target[0][0][action] = torch.FloatTensor(reward + self.gamma * t).unsqueeze(0)

            self.policy_model_optimizer.zero_grad()
            '''loss = self.criterion(self.policy_model(state), target)
            loss.backward()'''
            self.policy_model_optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Main
env = BuildingEnv()
state_size = 4
action_size = 21
agent = DQNAgent(state_size, action_size)


# Training
num_episodes = 100
batch_size = 20
reward_history = []

for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    # state 
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        # move to next state
        state = next_state
          
        if done:
            agent.update_target_model()
            break
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        
    reward_history.append(total_reward)
    print(f"Episode: {episode+1}/{num_episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

# Plot rewards
plt.plot(reward_history)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
print('Training done')
