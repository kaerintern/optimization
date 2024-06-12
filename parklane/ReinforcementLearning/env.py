import gym
import numpy as np
import pandas as pd
import warnings
from gym import spaces
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras import Sequential
warnings.filterwarnings("ignore")


dataset = pd.read_csv(r"/Users/admin/Desktop/optimization/parklane/ReinforcementLearning/train.csv")

def calculate_reward(ch_sysef, lift, ct):
    alpha = 1
    beta = 0.1
    gamma = 0.01

    reward = -(alpha * ch_sysef + beta * lift + gamma * ct)

    return reward

def discretize_state(state, state_bins):
    return tuple(np.digitize(state[i], state_bins[i]) for i in range(len(state)))


class BuildingEnv(gym.Env):

    def __init__(self):
        super(BuildingEnv, self).__init__()

        # Action Space
        self.action_space = spaces.Discrete(21)

        # State space: [CWST, Lift, Cooling Tower (KW), ch_sysef]
        self.observation_space = spaces.Box(
            low=np.array([27, 22, 3, 0.4], dtype=np.float32), 
            high=np.array([30, 26, 25, 0.65], dtype=np.float32)
        )

        self.dataset = dataset
        self.current_index = 0

        self.reset()
        
    def step(self, action):
        # Apply action
        self._take_action(action)

        # Calculate reward
        reward = self._calculate_reward()

        # Get new state
        self.current_index = (self.current_index + 1) % len(self.dataset)
        self.state = self._get_state()
        
        done = self.current_index == 0

        return self.state, reward, done, {}
    
    def reset(self):
        self.current_index = 0
        self.state = self._get_initial_state()  

        return self.state
    
    def _take_action(self, action):
        # Map action to CWST adjustment
        cwst_changes = np.linspace(-1, 1, num=21)
        cwst_change = cwst_changes[action]

        # Apply change
        self.state[0] += cwst_change
        print(self.state[0])
        # Ensure CWST stays within bounds
        self.state[0] = np.clip(self.state[0], 27, 30)

    def _calculate_reward(self):
        cwst, lift, ct_tot_kw, ch_sysef = self.state
        reward = calculate_reward(ch_sysef, lift, ct_tot_kw)

        return reward
    
    def _get_state(self):
        row = self.dataset.iloc[self.current_index]
        print(np.array([row['h_cwst'], row['lift'], row['ct_tot_kw'], row['ch_sysef']]))
        return np.array([row['h_cwst'], row['lift'], row['ct_tot_kw'], row['ch_sysef']])

    def _get_initial_state(self):
        row = dataset.iloc[0]
        return np.array([row['h_cwst'], row['lift'], row['ct_tot_kw'], row['ch_sysef']])

class LearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.alpha = 0.01
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_min = 0.001
        self.epsilon_decay =0.995
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(
            Dense(24, input_dim=self.state_size, activation='relu')
        )
        model.add(
            Dense(24, activation='relu')
        )
        model.add(
            Dense(self.action_size, activation='linear')
        )
        model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=self.alpha))

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = np.random.choice(len(self.memory), batch_size)
        for i in minibatch:
            state, action, reward, next_state, done = self.memory[i]
            print(state)
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


state_bins = [
    np.linspace(27, 30, 30), # CWST
    np.linspace(22, 26, 40), # Lift
    np.linspace(3, 25, 220), # CTKW
    np.linspace(0.4, 0.65, 250) # ch_sysef
]

env = BuildingEnv()

# Initialize Q-learning agent
state_size = env.observation_space.shape[0]
action_size = env.action_space.n # no.of discrete actions
agent = LearningAgent(state_size, action_size)
done = False
batch_size = 32
# Training Loop
num_episodes = 10
reward_history = []

for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        
        if done:
            agent.update_target_model()
            break
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    reward_history.append(total_reward)

    

plt.plot(reward_history)
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()
print('training done')