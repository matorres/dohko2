import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

import logging as log
log.basicConfig(format='%(asctime)s | %(levelname)-8s | %(module)-5s:%(lineno)-4d | %(message)s', level=log.INFO)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, middle_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, middle_dim)
        self.fc2 = nn.Linear(middle_dim, middle_dim)
        self.fc2p1 = nn.Linear(middle_dim, middle_dim)
        self.fc2p2 = nn.Linear(middle_dim, middle_dim)
        self.fc3 = nn.Linear(middle_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2p1(x))
        x = torch.relu(self.fc2p2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(
            self,
            observation_space,
            action_space,
            # lr=0.001,
            lr=0.1,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            min_epsilon=0.01,
            batch_size=64,
            buffer_size=10000):

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)

        self.action_space = action_space
        self.observation_space = observation_space

        self.state_dim = 384
        self.action_dim = 1
        self.middle_dim = batch_size

        self.q_network = QNetwork(self.state_dim, self.action_dim, self.middle_dim)
        self.target_network = QNetwork(self.state_dim, self.action_dim, self.middle_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def update_actions(self, env):

        # FIXME: Add new actions always
        update_prob = 1

        # Identify potential new actions
        if env.action_space and np.random.rand() > update_prob:
        # if env.action_space and self.epsilon < update_prob:
            return

        log.debug(f"Identifing new actions ...")
        new_actions = env.find_actions()

        if new_actions <= 0:
            # log.info('Update actions end')
            return

        self.action_space = env.action_space
        new_action_dim = env.action_space.n

        # Update the Q-network to handle the new action dimension
        self.q_network.fc3 = nn.Linear(self.middle_dim, new_action_dim)
        self.target_network.fc3 = nn.Linear(self.middle_dim, new_action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.action_dim = new_action_dim

        # log.info('Update actions end')

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state, done):
        # Add the experience to memory
        self.memory.append((state, action, reward, next_state, done))

        # Perform learning if the memory is sufficiently large
        if len(self.memory) < self.batch_size:
            log.info('Memory not big enough to learn')
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert states to a numpy array and then to a tensor
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Adjust the states before feeding into the network
        states = states.squeeze(1)  # Shape now: [4, 384]

        # Gather the Q-values for the actions
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)

        # log.info(f'{states.shape=}')
        # log.info(f'{actions.shape=}')

        # current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        current_q_values = self.q_network(states).gather(1, actions).squeeze()

        next_q_values = self.target_network(next_states).max(1)[0]

        # log.info(f"rewards.shape: {rewards.shape}")
        # log.info(f"next_q_values.shape: {next_q_values.shape}")
        # log.info(f"dones.shape: {dones.shape}")

        rewards = rewards.squeeze()  # If rewards has shape [batch_size, 15], adjust it to [batch_size]
        # log.info(f"rewards.shape: {rewards.shape}")

        next_q_values = next_q_values.max(dim=1)[0]
        # log.info(f"next_q_values.shape: {next_q_values.shape}")

        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        # log.debug('Learning end')

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)