import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

import logging as log

log_format = '%(asctime)s | %(levelname)-8s | %(module)-5s:%(lineno)-4d | %(message)s'
file_handler = log.FileHandler('experiment.logs')
file_handler.setFormatter(log.Formatter(log_format))
console_handler = log.StreamHandler()
console_handler.setFormatter(log.Formatter(log_format))
log.basicConfig(level=log.INFO, handlers=[file_handler, console_handler])


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_layers=3, max_action_dim=128, name=None):
        super(QNetwork, self).__init__()

        layer_sizes = np.linspace(state_dim, max_action_dim, num_layers + 1, dtype=int)
        log.info(f'Using the following structure for the {name} neural network: {list(layer_sizes)}')

        self.middle_layers = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(num_layers - 1)]
        )

        self.fc_out = nn.Linear(self.middle_layers[-1].out_features, action_dim)

    def forward(self, x):

        for fc in self.middle_layers:
            x = torch.relu(fc(x))

        x = self.fc_out(x)
        return x


class DQNAgent:
    def __init__(
            self,
            observation_space,
            action_space,
            lr=0.001,
            # lr=0.1,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            min_epsilon=0.01,
            batch_size=64,
            buffer_size=10000,
            state_dim=2048):

        log.info('Creating RL agent ...')

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.min_memory_size_to_learn = batch_size * 3

        self.action_space = action_space
        self.observation_space = observation_space

        self.state_dim = state_dim
        self.action_dim = 1
        self.num_layers = 2
        self.max_action_dim = 64

        self.temperature = 0.8

        self.q_network = QNetwork(
            self.state_dim, self.action_dim, num_layers=self.num_layers,
            max_action_dim=self.max_action_dim, name='main')
        self.target_network = QNetwork(
            self.state_dim, self.action_dim, num_layers=self.num_layers,
            max_action_dim=self.max_action_dim, name='target')
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def update_actions(self, env):

        # FIXME: Add new actions
        # update_prob = 1 - self.epsilon
        update_prob = 1

        # Identify potential new actions
        if env.action_space and np.random.rand() > update_prob:
            return

        log.debug("Identifing new actions ...")
        new_actions = env.find_actions()

        if new_actions <= 0:
            log.debug('Update actions end')
            return

        self.action_space = env.action_space
        new_action_dim = env.action_space.n

        # Update the Q-network to handle the new action dimension
        with torch.no_grad():
            self.q_network.fc_out = nn.Linear(self.q_network.middle_layers[-1].out_features, new_action_dim)
            self.target_network.fc_out = nn.Linear(self.target_network.middle_layers[-1].out_features, new_action_dim)
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.action_dim = new_action_dim

        log.debug('Update actions end')

    def act(self, state):
        # if np.random.rand() < self.epsilon:
        #     log.info('Action selected to explore environment ...')
        #     return random.randrange(self.action_dim)
        # else:
        #     with torch.no_grad():
        #         state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        #         q_value = self.q_network(state)
        #         log.info(f'Action selected to explote environment [{torch.max(q_value)}]!')
        #         return torch.argmax(q_value).item()

        # Select action using Softmax
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_network(state_tensor).squeeze()

            # Apply Softmax to the Q values
            action_probs = self.softmax(q_values)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs.numpy())

            log.info(f'Action selected using Softmax: {action}')
            return action

    def learn(self, state, action, reward, next_state, done):

        log.debug('Learning start')

        # Add the experience to memory
        self.memory.append((state, action, reward, next_state, done))

        # Perform learning if the memory is sufficiently large
        if len(self.memory) < self.min_memory_size_to_learn:
            util = len(self.memory) / (self.min_memory_size_to_learn) * 100
            log.info(f'Memory not big enough to learn [{util:.02f}%]')
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

        actions = actions.unsqueeze(1)
        current_q_values = self.q_network(states).gather(1, actions).squeeze()

        next_q_values = self.target_network(next_states).max(1)[0]
        rewards = rewards.squeeze()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.loss_fn(current_q_values, target_q_values)
        log.info(f'Loss: {loss}')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.decay_epsilon()

        log.debug('Learning end')

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def softmax(self, q_values):
        exp_q = torch.exp(q_values / self.temperature)
        softmax_probs = exp_q / torch.sum(exp_q)
        return softmax_probs