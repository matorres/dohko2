import numpy as np
import logging as log

log.basicConfig(format='%(asctime)s,%(msecs)03.0f | %(levelname)-8s | %(module)-5s:%(lineno)-4d | %(message)s', level=log.INFO)

class QLearningAgent:
    def __init__(self, observation_space, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Initialize Q-table with zeros
        self.q_table = {}
        self.action_space = action_space
        self.observation_space = observation_space

    def _get_q_values(self, state):
        # Convert state to tuple if it is an array
        # state_tuple = tuple(state.flatten()) if isinstance(state, np.ndarray) else (state,)
        state_tuple = tuple(state)

        # Initialize Q-values for new states
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = np.zeros(self.action_space.n)
        return self.q_table[state_tuple]

    def act(self, state):
        # state_tuple = tuple(state.flatten()) if isinstance(state, np.ndarray) else (state,)
        state_tuple = tuple(state)

        # Epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            action = self.action_space.sample()
        else:
            # Exploit: choose the best action from Q-table
            q_values = self._get_q_values(state_tuple)
            action = np.argmax(q_values)
        return action

    def learn(self, state, action, reward, next_state, done):
        # log.debug('Learning start')
        if action >= self.action_space.n:
            raise ValueError("Action out of bounds")

        # state_tuple = tuple(state.flatten()) if isinstance(state, np.ndarray) else (state,)
        state_tuple = tuple(state)
        # next_state_tuple = tuple(next_state.flatten()) if isinstance(next_state, np.ndarray) else (next_state,)
        next_state_tuple = tuple(next_state)
        q_values = self._get_q_values(state_tuple)

        # Update Q-value using the Q-learning formula
        if next_state_tuple in self.q_table:
            next_max = np.max(self.q_table[next_state_tuple])
        else:
            next_max = 0.0

        # Update Q-value for the current state-action pair
        target = reward + self.discount_factor * next_max * (1 - done)
        q_values[action] = q_values[action] + self.learning_rate * (target - q_values[action])

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        # log.debug('Learning end')

    def update_actions(self, env):

        # FIXME: Add new actions always
        update_prob = 1

        # Identify potential new actions
        if env.action_space and np.random.rand() > update_prob:
        # if env.action_space and self.epsilon < update_prob:
            return

        log.info(f"Identifing new actions ...")
        new_actions = env.find_actions()

        if new_actions <= 0:
            # log.info('Update actions end')
            return

        self.action_space = env.action_space

        # for state_tuple in self.q_table.keys():
        #     self.q_table[state_tuple] = np.append(self.q_table[state_tuple], [np.zeros(new_actions)])

        for state in self.q_table.keys():
            self.q_table[state] = np.append(self.q_table[state], [np.ones(new_actions)])

        # log.info('Update actions end')