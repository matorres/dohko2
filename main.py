import torch
import gym
import numpy as np
import logging as log
from time import time

from agents.dqn import DQNAgent
from agents.qlearning import QLearningAgent
from envs.aio import InstantOnDynamicEnv
from metrics import ssf, ctf

log.basicConfig(format='%(asctime)s,%(msecs)03.0f | %(levelname)-8s | %(module)-5s:%(lineno)-4d | %(message)s', level=log.INFO)

# List of lines of the goal config
goal_config = [
    'hostname happy-switch'
]

# Declare environment and agent
env = InstantOnDynamicEnv(
    address='http://192.168.1.1',
    console_ip='192.168.1.1',
    goal=goal_config)

agent = QLearningAgent(
    observation_space=env.observation_space,
    action_space=env.action_space)

# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
# agent = DQNAgent(state_dim, action_dim)

# Train the agent
episodes = 20
start_time = time()
max_steps_per_episode = 50

log.info("Training RL agent ...")
for e in range(episodes):
    state = env.reset()

    agent.update_actions(env)

    for s in range(max_steps_per_episode):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        agent.update_actions(env)

        agent.learn(state, action, reward, next_state, done)
        state = next_state

        if done:
            log.info(
                f"Episode: {str(e+1).rjust(2)}/{episodes} - "
                f"Score: {str(s).rjust(2)} - "
                f"Epsilon: {agent.epsilon:.03f} - "
                f"Actions: {str(env.action_space.n).rjust(2)}")
            break
        print()

log.info(f"Training finished after: {time() - start_time:.03f}s")

# Test the Agent
steps = 0
start_time = time()
max_solving_time = 1800

state = env.reset()
done = False

log.info("Testing RL agent ...")
while (not done) and (time() - start_time < 1800):
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    steps += 1
log.info(f"Testing finished after: {time() - start_time:.03f}s")

env.close()


# Calculate metrics
# ...
similarity = ssf(env, goal_config)
convergence = ctf(env, goal_config)