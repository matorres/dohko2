import torch
import gym
import json
import numpy as np
import logging as log
from time import time
import matplotlib.pyplot as plt

from agents.dqn import DQNAgent
from agents.qlearning import QLearningAgent
from envs.aio import InstantOnDynamicEnv
from metrics import ssf, ctf

log.basicConfig(format='%(asctime)s | %(levelname)-8s | %(module)-5s:%(lineno)-4d | %(message)s', level=log.DEBUG)

# List of lines of the goal config
with open('configs/aio_configs.json', 'r') as file:
    data = json.load(file)
goal_config = data['config1']

# Declare environment and agent
env = InstantOnDynamicEnv(
    address='http://192.168.51.100',
    console_ip='192.168.51.100',
    goal=goal_config,
    headless=True)

agent = QLearningAgent(
    observation_space=env.observation_space,
    action_space=env.action_space)

# agent = DQNAgent(
#     observation_space=env.observation_space,
#     action_space=env.action_space,
#     epsilon_decay=0.998,
#     batch_size=8)

# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
# agent = DQNAgent(state_dim, action_dim)

# Train the agent
max_episodes = 20
max_steps_per_episode = 50
# max_training_time = 1800
max_training_time = 3*60*60
start_time = time()

log.info("Training RL agent ...")
timeout = False
scores = []
plt.ion()
fig, ax = plt.subplots()

for e in range(max_episodes):

    state = env.reset()
    agent.update_actions(env)

    for s in range(max_steps_per_episode):
        log.info(f'Step: {s+1}')
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        agent.update_actions(env)

        agent.learn(state, action, reward, next_state, done)
        state = next_state

        timeout = (time() - start_time) > max_training_time

        if done or timeout:
            break

        print()
    if timeout:
        break

    score = ctf(s+1, goal_config)*100
    scores.append(score)
    score_str = f'{score:.02f}%'
    print('\n')
    print('-'*120)
    log.info(
        f"Episode: {str(e+1).rjust(2)}/{max_episodes} - "
        f"Score: {score_str.rjust(2)} - "
        f"Epsilon: {agent.epsilon:.03f} - "
        f"Actions: {str(env.action_space.n).rjust(2)}")
    print('-'*120)
    print('\n')

    ax.clear()
    ax.plot(scores)
    ax.set_ylabel('Score')
    ax.set_xlabel('Episode')
    plt.pause(0.1)

    if (time() - start_time) > max_training_time:
        break

plt.ioff()
plt.show()

log.info(f"Training finished after: {time() - start_time:.03f}s")

# Test the Agent
state = env.reset()

steps = 0
start_time = time()
max_solving_time = 1800

done = False
timeout = False

log.info("Testing RL agent ...")
while not done and not timeout:
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    steps += 1
    timeout = (time() - start_time) > max_training_time

log.info(f"Testing finished after: {time() - start_time:.03f}s")
env.close()

# Calculate metrics
print('-'*120)
log.info(f"Metrics")
similarity = ssf(env, goal_config)*100
ssf_str = f'{similarity:.02f}%'
log.info(f"Solution similarity: {similarity}%")
convergence = ctf(steps, goal_config)*100
ctf_str = f'{convergence:.02f}'
log.info(f"Convergence time:    {ctf_str}")
print('-'*120)