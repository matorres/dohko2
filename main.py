import sys
import json
import logging as log
from time import time, sleep

from agents.dqn import DQNAgent
from agents.qlearning import QLearningAgent
from envs.aio import IODynamicEnv
from metrics import ssf, ctf

log_format = '%(asctime)s | %(levelname)-8s | %(module)-5s:%(lineno)-4d | %(message)s'
file_handler = log.FileHandler('experiment.logs')
file_handler.setFormatter(log.Formatter(log_format))
console_handler = log.StreamHandler()
console_handler.setFormatter(log.Formatter(log_format))
log.basicConfig(level=log.INFO, handlers=[file_handler, console_handler])

# python3 main.py config1 aio dqn

config_name = sys.argv[1]
env_name = sys.argv[2]
agent_name = sys.argv[3]


log.info('-'*120)
log.info(f'Experiment {int(time())} - Env: {env_name} - Agent: {agent_name} - Goal: {config_name}')
log.info('-'*120)

# List of lines of the goal config
with open('configs/aio_configs.json', 'r') as file:
    data = json.load(file)
goal_config = data[config_name]['lines']
min_steps = data[config_name]['min_steps']

state_dim = 2048

# Declare environment and agent
env = IODynamicEnv(
    address='http://192.168.51.100',
    console_ip='192.168.51.100',
    goal=goal_config,
    headless=True,
    hash_table_size=state_dim)

if agent_name == 'dqn':

    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr=0.001,
        gamma=0.95,
        epsilon_decay=0.9995,
        batch_size=64,
        state_dim=state_dim)
else:
    agent = QLearningAgent(
        observation_space=env.observation_space,
        action_space=env.action_space)

log.info('')

# Train the agent
max_episodes = 100
max_steps_per_episode = 50
# max_training_time = 1800
max_training_time = 5*60*60
start_time = time()

log.info("Training RL agent ...")
timeout = False
scores = []

for e in range(max_episodes):

    state = env.reset()
    agent.update_actions(env)

    for s in range(max_steps_per_episode):
        log.info(f'Step: {s+1}')
        action = agent.act(state)

        # for i, n in enumerate(env.actions_list):
        #     log.info(f'{i} - {n} - {env.actions_dict[n]}')
        # action = int(input('Action: '))

        next_state, reward, done, _ = env.step(action)

        agent.update_actions(env)

        agent.learn(state, action, reward, next_state, done)
        state = next_state

        timeout = (time() - start_time) > max_training_time

        log.info('')
        if done or timeout:
            break
    if timeout:
        break

    episode_ssf = ssf(env, goal_config)
    episode_ctf = ctf(s+1, min_steps)
    scores.append([episode_ssf, episode_ctf])
    ssf_str = f'{episode_ssf:.03f}'
    ctf_str = f'{episode_ctf:.03f}'
    log.info('-'*120)
    log.info(
        f"Episode: {str(e+1).rjust(2)}/{max_episodes} - "
        f"Similarity: {ssf_str.rjust(2)} - "
        f"Convergence: {ctf_str.rjust(2)} - "
        f"Epsilon: {agent.epsilon:.03f} - "
        f"Actions: {str(env.action_space.n).rjust(2)} - "
        f"Time: {time() - start_time:.03f}s"
    )

    log.info('-'*120)
    log.info('')

    if (time() - start_time) > max_training_time:
        log.warning('Training time is over ...')
        break

training_time_sec = time() - start_time
training_time_hours = training_time_sec/60/60
log.info(f"Training finished after {len(scores)} episodes and {training_time_hours:.02f} hours ({training_time_sec:.03f} sec)")
log.info('-'*120)
for i, s in enumerate(scores):
    ssf_str = f'{float(s[0]):.03f}'
    ctf_str = f'{float(s[1]):.03f}'
    log.info(f'Episode: {str(i):>2s} - Simularity: {ssf_str:>5s} - Convergence: {ctf_str:>5s}')
log.info('-'*120)
log.info('')

# Test the Agent
state = env.reset()

steps = 0
start_time = time()
max_solving_time = 300
max_solving_steps = max_steps_per_episode

done = False
timeout = False

agent.epsilon = 0

log.info("Testing RL agent ...")
for s in range(max_solving_steps):
    log.info(f'Step: {steps+1}')
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    steps += 1
    timeout = (time() - start_time) > max_solving_time
    log.info('')
    if done or timeout:
        break

log.info(f"Testing finished after: {time() - start_time:.03f}s")
env.close()

# Calculate metrics
log.info('-'*120)
log.info("Metrics")
log.info('-'*120)
similarity = ssf(env, goal_config)
convergence = ctf(steps, min_steps)
ssf_str = f'{similarity:.02f}'
ctf_str = f'{convergence:.02f}'
log.info(f"Solution similarity: {similarity}")
log.info(f"Convergence time:    {ctf_str}")
log.info('-'*120)