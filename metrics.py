import numpy as np
import logging as log

log.basicConfig(format='%(asctime)s,%(msecs)03.0f | %(levelname)-8s | %(module)-5s:%(lineno)-4d | %(message)s', level=log.INFO)

# Calculate the Solution Similarity Factor
def ssf(env, goal_config):
    config = env._console_get_config()

    # Get the number in both lists
    hits = len(list(set(config).intersection(goal_config)))
    return hits / len(goal_config)

# Calculate the Convergence Time Factor
def ctf(steps, goal_config):

    # Number of steps required
    littleN = steps
    bigN = len(goal_config)

    return 1 / ((0.5 * littleN / bigN) + 0.5)