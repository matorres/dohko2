import logging as log

log_format = '%(asctime)s | %(levelname)-8s | %(module)-5s:%(lineno)-4d | %(message)s'
file_handler = log.FileHandler('experiment.logs')
file_handler.setFormatter(log.Formatter(log_format))
console_handler = log.StreamHandler()
console_handler.setFormatter(log.Formatter(log_format))
log.basicConfig(level=log.INFO, handlers=[file_handler, console_handler])


# Calculate the Solution Similarity Factor
def ssf(env, goal_config):
    config = env._console_get_config()

    # Get the number in both lists
    hits = len(list(set(config).intersection(goal_config)))
    return hits / len(goal_config)


# Calculate the Convergence Time Factor
def ctf(steps, min_steps):

    # Number of steps required
    little_n = steps
    big_n = min_steps

    return 1 / ((0.5 * little_n / big_n) + 0.5)
