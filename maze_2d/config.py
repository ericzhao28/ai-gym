'''
Defining the environment related constants
'''

import numpy as np


'''
Environment related constants
'''
MAZE_SIZE = None
NUM_BUCKETS = None
NUM_ACTIONS = None
STATE_BOUNDS = None
POSSIBLE_POSITIONS = None

'''
Learning related constants
'''
DISCOUNT_FACTOR = 0.99
MIN_EXPLORE_RATE = 0.001
MIN_LEARNING_RATE = 0.2
DECAY_FACTOR = None

'''
Defining the simulation related constants
'''
NUM_EPISODES = 50000
STREAK_TO_END = 1000
MAX_T = None
SOLVED_T = None

'''
For visualization and debugging
'''
VERBOSE = False
DEBUG_MODE = 0
RENDER_MAZE = False
ENABLE_RECORDING = False


def set_environment(env):
    global MAZE_SIZE, NUM_BUCKETS, NUM_ACTIONS, STATE_BOUNDS, \
        DECAY_FACTOR, MAX_T, SOLVED_T, POSSIBLE_POSITIONS

    # Number of discrete states (bucket) per state dimension
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAZE_SIZE  # one bucket per grid
    POSSIBLE_POSITIONS = [np.array([x, y], dtype=np.int)
                          for x in range(MAZE_SIZE[0]) for y in range(MAZE_SIZE[1])]

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0

    MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100
    SOLVED_T = np.prod(MAZE_SIZE, dtype=int)

