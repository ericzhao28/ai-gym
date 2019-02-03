import gym
import gym_maze
from gym.wrappers import Monitor
from .train import train
from . import config
from .agent import ClassicQAgent
import numpy as np


# Set environment
env = gym.make("maze-random-10x10-plus-v0", num_portals=0)
config.set_environment(env)

# Instantiate agent
sim_agent = ClassicQAgent()
real_agent = ClassicQAgent()

# Begin recording
recording_folder = "/tmp/maze_q_learning"
if config.ENABLE_RECORDING:
    env = Monitor(env, recording_folder, force=True)

# Begin training the initial agent
train(env, sim_agent)

# Break walls in the environment
env.unwrapped.real_life()

# Begin training the real life agent
real_agent.q_table = np.array(sim_agent.q_table, copy=True)
train(env, real_agent)
