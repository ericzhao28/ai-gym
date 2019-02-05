import gym
import gym_maze
from gym.wrappers import Monitor
from .q_train import train as q_train
from .dagger_train import train as dagger_train
from .focus_train import train as focus_train
from .evaluate import evaluate
from . import config
from .agent import ClassicQAgent, ClassicDaggerAgent
import numpy as np


# Set environment
env = gym.make("maze-random-10x10-plus-v0", num_portals=0)
env.seed(0)
real_env = gym.make("maze-random-10x10-plus-v0", num_portals=0)
real_env.seed(0)
real_env.unwrapped.maze_view.maze.maze_cells = env.unwrapped.maze_view.maze.maze_cells.copy()
config.set_environment(env)

# Instantiate agent
sim_expert = ClassicQAgent()
real_expert = ClassicQAgent()

sim_learn = ClassicQAgent()
real_learn = ClassicQAgent()
real_dagger = ClassicDaggerAgent()

# Begin recording
recording_folder = "/tmp/maze_q_learning"
if config.ENABLE_RECORDING:
    env = Monitor(env, recording_folder, force=True)

# Set high streak req for experts
config.STREAK_TO_END = 2000

# Train sim expert
q_train(env, sim_expert)
evaluate(env, sim_expert)
sim_expert.save("sim_expert")

# Train real expert
real_env.unwrapped.real_life()
q_train(real_env, real_expert)
real_expert.save("real_expert")

# Train dagger
config.VERBOSE = True
# config.STREAK_TO_END = 100
# dagger_train(real_env, real_expert, real_dagger, real_expert)
# config.STREAK_TO_END = 100
# dagger_train(real_env, real_dagger, real_dagger, real_expert)
# evaluate(real_env, real_dagger)

focus_train(real_env, sim_expert, real_dagger, real_expert)

# Begin training the sim learn
sim_learn.q_table = np.array(sim_expert.q_table, copy=True)
