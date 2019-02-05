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
from .stats import plot_scores


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

real_dagger = ClassicDaggerAgent()
focused_learner = ClassicQAgent()
greedy_learner = ClassicQAgent()

# Set high streak req for experts
config.STREAK_TO_END = 2000

# Train sim expert
q_train(env, sim_expert)
sim_expert.save("sim_expert")

# Transfer learning to sim learn
greedy_learner.q_table = np.array(sim_expert.q_table, copy=True)
focused_learner.q_table = np.array(sim_expert.q_table, copy=True)

# Train real expert
real_env.unwrapped.real_life()
q_train(real_env, real_expert)
real_expert.save("real_expert")

# Train dagger
# config.NUM_EPISODES = 100
# dagger_train(real_env, real_expert, real_dagger, real_expert)
# evaluate(real_env, real_dagger)

# Check on the experts
evaluate(real_env, sim_expert)
evaluate(real_env, real_expert)

# Begin recording
config.RECORD_STATS = True
# config.ENABLE_RECORDING = True
# config.VERBOSE = True
# config.RENDER_MAZE = True
recording_folder = "/tmp/maze_q_learning"
if config.ENABLE_RECORDING:
    env = Monitor(env, recording_folder, force=True)

# Begin training the sim learn
config.NUM_EPISODES = 500
config.STREAK_TO_END = 100
config.MIN_EXPLORE_RATE = 0.05
config.DECAY_FACTOR = config.DECAY_FACTOR * 0.01
q_train(real_env, greedy_learner)

# Focus train
# config.NUM_EPISODES = 200
# config.DECAY_FACTOR = config.DECAY_FACTOR / 100.0
# config.MIN_EXPLORE_RATE = 0.01
# focus_train(real_env, focused_learner, real_dagger, real_expert)

plot_scores()
