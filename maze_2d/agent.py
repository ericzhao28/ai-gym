from . import config
import random
import math
import numpy as np


class Agent():
    def __init__(self):
        self.total_reward = 0
        self.episode_count = 0

    def reset(self):
        self.total_reward = 0
        self.episode_count += 1

    def select_action(self, state, env):
        raise NotImplementedError()

    def update(self, reward, state, old_state, old_action):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    @property
    def explore_rate(self):
        raise NotImplementedError()

    @property
    def learning_rate(self):
        raise NotImplementedError()


class ClassicQAgent(Agent):
    def __init__(self):
        super().__init__()
        self.q_table = np.zeros(config.NUM_BUCKETS + (config.NUM_ACTIONS,),
                                dtype=np.float32)

    def select_action(self, state, env):
        if random.random() < self.explore_rate:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(self.q_table[state]))
        return action

    def update(self, reward, state, old_state, old_action):
        best_q = np.amax(self.q_table[state])
        self.q_table[old_state + (old_action,)] += self.learning_rate * \
            (reward + config.DISCOUNT_FACTOR * (best_q) - self.q_table[old_state + (old_action,)])

    @property
    def explore_rate(self):
        return max(config.MIN_EXPLORE_RATE,
                   min(0.8, 1.0 - math.log10((self.episode_count + 1) / config.DECAY_FACTOR)))

    @property
    def learning_rate(self):
        return max(config.MIN_LEARNING_RATE,
                   min(0.8, 1.0 - math.log10((self.episode_count + 1) / config.DECAY_FACTOR)))

    def save(self, name):
        raise NotImplementedError()

    def load(self, name):
        self.q_table = np.load(name + ".npy")
