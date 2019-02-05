from . import config
import random
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


class Agent():
    def __init__(self):
        self.episode_count = 0

    def reset(self):
        self.episode_count += 1

    def select_action(self, state, env, evaluation=False):
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

    def select_action(self, state, env, evaluation=False):
        if (random.random() < self.explore_rate) and (not evaluation):
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
        self.q_table.dump(name + ".npy")

    def load(self, name):
        self.q_table = np.load(name + ".npy")


class NewQAgent(Agent):
    def select_action(self, state, env, evaluation=False, explore_rate=None):
        if evaluation:
            explore_rate = 0
        elif explore_rate is None:
            explore_rate = self.explore_rate
        else:
            explore_rate = max(explore_rate, self.explore_rate)

        if random.random() < explore_rate:
            return env.action_space.sample()
        else:
            return int(np.argmax(self.q_table[state]))

    def update(self, reward, state, old_state, old_action, learning_rate=None):
        best_q = np.amax(self.q_table[state])

        if learning_rate is None:
            learning_rate = self.learning_rate
        else:
            learning_rate = max(self.learning_rate, learning_rate)

        self.q_table[old_state + (old_action,)] += learning_rate * \
            (reward + config.DISCOUNT_FACTOR * (best_q) - self.q_table[old_state + (old_action,)])


class ClassicDaggerAgent(Agent):
    HIDDEN_DIM = 16
    N_EPOCHS = 4

    def __init__(self):
        super().__init__()
        self.state_data = []
        self.action_data = []

        self.model = Sequential()
        self.model.add(Dense(self.HIDDEN_DIM, input_dim=(config.MAZE_SIZE[0] * config.MAZE_SIZE[1]), activation='relu'))
        self.model.add(Dense(config.NUM_ACTIONS, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.has_updated = False

    def select_action(self, state, env, force=True, evaluation=False):
        if not self.has_updated and not force:
            raise ValueError("No data added yet.")
        action = self.model.predict(np.array([self.featurize(state)]))[0]
        return int(np.argmax(action))

    def featurize(self, state):
        assert(state[0] < config.MAZE_SIZE[0])
        assert(state[1] < config.MAZE_SIZE[1])

        index = state[0] * config.MAZE_SIZE[1] + state[1]
        feature_vec = np.zeros(config.MAZE_SIZE[0] * config.MAZE_SIZE[1])
        np.put(feature_vec, index, 1)
        return feature_vec

    def aggregate(self, state, action):
        self.state_data.append(self.featurize(state))

        assert(config.NUM_ACTIONS == 4)
        action_vec = np.zeros(config.NUM_ACTIONS)
        np.put(action_vec, action, 1)
        self.action_data.append(action_vec)

    def update(self):
        self.has_updated = True
        self.model.fit(np.array(self.state_data), np.array(self.action_data),
                       nb_epoch=self.N_EPOCHS, shuffle=True, verbose=0)
