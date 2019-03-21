"""
SIREN model

"""

from .utils import state_to_bucket
from . import config
import time
import numpy as np
from .stats import add_env_sample, add_expert_sample, add_score


def expanded_sample(env, rl_agent, expert, state):
    """
    Extended sample by rolling out with expert if
    state has high KL-divergence.
    """

    # print("Extended sample!", time.time())
    state_0 = state

    for t in range(config.MAX_T):
        # Sample from the environment
        action = rl_agent.select_action(state_0, env)
        obv, reward, done, _ = env.step(action)
        add_env_sample()

        # Observe the result
        state = state_to_bucket(obv)
        rl_agent.update(reward, state, state_0, action)

        # Setting up for the next iteration
        state_0 = state

        # Exit if necessary
        if done:
            break
        elif t >= config.MAX_T - 1:
            raise ValueError("Timed out at %d." % t)


def sample_initial_state(env, discrepancy_critic):
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference
    initial_states = env.initial_states
    d_states = [discrepancy_critic.score_state(s) for s in initial_states]
    np.random.choice(initial_states, p=softmax(d_states))
    return d_states


def train(sim_env, real_env, rl_agent, discrepancy_critic, expert):
    """ Train agent. """
    num_streaks = 0

    for episode in range(config.NUM_EPISODES):
        state = sample_initial_state(real_env, discrepancy_critic)
        obv = real_env.reset(state)
        state_0 = state_to_bucket(obv)

        total_reward = 0.0
        rl_agent.reset()
        discrepancy_critic.reset()
        expert.reset()

        for t in range(config.MAX_T):
            # Sample from the environment
            action = rl_agent.select_action(state_0, real_env)
            obv, reward, done, _ = real_env.step(action)
            add_env_sample()

            # Observe the result
            state = state_to_bucket(obv)
            total_reward += reward
            rl_agent.update(reward, state, state_0, action)

            # Sample expert action
            expert_action = rl_agent.select_action(state_0, real_env, evaluation=True)
            add_expert_sample()

            # Check if we need to roll-out the rl-agent
            discrepancy = 0
            if action != expert_action:
                discrepancy = 1
            discrepancy_critic.update(discrepancy, state, state_0)

            # Setting up for the next iteration
            state_0 = state

            if config.RENDER_MAZE:
                real_env.render()
            if done:
                if config.VERBOSE:
                    print("Episode %d finished after %f time steps "
                          "with total reward = %f (streak %d)."
                          % (episode, t, total_reward, num_streaks))
                break
            elif t >= config.MAX_T - 1:
                raise ValueError("Episode %d timed out at %d with "
                                 "total reward = %f."
                                 % (episode, t, total_reward))

        if episode % 5 == 0:
            add_score(real_env, rl_agent)
