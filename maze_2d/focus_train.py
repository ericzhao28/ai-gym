"""
Function for training agent in maze2d.
"""

from .utils import state_to_bucket
from . import config
import time

from .stats import add_env_sample, add_expert_sample, add_score



def kl_divergence(env, rl_agent, il_agent, state):
    # from scipy.stats import entropy
    # return entropy(rl_agent.distribution(state), il_agent.distribution(state)) > 0.5
    return rl_agent.select_action(state, env) != il_agent.select_action(state, env)


def expanded_sample(env, rl_agent, il_agent, expert, state):
    """
    Extended sample by rolling out with expert if
    state has high KL-divergence.
    """

    # print("Extended sample!", time.time())
    state_0 = state
    il_agent.reset()
    expert.reset()

    for t in range(config.MAX_T):
        # Sample from the environment
        action = expert.select_action(state_0, env)
        add_expert_sample()
        obv, reward, done, _ = env.step(action)
        add_env_sample()

        # Observe the result
        state = state_to_bucket(obv)

        # Update the Q based on the result
        il_agent.aggregate(state_0, action)
        rl_agent.update(reward, state, state_0, action)

        # Setting up for the next iteration
        state_0 = state

        # Exit if necessary
        if done:
            break
        elif t >= config.MAX_T - 1:
            raise ValueError("Timed out at %d." % t)

    # Update il agent
    il_agent.update()


def train(env, rl_agent, il_agent, expert):
    """ Train agent. """
    num_streaks = 0

    for episode in range(config.NUM_EPISODES):
        obv = env.reset()

        state_0 = state_to_bucket(obv)
        total_reward = 0.0
        rl_agent.reset()

        for t in range(config.MAX_T):
            # Sample from the environment
            action = rl_agent.select_action(state_0, env)
            obv, reward, done, _ = env.step(action)
            add_env_sample()

            # Observe the result
            state = state_to_bucket(obv)
            total_reward += reward

            # Check if we need to roll-out the rl-agent
            if kl_divergence(env, rl_agent, il_agent, state_0):
                expanded_sample(env, rl_agent, il_agent, expert, state_0)
            rl_agent.update(reward, state, state_0, action)

            # Setting up for the next iteration
            state_0 = state

            if config.RENDER_MAZE:
                env.render()

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
            add_score(env, rl_agent)
