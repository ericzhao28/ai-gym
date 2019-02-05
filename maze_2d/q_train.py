"""
Function for training q-learning agent in maze2d.
"""

from .utils import state_to_bucket
from . import config
from .stats import add_env_sample, add_score


def train(env, agent):
    """ Train agent. """
    num_streaks = 0

    for episode in range(config.NUM_EPISODES):
        obv = env.reset()
        agent.reset()

        state_0 = state_to_bucket(obv)
        total_reward = 0.0

        for t in range(config.MAX_T):
            action = agent.select_action(state_0, env)
            obv, reward, done, _ = env.step(action)
            add_env_sample()

            # Observe the result
            state = state_to_bucket(obv)
            total_reward += reward

            # Update the Q based on the result
            agent.update(reward, state, state_0, action)

            # Setting up for the next iteration
            state_0 = state

            if config.RENDER_MAZE:
                env.render()

            if done:
                if config.VERBOSE:
                    print("Episode %d finished after %f time steps "
                          "with total reward = %f (streak %d)."
                          % (episode, t, total_reward, num_streaks))
                if t <= config.SOLVED_T:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

            elif t >= config.MAX_T - 1:
                print("Episode %d timed out at %d with total reward = %f."
                      % (episode, t, total_reward))

        # Add score
        if episode % 3 == 0:
            add_score(env, agent)

        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > config.STREAK_TO_END:
            break
