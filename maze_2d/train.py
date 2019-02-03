"""
Function for training agent in maze2d.
"""

import sys

from .utils import state_to_bucket
from . import config


def train(env, agent):
    """ Train agent. """
    env.render()

    num_streaks = 0
    timestep_count = 0

    for episode in range(config.NUM_EPISODES):
        obv = env.reset()
        agent.reset()

        state_0 = state_to_bucket(obv)
        total_reward = 0

        for t in range(config.MAX_T):
            timestep_count += 1
            action = agent.select_action(state_0, env)
            obv, reward, done, _ = env.step(action)

            # Observe the result
            state = state_to_bucket(obv)
            total_reward += reward

            # Update the Q based on the result
            agent.update(reward, state, state_0, action)

            # Setting up for the next iteration
            state_0 = state

            if config.DEBUG_MODE == 1:
                if done or t >= config.MAX_T - 1:
                    print("\nEpisode = %d" % episode)
                    print("t = %d" % t)
                    print("Total timestep = %d" % timestep_count)
                    print("Streaks: %d" % num_streaks)
                    print("Total reward: %f" % total_reward)
                    print("")

            if config.RENDER_MAZE:
                env.render()

            if env.unwrapped.is_game_over():
                sys.exit()

            if done:
                print("Episode %d finished after %f time steps "
                      "with total reward = %f (streak %d). Total "
                      "time steps are %d."
                      % (episode, t, total_reward, num_streaks, timestep_count))
                if t <= config.SOLVED_T:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

            elif t >= config.MAX_T - 1:
                print("Episode %d timed out at %d with total reward = %f."
                      % (episode, t, total_reward))

        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > config.STREAK_TO_END:
            break
