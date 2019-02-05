"""
Evaluate agents
"""

from .utils import state_to_bucket
from . import config


def evaluate(env, agent):
    """ Evaluate agent. """
    timestep_count = 0
    all_rewards = 0.0

    for position in config.POSSIBLE_POSITIONS:
        obv = env.reset()
        obv = env.unwrapped.reset(position)
        agent.reset()

        state_0 = state_to_bucket(obv)
        total_reward = 0.0

        for t in range(config.MAX_T):
            timestep_count += 1
            action = agent.select_action(state_0, env, evaluation=True)
            obv, reward, done, _ = env.step(action)

            # Observe the result
            state = state_to_bucket(obv)
            total_reward += float(reward)

            # Setting up for the next iteration
            state_0 = state

            if config.RENDER_MAZE:
                env.render()

            if done:
                all_rewards += total_reward
                break
            elif t >= config.MAX_T - 1:
                # print("Episode timed out at %d with total reward = %f."
                #       % (t, total_reward))
                all_rewards += total_reward

    print("Total rewards: %f, total timesteps: %d." % (all_rewards, timestep_count))
    return all_rewards
