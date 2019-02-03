"""
Utilities for 2d maze environment.
"""

from . import config


def state_to_bucket(state):
    """
    Convert state value to a bucket
    """
    bucket_indice = []
    for i, state_i in enumerate(state):
        if state_i <= config.STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state_i >= config.STATE_BOUNDS[i][1]:
            bucket_index = config.NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = config.STATE_BOUNDS[i][1] - config.STATE_BOUNDS[i][0]
            offset = (config.NUM_BUCKETS[i] - 1) * \
                config.STATE_BOUNDS[i][0] / bound_width
            scaling = (config.NUM_BUCKETS[i] - 1) / bound_width
            bucket_index = int(round(scaling * state_i - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)
