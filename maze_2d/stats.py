import matplotlib.pyplot as plt
from .evaluate import evaluate
from . import config


total_env_samples = 0
total_expert_samples = 0

evaluation_scores = []

def add_expert_sample():
    global total_expert_samples
    if not config.RECORD_STATS:
        return
    total_expert_samples += 1

def add_env_sample():
    global total_env_samples
    if not config.RECORD_STATS:
        return
    total_env_samples += 1

def add_score(env, agent):
    global total_env_samples, total_expert_samples, evaluation_scores
    if not config.RECORD_STATS:
        return
    score = evaluate(env, agent)
    evaluation_scores.append((total_env_samples, total_expert_samples, score))


def plot_scores():
    global evaluation_scores
    plt.scatter([x[0] for x in evaluation_scores], [x[2] for x in evaluation_scores])
    # plt.ylim(95.0, 100.0)
    plt.show()

