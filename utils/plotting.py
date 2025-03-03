from matplotlib import pyplot as plt
import numpy as np

from .utils import softmax


def plot_agent_actions_2d(states: list[list], agent, n_action, bar_width=0.2):
    """Plots a grid of bar charts of the values of each action for each state"""
    if len(states) <= 0:
        return
    xs = np.arange(n_action)
    fig, axes = plt.subplots(len(states), len(states[0]))
    for y, row in enumerate(states):
        for x, value in enumerate(row):
            ax = axes[x, y]
            ax.set_xticks([])
            ax.set_yticks([])
            # if value >= 0.0 or value >= 0:
            idx = np.array([x, y])
            agent_info = agent.get_agent_info(idx)
            advantages1 = agent_info[0]
            advantages2 = agent_info[1]
            advantages1 = advantages1.tolist()[0]
            advantages2 = advantages2.tolist()[0]
            soft_array1 = softmax(advantages1)
            soft_array2 = softmax(advantages2)
            array1 = advantages1 - np.mean(advantages1)
            array2 = advantages2 - np.mean(advantages2)
            # Plot bars in the current subplot
            ax.bar(xs - bar_width / 2, soft_array1, width=bar_width, label="treasure", color="blue", alpha=0.7)
            ax.bar(xs + bar_width / 2, soft_array2, width=bar_width, label="speed", color="green", alpha=0.7)

    plt.legend()
    plt.show()


def smooth(array, window_size=1):
    """Calculates a smoothed moving average of the array"""
    smoothed_rewards = np.convolve(array, np.ones(window_size) / window_size, mode="valid")
    return np.arange(len(smoothed_rewards)) + window_size // 2


def plot_over_time_multiple_subplots(n_policy, values_to_plot, label=None, color="red", colors=None, xlabel="", ylabel="", titles=None):
    """Plots the values of each individual policy over multiple iterations in multiple subplots"""
    _, axes = plt.subplots(n_policy, 1)
    for i, current_values in enumerate(values_to_plot):
        x_values = np.arange(len(current_values))
        if not colors is None and len(colors) == n_policy:
            color = colors[i]
        axes[i].plot(x_values, current_values, label=label, color=color)
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(ylabel)
        if not titles is None and len(titles) == n_policy:
            axes[i].set_title(titles[i])

    plt.tight_layout()
    plt.show()
