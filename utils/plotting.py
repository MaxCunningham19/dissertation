from matplotlib import pyplot as plt
import numpy as np

from .utils import softmax


def plot_agent_actions_2d(states: list[list], agent, n_action, bar_width=0.2, save_path: str = None, plot: bool = False, n_policy: int = 2):
    """Plots a grid of bar charts of the values of each action for each state"""
    if len(states) <= 0 or n_policy <= 0 or n_action <= 0:
        return

    colors = plt.cm.viridis(np.linspace(0, 1, n_policy))
    xs = np.arange(n_action)
    fig, axes = plt.subplots(len(states), len(states[0]))

    # Pre-calculate the bar width for each objective
    bar_width_single = bar_width / n_policy
    midpoint_policy = n_policy // 2
    even_poicy_offset = 1 if n_policy % 2 == 0 else 0

    for y, row in enumerate(states):
        for x, value in enumerate(row):
            ax = axes[x, y]
            ax.set_xticks([])
            ax.set_yticks([])
            idx = np.array([x, y])
            agent_info = agent.get_objective_info(idx)
            for i, info in enumerate(agent_info):
                advantages = info - np.mean(info)
                soft_array = softmax(advantages)
                # Plot bars in the current subplot
                negative_offset = -bar_width_single * ((n_policy) // 2)
                offset = negative_offset + (i) * (bar_width_single)
                print(i, negative_offset, offset, (bar_width_single))
                ax.bar(xs + offset, soft_array, width=bar_width_single, label=f"objective {i}", color=colors[i], alpha=0.7)

    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    if plot:
        plt.show()


def smooth(array, window_size=1):
    """Calculates a smoothed moving average of the array"""
    smoothed_rewards = np.convolve(array, np.ones(window_size) / window_size, mode="valid")
    return np.arange(len(smoothed_rewards)) + window_size // 2


def plot_over_time_multiple_subplots(
    n_policy, values_to_plot, label=None, color="red", colors=None, xlabel="", ylabel="", titles=None, save_path: str = None, plot: bool = False
):
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
    if save_path is not None:
        plt.savefig(save_path)
    if plot:
        plt.show()
