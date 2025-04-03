from matplotlib import pyplot as plt
import numpy as np

from agents.AbstractAgent import AbstractAgent

from .utils import softmax


def plot_agent_actions_2d(
    states: list[list],
    agent,
    n_action,
    n_policy,
    bar_width=0.2,
    save_path: str = None,
    plot: bool = False,
    title: str = "",
    should_plot: lambda state: bool = lambda state: True,
    objective_labels: list[str] | None = None,
):
    """Plots a grid of bar charts of the values of each action for each state"""
    if len(states) <= 0 or n_policy <= 0 or n_action <= 0:
        return

    colors = plt.cm.viridis(np.linspace(0, 1, n_policy))
    xs = np.arange(n_action)

    # Handle single subplot case
    fig, axes = plt.subplots(len(states), len(states[0]))
    if not hasattr(axes, "__len__"):
        axes = [axes]
    else:
        axes = axes.flatten()

    bar_width_single = bar_width / n_policy
    if bar_width_single < 0.1:
        bar_width_single = 0.1

    negative_offset = -bar_width_single * ((n_policy) // 2)
    legend = {}
    legend_set = False

    if objective_labels is None:
        objective_labels = [f"objective {i+1}" for i in range(n_policy)]

    for y, row in enumerate(states):
        for x, state in enumerate(row):
            # Get the correct axis based on whether we have a single subplot or multiple
            current_ax = axes[y * len(states[0]) + x]

            current_ax.set_xticks([])
            current_ax.set_yticks([])

            if x == 0:
                current_ax.set_ylabel(f"{y}")
            if y == len(states) - 1:
                current_ax.set_xlabel(f"{x}")

            idx = np.array([x, y])
            if should_plot(state):
                agent_info = agent.get_objective_info(state)
                for i, info in enumerate(agent_info):
                    advantages = info - np.mean(info)
                    soft_array = softmax(advantages)

                    offset = negative_offset + (i) * (bar_width_single)
                    bar = current_ax.bar(xs + offset, soft_array, width=bar_width_single, label=objective_labels[i], color=colors[i], alpha=0.7)
                    legend[objective_labels[i]] = bar

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.13, wspace=0.01, hspace=0.01)
    plt.figlegend(legend.values(), legend.keys(), loc="lower center", ncol=n_policy, bbox_to_anchor=(0.5, 0.02), fontsize=8)
    plt.title(f"{title}")
    if save_path is not None:
        plt.savefig(save_path, dpi=500)
    if plot:
        plt.show()


def smooth(array, window_size=1):
    """Calculates a smoothed moving average of the array"""
    smoothed_rewards = np.convolve(array, np.ones(window_size) / window_size, mode="valid")
    return smoothed_rewards


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
        plt.savefig(save_path, dpi=500)
    if plot:
        plt.show()


def plot_agent_actions_2d_seperated(
    states: list[list],
    agent: AbstractAgent,
    n_action,
    n_policy,
    bar_width=0.2,
    save_path: str = None,
    plot: bool = False,
    objective_labels: list[str] | None = None,
    should_plot: lambda state: bool = lambda state: True,
):
    """Plots a grid of bar charts of the values of each action for each state"""
    if len(states) <= 0 or n_policy <= 0 or n_action <= 0:
        return

    if objective_labels is None:
        objective_labels = [f"{i}" for i in range(n_policy)]

    colors = plt.cm.viridis(np.linspace(0, 1, n_policy + 1))
    xs = np.arange(n_action)

    # Handle single subplot case

    for i in range(n_policy):
        fig, axes = plt.subplots(len(states), len(states[0]))
        if not hasattr(axes, "__len__"):
            axes = [axes]
        else:
            axes = axes.flatten()
        for y, row in enumerate(states):
            for x, state in enumerate(row):
                # Get the correct axis based on whether we have a single subplot or multiple
                current_ax = axes[y * len(states[0]) + x]

                current_ax.set_xticks([])
                current_ax.set_yticks([])

                if x == 0:
                    current_ax.set_ylabel(f"{y}")
                if y == len(states) - 1:
                    current_ax.set_xlabel(f"{x}")

                idx = np.array([x, y])
                if should_plot(state):
                    agent_info = agent.get_objective_info(state)
                    info = agent_info[i]
                    bar = current_ax.bar(xs, info, width=bar_width, color=colors[i], alpha=0.7)
                    for rect in bar:
                        height = rect.get_height()
                        current_ax.text(rect.get_x() + rect.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom", fontsize=5)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.13, wspace=0.01, hspace=0.01)
        if save_path is not None:
            plt.savefig(f"{save_path}/objective_{objective_labels[i]}_q_values.png", dpi=500)
        if plot:
            plt.show()

    fig, axes = plt.subplots(len(states), len(states[0]))
    if not hasattr(axes, "__len__"):
        axes = [axes]
    else:
        axes = axes.flatten()
    for y, row in enumerate(states):
        for x, state in enumerate(row):
            # Get the correct axis based on whether we have a single subplot or multiple
            current_ax = axes[y * len(states[0]) + x]
            current_ax.set_xticks([])
            current_ax.set_yticks([])
            if x == 0:
                current_ax.set_ylabel(f"{y}")
            if y == len(states) - 1:
                current_ax.set_xlabel(f"{x}")
            if should_plot(state):
                action_values = agent.get_actions(state)
                bar = current_ax.bar(xs, action_values, width=bar_width, color=colors[0], alpha=0.7)
                for rect in bar:
                    height = rect.get_height()
                    current_ax.text(rect.get_x() + rect.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom", fontsize=5)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.13, wspace=0.01, hspace=0.01)
    if save_path is not None:
        plt.savefig(f"{save_path}/summed_q_values.png", dpi=500)
    if plot:
        plt.show()


def plot_agent_w_values(
    states: list[list],
    w_values: dict[tuple[int, int], dict[int, list[float]]],
    n_policy,
    save_path: str = None,
    plot: bool = False,
    should_plot: lambda state: bool = lambda state: True,
    objective_labels: list[str] | None = None,
):
    """Plots a grid of line charts of the w_values for each state over time"""
    if len(states) <= 0 or n_policy <= 0:
        return

    if objective_labels is None:
        objective_labels = [f"objective {i}" for i in range(n_policy)]

    colors = plt.cm.viridis(np.linspace(0, 1, n_policy + 1))
    lines = []
    labels = []

    fig, axes = plt.subplots(len(states), len(states[0]))
    if not hasattr(axes, "__len__"):
        axes = [axes]
    else:
        axes = axes.flatten()
    for y, row in enumerate(states):
        for x, state in enumerate(row):
            # Get the correct axis based on whether we have a single subplot or multiple
            current_ax = axes[y * len(states[0]) + x]
            current_ax.set_xticks([])
            current_ax.set_yticks([])
            if x == 0:
                current_ax.set_ylabel(f"{y}")
            if y == len(states) - 1:
                current_ax.set_xlabel(f"{x}")
            idx = np.array([x, y])
            w_values_dict = w_values.get((y, x), {})
            if should_plot(state):
                for i in range(n_policy):
                    if i in w_values_dict and len(w_values_dict[i]) > 0:
                        w_valuez = w_values_dict[i]
                        xs = np.arange(len(w_valuez))
                        line = current_ax.plot(xs, w_valuez, color=colors[i], alpha=0.7, label=objective_labels[i])
                        if i >= len(lines):  # Only add to legend once per policy
                            lines.append(line[0])
                            labels.append(objective_labels[i])

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.13, wspace=0.2, hspace=0.3)
    fig.legend(lines, labels, loc="lower center", ncol=n_policy, bbox_to_anchor=(0.5, 0.02), fontsize=8)
    if save_path is not None:
        plt.savefig(f"{save_path}/w_values.png", dpi=500, bbox_inches="tight")
    if plot:
        plt.show()
