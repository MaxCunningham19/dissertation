from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

from agents.AbstractAgent import AbstractAgent
from .utils import softmax


def get_agent_name(agent: AbstractAgent | None) -> str:
    if agent is None:
        return "Agent"
    return agent.name()


def format_preferences(human_preferences: np.ndarray) -> str:
    st = ""
    for i, pref in enumerate(human_preferences):
        st += f"{pref * 100}%"
        if i < len(human_preferences) - 1:
            st += ", "
    return st


def text_color(color):
    r, g, b = to_rgb(color)
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    if brightness < 0.5:
        return "white"
    return "black"


def smooth(array, window_size=1):
    """Calculates a smoothed moving average of the array"""
    smoothed_rewards = np.convolve(array, np.ones(window_size) / window_size, mode="valid")
    return smoothed_rewards


def plot_over_time_multiple_subplots(
    n_policy,
    values_to_plot,
    label=None,
    color="red",
    colors=None,
    xlabel="",
    ylabel="",
    titles=None,
    save_path: str = None,
    plot: bool = False,
    fig_title: str = "",
):
    """Plots the values of each individual policy over multiple iterations in multiple subplots"""
    fig, axes = plt.subplots(n_policy, 1)
    for i, current_values in enumerate(values_to_plot):
        x_values = np.arange(len(current_values))
        if not colors is None and len(colors) == n_policy:
            color = colors[i]
        axes[i].plot(x_values, current_values, label=label, color=color)
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(ylabel)
        if not titles is None and len(titles) == n_policy:
            axes[i].set_title(titles[i], fontsize=10)
    plt.subplots_adjust(top=0.9)
    fig.suptitle(fig_title, fontsize=14)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=500)
    if plot:
        plt.show()


def plot_agent_objective_q_values(
    states: list[list],
    agent: AbstractAgent,
    n_action,
    n_policy,
    bar_width=0.2,
    save_path: str = None,
    plot: bool = False,
    title: str = "",
    should_plot: lambda state: bool = lambda state: True,
    objective_labels: list[str] | None = None,
    human_preference: np.ndarray | None = None,
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
                agent_info = agent.get_objective_info(state, human_preference=human_preference)
                for i, info in enumerate(agent_info):

                    offset = negative_offset + (i) * (bar_width_single)
                    bar = current_ax.bar(xs + offset, agent_info[i], width=bar_width_single, label=objective_labels[i], color=colors[i], alpha=0.7)
                    legend[objective_labels[i]] = bar

    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.13, wspace=0.01, hspace=0.01)
    plt.figlegend(legend.values(), legend.keys(), loc="lower center", ncol=n_policy, bbox_to_anchor=(0.5, 0.02), fontsize=8)
    fig.suptitle(
        f"Q-Values of all objectives for {get_agent_name(agent)}\nwith human preferences {format_preferences(human_preference)}", fontsize=10
    )
    if save_path is not None:
        plt.savefig(f"{save_path}/objective_q_values.png", dpi=500)
    if plot:
        plt.show()


def plot_agent_objective_q_values_seperated(
    states: list[list],
    agent: AbstractAgent,
    n_action,
    n_policy,
    bar_width=0.2,
    save_path: str = None,
    plot: bool = False,
    objective_labels: list[str] | None = None,
    should_plot: lambda state: bool = lambda state: True,
    human_preference: np.ndarray | None = None,
):
    """Plots a grid of bar charts of the values of each action for each state"""
    if len(states) <= 0 or n_policy <= 0 or n_action <= 0:
        return

    if objective_labels is None:
        objective_labels = [f"{i}" for i in range(n_policy)]

    colors = plt.cm.viridis(np.linspace(0, 1, n_policy))
    xs = np.arange(n_action)

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
                    agent_info = agent.get_objective_info(state, human_preference=human_preference)
                    info = agent_info[i]
                    bar = current_ax.bar(xs, info, width=bar_width, color=colors[i], alpha=0.7)
                    for rect in bar:
                        height = rect.get_height()
                        current_ax.text(rect.get_x() + rect.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom", fontsize=5)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.13, wspace=0.01, hspace=0.01)
        fig.suptitle(
            f"Q-Values of Objective {objective_labels[i]} for {get_agent_name(agent)}\nwith human preference {human_preference[i]*100}%", fontsize=10
        )
        if save_path is not None:
            plt.savefig(f"{save_path}/q_values_objective_{objective_labels[i]}.png", dpi=500)
        if plot:
            plt.show()


def plot_agent_q_values(
    states: list[list],
    agent: AbstractAgent,
    n_action,
    n_policy,
    bar_width=0.2,
    save_path: str = None,
    plot: bool = False,
    objective_labels: list[str] | None = None,
    should_plot: lambda state: bool = lambda state: True,
    human_preference: np.ndarray | None = None,
):
    """Plots a grid of bar charts showing the summed Q-values for each action in each state"""
    if len(states) <= 0 or n_policy <= 0 or n_action <= 0:
        return
    if objective_labels is None:
        objective_labels = [f"{i}" for i in range(n_policy)]
    xs = np.arange(n_action)
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
                action_values = agent.get_actions(state, human_preference=human_preference)
                bar = current_ax.bar(xs, action_values, width=bar_width)
                for rect in bar:
                    height = rect.get_height()
                    current_ax.text(rect.get_x() + rect.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom", fontsize=5)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.13, wspace=0.01, hspace=0.01)
    fig.suptitle(f"Q-Values for {get_agent_name(agent)}\nwith human preferences {format_preferences(human_preference)}", fontsize=10)
    if save_path is not None:
        plt.savefig(f"{save_path}/q_values.png", dpi=500)
    if plot:
        plt.show()
    plt.close()


def plot_agent_w_values(
    states: list[list],
    w_values: dict[tuple[int, int], dict[int, list[float]]],
    n_policy,
    save_path: str = None,
    plot: bool = False,
    should_plot: lambda state: bool = lambda state: True,
    objective_labels: list[str] | None = None,
    agent: AbstractAgent | None = None,
):
    """Plots a grid of line charts of the w_values for each state over time"""
    if len(states) <= 0 or n_policy <= 0:
        return

    if objective_labels is None:
        objective_labels = [f"objective {i}" for i in range(n_policy)]

    colors = plt.cm.viridis(np.linspace(0, 1, n_policy))
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

    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.13, wspace=0.2, hspace=0.3)
    fig.legend(lines, labels, loc="lower center", ncol=n_policy, bbox_to_anchor=(0.5, 0.02), fontsize=8)
    fig.suptitle(f"Objective W-Values Over Time for {get_agent_name(agent)}", fontsize=10)
    if save_path is not None:
        plt.savefig(f"{save_path}/w_values.png", dpi=500, bbox_inches="tight")
    if plot:
        plt.show()


def plot_agent_actions(
    states: list[list],
    agent: AbstractAgent,
    n_actions: int,
    n_policy: int,
    save_path: str = None,
    plot: bool = False,
    action_labels: list[str] | None = None,
    objective_labels: list[str] | None = None,
    should_plot: lambda state: bool = lambda state: True,
    human_preference: np.ndarray | None = None,
):
    """Plots a grid of where the selected action is plotted for each state"""
    if action_labels is None:
        action_labels = [f"{i}" for i in range(n_actions)]
    if objective_labels is None:
        objective_labels = [f"Objective {i}" for i in range(n_policy)]
    colors = plt.cm.viridis(np.linspace(0, 1, n_policy))

    fig, axes = plt.subplots(len(states), len(states[0]))
    if not hasattr(axes, "__len__"):
        axes = [axes]
    else:
        axes = axes.flatten()
    for y, row in enumerate(states):
        for x, state in enumerate(row):
            current_ax = axes[y * len(states[0]) + x]
            current_ax.set_xticks([])
            current_ax.set_yticks([])
            if x == 0:
                current_ax.set_ylabel(f"{y}")
            if y == len(states) - 1:
                current_ax.set_xlabel(f"{x}")
            if should_plot(state):
                objective_actions = agent.get_objective_info(state, human_preference=human_preference)
                action, info = agent.get_action(state, human_preference=human_preference)

                max_obj_idx = np.argmax([obj_acts[action] for obj_acts in objective_actions])
                current_ax.set_facecolor(colors[max_obj_idx])
                current_ax.text(0.5, 0.5, f"{action_labels[action]}", ha="center", va="center", color=text_color(colors[max_obj_idx]), fontsize=8)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.13, wspace=0.01, hspace=0.01)
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=colors[i], label=objective_labels[i]) for i in range(n_policy)]
    fig.legend(handles=legend_elements, labels=objective_labels, loc="lower center", ncol=n_policy, bbox_to_anchor=(0.5, 0.02), fontsize=8)
    fig.suptitle(f"Selected Actions per state for {get_agent_name(agent)}\nwith human preference {format_preferences(human_preference)}", fontsize=10)
    plt.savefig(f"{save_path}/actions.png", dpi=500)
    if plot:
        plt.show()

    plt.close()
