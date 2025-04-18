import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=False, help="the environment to run")
parser.add_argument("--objective_labels", type=str, nargs="*", default=None, help="objective labels")
parser.add_argument("--folder_path", type=str, required=True, help="path to the folder")
args = parser.parse_args()

df = pd.read_csv(f"{args.folder_path}/episode_rewards.csv")

reward_cols = [col for col in df.columns if "Episode Reward Objective" in col]
pref_cols = [col for col in df.columns if "Human Preference Objective" in col]
n_policy = len(reward_cols)

objective_labels = args.objective_labels
if objective_labels is None or len(objective_labels) != n_policy:
    objective_labels = [f"Objective {i}" for i in range(n_policy)]

df["Preference Label"] = df[pref_cols].apply(lambda row: "(" + ", ".join(map(str, row)) + ")", axis=1)
key_words = ["Linear", "Chebyshev", "Softmax", "L1"]
all_agents = df["Agent"].unique()

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

grouped_agents = defaultdict(list)
for agent in df["Agent"].unique():
    sp = agent.split("with")
    prefix = sp[0].strip()
    grouped_agents[prefix].append(agent)

for group_name, agents in grouped_agents.items():
    agents = sorted(agents)
    print(f"\nPlotting group: {group_name} ({len(agents)} agents)")

    x_labels = df[df["Agent"] == agents[0]]["Preference Label"]
    x = np.arange(len(x_labels))

    marker_styles = ["o", "s", "D", "^", "v", "<", ">", "X", "P", "*"]
    colors = plt.cm.viridis(np.linspace(0, 1, n_policy))

    ncols = 2 if len(agents) >= 2 else 1
    nrows = len(agents) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(16 * ncols, 12 * nrows), sharex=False, sharey=False)
    axs = np.array(axs).reshape(-1)  # Always a flat array

    print(axs, len(axs))
    print(f"Agents: {agents}, axs type: {type(axs)}, axs len: {len(axs) if isinstance(axs, (list, np.ndarray)) else 'not iterable'}")

    y_min = float("inf")
    y_max = float("-inf")
    legend = []
    first = True

    for j, agent in enumerate(agents):
        ax = axs[j]
        agent_data = df[df["Agent"] == agent]
        for i, reward_col in enumerate(reward_cols):
            line = ax.plot(
                x_labels, agent_data[reward_col], marker=marker_styles[i % len(marker_styles)], color=colors[i], alpha=0.8, linewidth=5, markersize=12
            )
            y_min = min(y_min, agent_data[reward_col].min())
            y_max = max(y_max, agent_data[reward_col].max())
            if first:
                legend.append(line[0])
        first = False

        ax.set_title(f"{agent}", fontsize=28)
        ax.tick_params(axis="y", labelsize=28)
        ax.tick_params(axis="x", labelsize=28)

        row_idx = j // ncols
        col_idx = j % ncols

        if row_idx != nrows - 1:
            ax.tick_params(labelbottom=False)
            ax.set_xlabel("")
        else:
            font_size = 28 if len(x_labels) <= 12 else 18
            step = 1 if len(x_labels) <= 12 else 2
            tick_labels = [label if i % step == 0 else "" for i, label in enumerate(x_labels)]
            ax.set_xlabel("Human Preferences", fontsize=28)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=font_size)
            ax.tick_params(axis="x", pad=10)  # Adds padding between ticks and labels

        if col_idx != 0:
            ax.tick_params(labelleft=False)
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Rewards", fontsize=28)

        ax.grid(True, axis="y")
        ax.grid(True, axis="x")

    margin = abs(y_max - y_min) * 0.05
    for k in range(len(agents)):
        axs[k].set_ylim(y_min - margin, y_max + margin)

    for k in range(len(agents), len(axs)):
        print(k)
        axs[k].set_visible(False)

    plt.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.12, wspace=0.1, hspace=0.1)

    plt.figlegend(legend, objective_labels, loc="lower center", ncol=len(objective_labels), fontsize=30)

    filename_safe_group = group_name.replace(" ", "_").lower()
    file_path = f"{args.folder_path}/reward_objectives_{filename_safe_group}.png"
    plt.savefig(file_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {file_path}")
    plt.close()


for j, agent in enumerate(all_agents):
    agent_data = df[df["Agent"] == agent]
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, reward_col in enumerate(reward_cols):
        ax.plot(
            x,
            agent_data[reward_col],
            label=f"{objective_labels[i]}",
            marker=marker_styles[i % len(marker_styles)],
            color=colors[i],
            alpha=0.8,
            linewidth=3,
            markersize=10,
        )

    ax.set_title(f"Rewards for Agent: {agent}")
    ax.set_ylabel("Rewards")
    ax.set_xlabel("Human Preferences")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.grid(True, axis="y")
    ax.legend(title="Objectives")

    file_path = f"{args.folder_path}/reward_objectives_{agent}.png"
    plt.tight_layout()
    plt.savefig(file_path)
    print(f"Saved separate plot for {agent} to {file_path}")
    plt.close()
