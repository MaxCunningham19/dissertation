import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, required=True, help="the environment to run")
parser.add_argument("--objective_labels", type=str, nargs="*", default=None, help="objective labels")
parser.add_argument("--folder_path", type=str, required=True, help="path to the folder")
args = parser.parse_args()

df = pd.read_csv(f"{args.folder_path}/episode_rewards.csv")

# Get number of objectives
reward_cols = [col for col in df.columns if "Episode Reward Objective" in col]
pref_cols = [col for col in df.columns if "Human Preference Objective" in col]
n_policy = len(reward_cols)

objective_labels = args.objective_labels
if objective_labels is None or len(objective_labels) != n_policy:
    objective_labels = [f"Objective {i}" for i in range(n_policy)]

# Create a human preference label for x-axis
df["Preference Label"] = df[pref_cols].apply(lambda row: "(" + ", ".join(map(str, row)) + ")", axis=1)

# Get unique agents
agents = df["Agent"].unique()
x_labels = df[df["Agent"] == agents[0]]["Preference Label"]
x = np.arange(len(x_labels))  # x locations

marker_styles = ["o", "s", "D", "^", "v", "<", ">", "X", "P", "*"]  # Extend if you have many agents
colors = plt.cm.viridis(np.linspace(0, 1, n_policy))
import matplotlib.pyplot as plt

ncols = 2
nrows = len(agents) // ncols + 1  # One row for each agent
# One column, since we are plotting one reward column per figure
fig, axs = plt.subplots(nrows, ncols, figsize=(15 * ncols, 10 * nrows), sharex=False, sharey=False)
if nrows == 1:
    axs = [axs]
axs = axs.flatten()
y_min = float("inf")
y_max = float("-inf")
legend = []
first = True
# Loop through each agent and create a subplot for each one
for j, agent in enumerate(agents):
    ax = axs[j]  # Select the appropriate subplot for each agent
    agent_data = df[df["Agent"] == agent]
    for i, reward_col in enumerate(reward_cols):
        print(agent, x, agent_data[reward_col])
        line = ax.plot(
            x,
            agent_data[reward_col],
            marker=marker_styles[j % len(marker_styles)],
            color=colors[i],
            alpha=0.8,  # Transparency
            linewidth=5,  # Thinner lines
            markersize=12,
        )
        if first:
            legend.append(line[0])
    first = False
    ax.set_title(f"Agent: {agent}")
    ax.set_ylabel("Rewards")
    # Set x-ticks and x-tick labels for every subplot
    ax.set_xlabel("Human Preferences")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    y_min = min(y_min, agent_data[reward_col].min())
    y_max = max(y_max, agent_data[reward_col].max())
    # Optional: Add gridlines for better readability
    ax.grid(True, axis="y")


for k in range(len(agents), len(axs)):
    axs[k].set_visible(False)
# Adjust the layout to prevent overlap of titles/labels
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.2)
# Create a legend below the plots
plt.figlegend(
    legend,  # Use the lines from the first subplot for the legend
    objective_labels,
    loc="lower right",
    ncol=len(agents) // 2 + 1,  # Number of columns based on the number of agents
    bbox_to_anchor=(0.5, -0.05),  # Position the legend below the plot
    fontsize=8,
)
# Save the figure for the current reward column
file_path = f"{args.folder_path}/reward_objectives.png"
plt.savefig(file_path)
print(f"Saved plot to {file_path}")
plt.close()


# Plot each agent in a separate figure
for j, agent in enumerate(agents):
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
