import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import itertools
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument("--objective_labels", type=str, nargs="*", default=None, help="objective labels")
parser.add_argument("--folder_path", type=str, required=True, help="path to the folder")
parser.add_argument("--pareto_front", type=str, nargs="*", help="The get_pareto_front")
args = parser.parse_args()

df = pd.read_csv(f"{args.folder_path}/episode_rewards.csv")

reward_cols = [col for col in df.columns if "Episode Reward Objective" in col]
preference_cols = [col for col in df.columns if "Preference" in col]
n_objectives = len(reward_cols)
agents = df["Agent"].unique()
df = df.drop(preference_cols, axis=1)
objective_labels = args.objective_labels
if objective_labels is None or len(objective_labels) != n_objectives:
    objective_labels = [f"Objective {i}" for i in range(n_objectives)]


def dominates(a, b):
    return np.all(a >= b) and np.any(a > b)


def get_pareto_front(reward_array):
    pareto = []
    for i, a in enumerate(reward_array):
        is_dominated = False
        for j, b in enumerate(reward_array):
            if i != j and dominates(b, a):
                is_dominated = True
                break
        if not is_dominated:
            pareto.append(tuple(a))
    return pareto


def tuple_array(reward_array):
    f = []
    for a in reward_array:
        f.append(tuple(a))
    return f


def parse_pareto_front(pf: list[str]):
    if pf is None:
        return []
    ppf = []
    for s in pf:
        print(s)
        s = s.strip("()").split(",")
        print(s)
        ppf.append(list(map(lambda x: float(x.strip()), s)))
    return ppf


filtered_rows = []

pareto_front = parse_pareto_front(args.pareto_front)
agent_rewards = df[reward_cols].drop_duplicates().values
agent_rewards = np.round(agent_rewards, 1)
pareto_solutions = tuple_array(agent_rewards)
pareto_solutions = set(pareto_solutions)

for agent in agents:
    agent_data = df[df["Agent"] == agent]

    for _, row in agent_data.iterrows():
        row_values = row.values.copy()

        reward_indexes = [df.columns.get_loc(col) for col in reward_cols]

        for idx in reward_indexes:
            if isinstance(row_values[idx], (int, float)):
                row_values[idx] = round(row_values[idx], 1)

        rounded_rewards = tuple(row_values[i] for i in reward_indexes)
        if rounded_rewards in pareto_solutions:
            filtered_rows.append(row_values)


combined_df = pd.DataFrame(filtered_rows, columns=df.columns)

combined_df.to_csv(f"{args.folder_path}/pareto_solutions.csv", index=False)

marker_styles = [
    "o",  # circle
    "s",  # square
    "D",  # diamond
    "^",  # triangle up
    "v",  # triangle down
    "<",  # triangle left
    ">",  # triangle right
    "P",  # plus (filled)
    "X",  # x (filled)
    "*",  # star
    "d",  # thin diamond
    "h",  # hexagon 1
    "H",  # hexagon 2
    "+",  # plus
    "x",  # x
    "|",  # vertical line
    "_",  # horizontal line
    ".",  # point
    ",",  # pixel
    "1",  # tri-down
    "2",  # tri-up
    "3",  # tri-left
    "4",  # tri-right
]


marker_styles = itertools.cycle(marker_styles)
import numpy as np


df_values = [tuple([round(v, 1) for v in row]) for row in combined_df[reward_cols].drop_duplicates().values]
pareto_values = [tuple(row) for row in pareto_front]


unique_combinations = list(set(df_values + pareto_values))

combination_marker_map = {tuple(comb): next(marker_styles) for comb in unique_combinations}


colors = plt.cm.tab20(np.linspace(0, 1, len(agents)))
agent_color_map = dict(zip(agents, colors))


fig = plt.figure(figsize=(15, 10))
if n_objectives == 3:
    ax = fig.add_subplot(111, projection="3d")
else:
    ax = fig.add_subplot(111)
print("unbc", unique_combinations)
for comb in unique_combinations:
    marker = combination_marker_map[comb]
    c = ["0.7"]
    if comb in pareto_values:
        c = ["k"] if comb in df_values else ["0.35"]
    if n_objectives == 3:
        ax.scatter(comb[0], comb[1], comb[2], c=c, marker=marker, s=200, label=None)
    else:
        ax.scatter(comb[0], comb[1], c=c, marker=marker, s=100, label=None)


for i, label in enumerate(objective_labels):
    if n_objectives == 3:
        ax.set_xlabel(objective_labels[0], fontsize=10)
        ax.set_ylabel(objective_labels[1], fontsize=10)
        ax.set_zlabel(objective_labels[2], fontsize=10)
    else:
        ax.set_xlabel(objective_labels[0], fontsize=10)
        ax.set_ylabel(objective_labels[1], fontsize=10)

legend_handles = []
legend_labels = []

for agent in agents:
    agent_data = combined_df[combined_df["Agent"] == agent]
    combinations = agent_data[reward_cols].drop_duplicates().values

    agent_color = agent_color_map[agent]
    agent_markers = []

    for comb in combinations:
        marker = combination_marker_map[tuple(comb)]
        handle = Line2D([0], [0], marker=marker, color="w", markerfacecolor=agent_color, markeredgecolor="black", linestyle="None", markersize=8)
        agent_markers.append(handle)

    legend_handles.extend(agent_markers)
    legend_labels.extend([""] * (len(agent_markers) - 1) + [agent])  # Label only last

for i in range(len(legend_labels)):
    if len(legend_labels[i]) > 25:
        tmp = legend_labels[i].split(" ")
        tmp[len(tmp) // 2] = "\n" + tmp[len(tmp) // 2]
        legend_labels[i] = " ".join(tmp)

plt.legend(
    handles=legend_handles,
    labels=legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=min(4, len(agents)),
    fontsize=10,
    handletextpad=1.0,
)
if n_objectives == 3:
    ax.view_init(elev=70, azim=45)
plt.title("Discovered Pareto Solutions")
plt.tight_layout()
plt.savefig(f"{args.folder_path}/reward_combinations_plot.png", bbox_inches="tight")
