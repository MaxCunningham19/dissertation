import argparse
import pandas as pd
import numpy as np

from utils.plotting import plot_over_time_multiple_subplots, smooth

parser = argparse.ArgumentParser()
parser.add_argument("--objective_labels", type=str, nargs="*", default=None, help="objective labels")
parser.add_argument("--results_path", type=str, required=True, help="path to results")
parser.add_argument("--images_dir", type=str, default=None, help="the path to save the images, if one doesnt exist the results path will be used")
parser.add_argument("--plot", action="store_true", default=False, help="should plot the saved plots")
args = parser.parse_args()


def parse_loss_string(s):
    try:
        tuples = s.strip("[]").split("),")
        tuples = [t.strip().strip("()") for t in tuples]
        return np.array([tuple(float(x.strip()) if x.strip() != "nan" else np.nan for x in t.split(",")) for t in tuples])
    except:
        return np.array([(np.nan, np.nan), (np.nan, np.nan)])


images_dir = args.images_dir
if images_dir is None:
    images_dir = args.results_path.split("/")[:-1].join("/")

df = pd.read_csv(args.results_path)
loss_arrays = df["loss"].apply(parse_loss_string).values
loss = np.array([x for x in loss_arrays]).T


def parse_list_string(s):
    try:
        s = s.strip("[]").split(",")
        return np.array([float(x) for x in s])
    except:
        return np.array([np.nan, np.nan])


episode_rewards_arrays = df["episode_reward"].apply(parse_list_string).values
n_policy = len(episode_rewards_arrays[0])
episode_rewards = np.array([x for x in episode_rewards_arrays]).T

objective_labels = args.objective_labels


q_loss, w_loss = [], []
has_w_loss = False
for x in loss_arrays:
    if len(x) > 1:
        has_w_loss = True
        q_loss.append([x[i][0] for i in range(len(x))])
        w_loss.append([x[i][1] for i in range(len(x))])
    else:
        q_loss.append([x[0][i] for i in range(len(x[0]))])
        w_loss.append([float("nan") for _ in range(len(x[0]))])

q_loss = np.array(q_loss).T
w_loss = np.array(w_loss).T

plot_over_time_multiple_subplots(
    n_policy,
    q_loss,
    save_path=f"{images_dir}/q_loss.png",
    plot=args.plot,
    xlabel="episodes",
    ylabel="error",
    titles=objective_labels,
    fig_title="Q-Values Loss",
)
if has_w_loss:
    plot_over_time_multiple_subplots(
        n_policy,
        w_loss,
        save_path=f"{images_dir}/w_loss.png",
        plot=args.plot,
        xlabel="episodes",
        ylabel="error",
        titles=objective_labels,
        fig_title="W-Values Loss",
    )


plot_over_time_multiple_subplots(
    n_policy,
    episode_rewards,
    save_path=f"{images_dir}/rewards.png",
    plot=args.plot,
    xlabel="episodes",
    ylabel="total reward",
    titles=objective_labels,
    fig_title="Rewards",
)

window_size = 50
smoothed_rewards = []
for i in range(n_policy):
    smoothed_rewards.append(smooth(episode_rewards[i], window_size))
plot_over_time_multiple_subplots(
    n_policy,
    smoothed_rewards,
    save_path=f"{images_dir}/smoother_rewards.png",
    plot=args.plot,
    xlabel="episodes",
    ylabel="total reward",
    titles=objective_labels,
    fig_title="Smooth Rewards",
)
