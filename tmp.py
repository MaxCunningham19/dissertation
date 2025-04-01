import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils.plotting import plot_over_time_multiple_subplots

# Read the CSV file
df = pd.read_csv(
    "results/simplemoenv/dwl_init_learn_steps_num_200__gamma_0_0__decay_epsilon_epsilon_0_99__epsilon_decay_0_998__epsilon_min_0_1/results.csv"
)

# Extract loss values and convert from string to numeric
# The loss column contains strings like "[(nan, nan), (nan, nan)]"
# We'll need to parse this carefully
# Convert string representation of loss to actual values
colors = plt.cm.viridis(np.linspace(0, 1, 2))
loss_str = df["loss"].values
losses = []
for loss_tuple_str in loss_str:
    # Remove brackets and split into individual tuples
    loss_tuple_str = loss_tuple_str.strip("[]")
    loss_pairs = loss_tuple_str.split("), (")
    loss_pairs = [pair.strip("()") for pair in loss_pairs]

    # Parse each pair into q_loss and w_loss
    q_losses = []
    w_losses = []
    for pair in loss_pairs:
        if pair:
            q, w = map(float, pair.split(","))
            q_losses.append(q)
            w_losses.append(w)
    losses.append([q_losses, w_losses])

# Convert to numpy array and reshape
losses = np.array(losses)
q_losses = losses[:, 0, :].T  # Shape: (2, num_episodes)
w_losses = losses[:, 1, :].T  # Shape: (2, num_episodes)

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot Q-losses
for i in range(q_losses.shape[0]):
    ax1.plot(df["episode"], q_losses[i], label=f"Objective {i+1}", color=colors[i], alpha=0.7)
ax1.set_title("Q-Network Loss")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

# Plot W-losses
for i in range(w_losses.shape[0]):
    ax2.plot(df["episode"], w_losses[i], label=f"Objective {i+1}", color=colors[i], alpha=0.7)
ax2.set_title("W-Network Loss")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.grid(True)
# Save the figure with high resolution
plt.savefig("loss_plots.png", dpi=500, bbox_inches="tight")

plt.show()
