import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from env import MicroserviceAutoScaleEnv

test_df = pd.read_csv("data/test.csv")
dqn_model = DQN.load("models/dqn_autoscaler")
ppo_model = PPO.load("models/ppo_autoscaler")

# 1. Generate DQN Decision Boundary Heatmap (CPU Load vs. Active Nodes)
loads = np.linspace(0.1, 1.2, 30)
nodes = np.arange(1, 17)
action_grid = np.zeros((len(nodes), len(loads)))

for i, n in enumerate(nodes):
    for j, rps in enumerate(loads):
        cap = n / 16.0
        sim_cpu = np.clip(rps / cap, 0.0, 2.0)
        # Construct synthetic state vector matching env observation space
        obs = np.array([sim_cpu, sim_cpu * 0.8, cap, 0.0, 1.0, 0.0, 1.0] + [rps]*5, dtype=np.float32)
        act, _ = dqn_model.predict(obs, deterministic=True)
        action_grid[i, j] = act - 1  # Map {0,1,2} -> {-1, 0, +1}

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# Left Plot: Learned Policy Decision Map
im = axes[0].imshow(action_grid, origin="lower", aspect="auto", cmap="coolwarm", vmin=-1, vmax=1,
                    extent=[loads[0], loads[-1], nodes[0], nodes[-1]])
cbar = plt.colorbar(im, ax=axes[0], ticks=[-1, 0, 1])
cbar.ax.set_yticklabels(["Scale Down (-1)", "Hold (0)", "Scale Up (+1)"])
axes[0].set_title("Learned DQN Policy Decision Surface", fontweight="bold")
axes[0].set_xlabel("Incoming Workload (Normalized RPS)")
axes[0].set_ylabel("Currently Active Containers")

# Right Plot: Action Frequency Distribution on Test Set
def get_action_counts(policy_type):
    env = MicroserviceAutoScaleEnv(test_df)
    obs, _ = env.reset()
    counts = {-1: 0, 0: 0, 1: 0}
    done = False
    while not done:
        if policy_type == "Heuristic":
            act = 2 if obs[0] > 0.80 else (0 if obs[0] < 0.30 else 1)
        elif policy_type == "DQN":
            act, _ = dqn_model.predict(obs, deterministic=True)
        else:
            act, _ = ppo_model.predict(obs, deterministic=True)
        counts[int(act) - 1] += 1
        obs, _, terminated, truncated, _ = env.step(act)
        done = terminated or truncated
    total = sum(counts.values())
    return [counts[-1]/total*100, counts[0]/total*100, counts[1]/total*100]

labels = ["Scale Down (-1)", "Hold (0)", "Scale Up (+1)"]
x = np.arange(len(labels))
w = 0.25
axes[1].bar(x - w, get_action_counts("Heuristic"), width=w, color="#d9534f", label="Static Heuristic")
axes[1].bar(x,     get_action_counts("DQN"),       width=w, color="#0275d8", label="DQN Agent")
axes[1].bar(x + w, get_action_counts("PPO"),       width=w, color="#5cb85c", label="PPO Agent")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].set_ylabel("Percentage of Test Decisions (%)")
axes[1].set_title("Action Distribution on Unseen Test Set", fontweight="bold")
axes[1].legend()

plt.tight_layout()
plt.savefig("results/policy_heatmap.png", dpi=300)
plt.close()
print("Saved -> results/policy_heatmap.png")