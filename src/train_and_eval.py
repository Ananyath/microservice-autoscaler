import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from env import MicroserviceAutoScaleEnv

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. Load chronological splits
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
test_df = pd.read_csv("data/test.csv")

# 2. Define Standard Heuristic Baseline (Scale up if CPU > 80%, Scale down if CPU < 30%)
def run_heuristic(df):
    env = MicroserviceAutoScaleEnv(df)
    obs, _ = env.reset()
    records = []
    done = False
    while not done:
        sim_cpu = obs[0]
        if sim_cpu > 0.80:
            action = 2  # +1 provision node
        elif sim_cpu < 0.30:
            action = 0  # -1 terminate node
        else:
            action = 1  #  0 maintain
        obs, reward, terminated, truncated, info = env.step(action)
        records.append(info)
        done = terminated or truncated
    return pd.DataFrame(records)

# 3. Helper to evaluate any trained RL agent
def evaluate_agent(model, df):
    env = MicroserviceAutoScaleEnv(df)
    obs, _ = env.reset()
    records = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        records.append(info)
        done = terminated or truncated
    return pd.DataFrame(records)

# 4. Tune DQN on Validation Set (Day 6) across learning rates
print("Tuning DQN hyperparameters on Validation Set (Day 6)...")
train_env = MicroserviceAutoScaleEnv(train_df)
best_lr = 1e-3
best_val_reward = -float("inf")

for lr in [1e-3, 5e-4]:
    candidate = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=lr,
        buffer_size=10000,
        learning_starts=200,
        batch_size=64,
        gamma=0.95,
        exploration_fraction=0.3,
        verbose=0,
        seed=42,
    )
    candidate.learn(total_timesteps=25000)
    val_res = evaluate_agent(candidate, val_df)
    mean_rew = val_res["reward"].sum()
    print(f"  DQN (lr={lr}) -> Validation Total Reward: {mean_rew:.2f}")
    if mean_rew > best_val_reward:
        best_val_reward = mean_rew
        best_lr = lr

print(f"\nTraining Final DQN Model (best lr={best_lr}) for 60,000 steps...")
dqn_model = DQN(
    "MlpPolicy",
    train_env,
    learning_rate=best_lr,
    buffer_size=10000,
    learning_starts=200,
    batch_size=64,
    gamma=0.95,
    exploration_fraction=0.3,
    verbose=0,
    seed=42,
)
dqn_model.learn(total_timesteps=60000)
dqn_model.save("models/dqn_autoscaler")

print("Training PPO Comparison Model for 60,000 steps...")
ppo_model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=64,
    gamma=0.95,
    verbose=0,
    seed=42,
)
ppo_model.learn(total_timesteps=60000)
ppo_model.save("models/ppo_autoscaler")

# 5. Final Benchmark Evaluation on Unseen Test Set (Days 7 & 8)
print("\nEvaluating all models on Unseen Test Set (Days 7 & 8)...")
heur_df = run_heuristic(test_df)
dqn_df = evaluate_agent(dqn_model, test_df)
ppo_df = evaluate_agent(ppo_model, test_df)

# Print Summary Table
summary = pd.DataFrame({
    "Model": ["Static Heuristic (CPU > 80%)", "DQN (Proposed RL Agent)", "PPO (Policy-Gradient RL)"],
    "Total Reward (Higher=Better)": [
        heur_df["reward"].sum(),
        dqn_df["reward"].sum(),
        ppo_df["reward"].sum(),
    ],
    "SLA Violations (Count)": [
        heur_df["sla_violated"].sum(),
        dqn_df["sla_violated"].sum(),
        ppo_df["sla_violated"].sum(),
    ],
    "Avg Latency (ms)": [
        heur_df["latency_ms"].mean(),
        dqn_df["latency_ms"].mean(),
        ppo_df["latency_ms"].mean(),
    ],
    "Avg Active Nodes (Cost)": [
        heur_df["active_nodes"].mean(),
        dqn_df["active_nodes"].mean(),
        ppo_df["active_nodes"].mean(),
    ],
    "Avg Idle Nodes (Waste)": [
        heur_df["idle_nodes"].mean(),
        dqn_df["idle_nodes"].mean(),
        ppo_df["idle_nodes"].mean(),
    ],
})

print("\n=== TEST SET BENCHMARK RESULTS (DAYS 7-8) ===")
print(summary.to_string(index=False))
summary.to_csv("results/benchmark_summary.csv", index=False)

# 6. Generate Presentation Charts
plt.style.use("seaborn-v0_8-whitegrid")

# Chart 1: Node Scaling Decisions & Latency over Test Set (Days 7-8)
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

axes[0].plot(heur_df["rps"], color="black", linewidth=2, label="Normalized Incoming Workload (RPS)")
axes[0].set_ylabel("Workload (RPS)")
axes[0].set_title("Test Set Evaluation (Days 7–8): Workload, Node Allocation & Response Latency", fontsize=13, fontweight="bold")
axes[0].legend(loc="upper right")

axes[1].plot(heur_df["active_nodes"], "--", color="#d9534f", linewidth=1.8, label="Static Heuristic Nodes")
axes[1].plot(dqn_df["active_nodes"], "-", color="#0275d8", linewidth=2.2, label="DQN Agent Nodes")
axes[1].plot(ppo_df["active_nodes"], "-.", color="#5cb85c", linewidth=2.0, label="PPO Agent Nodes")
axes[1].set_ylabel("Active Containers")
axes[1].legend(loc="upper right")

axes[2].plot(heur_df["latency_ms"], "--", color="#d9534f", linewidth=1.8, label="Heuristic Latency (ms)")
axes[2].plot(dqn_df["latency_ms"], "-", color="#0275d8", linewidth=2.2, label="DQN Latency (ms)")
axes[2].plot(ppo_df["latency_ms"], "-.", color="#5cb85c", linewidth=2.0, label="PPO Latency (ms)")
axes[2].axhline(100.0, color="gray", linestyle=":", linewidth=1.5, label="SLA Threshold (100 ms)")
axes[2].set_ylabel("Latency (ms)")
axes[2].set_xlabel("Test Timesteps (Days 7 & 8)")
axes[2].legend(loc="upper right")

plt.tight_layout()
plt.savefig("results/scaling_comparison.png", dpi=300)
plt.close()

# Chart 2: Bar Chart Comparing SLA Violations & Average Node Cost
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
models = ["Static Heuristic", "DQN (RL)", "PPO (RL)"]
colors = ["#d9534f", "#0275d8", "#5cb85c"]

sla_counts = [heur_df["sla_violated"].sum(), dqn_df["sla_violated"].sum(), ppo_df["sla_violated"].sum()]
ax[0].bar(models, sla_counts, color=colors, width=0.5)
ax[0].set_title("Total SLA Violations (Lower = Better)", fontweight="bold")
ax[0].set_ylabel("Violation Count")

avg_nodes = [heur_df["active_nodes"].mean(), dqn_df["active_nodes"].mean(), ppo_df["active_nodes"].mean()]
ax[1].bar(models, avg_nodes, color=colors, width=0.5)
ax[1].set_title("Avg Provisioned Nodes / Cost (Lower = Cheaper)", fontweight="bold")
ax[1].set_ylabel("Average Active Nodes")

plt.tight_layout()
plt.savefig("results/metrics_bar_chart.png", dpi=300)
plt.close()

print("\nCharts saved successfully to:")
print("  - results/scaling_comparison.png")
print("  - results/metrics_bar_chart.png")