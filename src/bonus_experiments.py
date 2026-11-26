import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import MicroserviceAutoScaleEnv

os.makedirs("results", exist_ok=True)
os.makedirs("logs/dqn", exist_ok=True)
os.makedirs("logs/ppo", exist_ok=True)

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Load tuned hyperparameters if available
dqn_params = {"learning_rate": 1e-3, "gamma": 0.95, "exploration_fraction": 0.25}
ppo_params = {"learning_rate": 3e-4, "gamma": 0.95, "ent_coef": 0.005}
if os.path.exists("results/best_hyperparams.json"):
    with open("results/best_hyperparams.json") as f:
        best = json.load(f)
        dqn_params.update(best.get("dqn", {}))
        ppo_params.update(best.get("ppo", {}))

# ==============================================================================
# PART A: TRAINING CONVERGENCE DIAGNOSTICS (DQN vs. PPO Learning Curves)
# ==============================================================================
print("1/2 Running Convergence Diagnostics (logging episode rewards over 60k steps)...")

dqn_mon_env = Monitor(
    MicroserviceAutoScaleEnv(train_df, sample_by_day=True, traffic_jitter=True),
    filename="logs/dqn/monitor.csv",
)
dqn = DQN(
    "MlpPolicy", dqn_mon_env, learning_rate=dqn_params["learning_rate"], buffer_size=15000,
    learning_starts=500, batch_size=64, gamma=dqn_params["gamma"],
    exploration_fraction=dqn_params["exploration_fraction"], exploration_final_eps=0.02,
    target_update_interval=500, policy_kwargs={"net_arch": [128, 128]}, verbose=0, seed=42
)
dqn.learn(total_timesteps=60000)

ppo_mon_raw = Monitor(
    MicroserviceAutoScaleEnv(train_df, sample_by_day=True, traffic_jitter=True),
    filename="logs/ppo/monitor.csv",
)
ppo_vec_env = VecNormalize(DummyVecEnv([lambda: ppo_mon_raw]), norm_obs=False, norm_reward=True, clip_reward=10.0)
ppo = PPO(
    "MlpPolicy", ppo_vec_env, learning_rate=ppo_params["learning_rate"], n_steps=256,
    batch_size=64, gamma=ppo_params["gamma"], ent_coef=ppo_params["ent_coef"],
    policy_kwargs={"net_arch": [128, 128]}, verbose=0, seed=42
)
ppo.learn(total_timesteps=60000)

# Read monitor logs and smooth episode rewards
dqn_log = pd.read_csv("logs/dqn/monitor.csv", skiprows=1)
ppo_log = pd.read_csv("logs/ppo/monitor.csv", skiprows=1)

dqn_log["cum_steps"] = dqn_log["l"].cumsum()
ppo_log["cum_steps"] = ppo_log["l"].cumsum()

dqn_smooth = dqn_log["r"].rolling(window=15, min_periods=1).mean()
ppo_smooth = ppo_log["r"].rolling(window=15, min_periods=1).mean()

plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(9, 4.5))
plt.plot(dqn_log["cum_steps"], dqn_smooth, color="#0275d8", linewidth=2.2, label="DQN (Value-Based RL)")
plt.plot(ppo_log["cum_steps"], ppo_smooth, color="#5cb85c", linewidth=2.2, label="PPO (Policy-Gradient RL)")
plt.title("Training Convergence Curve: Smoothed Episode Reward vs. Timesteps", fontsize=12, fontweight="bold")
plt.xlabel("Training Timesteps (Days 1–5 Environment)")
plt.ylabel("Episode Cumulative Reward (Higher = Better)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results/learning_curves.png", dpi=300)
plt.close()
print("  Saved -> results/learning_curves.png")

# ==============================================================================
# PART B: PARETO FRONTIER SWEEP (FinOps Cost vs. SLA Latency Trade-off)
# ==============================================================================
print("2/2 Running Pareto Frontier Sweep across (alpha, beta) FinOps regimes...")

regimes = [
    {"name": "Cost-First\n(α=1.0, β=0.50)", "alpha": 1.0, "beta": 0.50},
    {"name": "Balanced\n(α=2.5, β=0.25)",   "alpha": 2.5, "beta": 0.25},
    {"name": "SLA-Focused\n(α=4.0, β=0.15)","alpha": 4.0, "beta": 0.15},
    {"name": "Zero-Breach\n(α=8.0, β=0.05)","alpha": 8.0, "beta": 0.05},
]

pareto_rows = []
for reg in regimes:
    env_train = VecNormalize(
        DummyVecEnv([lambda r=reg: MicroserviceAutoScaleEnv(train_df, alpha=r["alpha"], beta=r["beta"], sample_by_day=True, traffic_jitter=True)]),
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
    )
    agent = PPO(
        "MlpPolicy", env_train, learning_rate=ppo_params["learning_rate"], n_steps=256,
        batch_size=64, gamma=ppo_params["gamma"], ent_coef=ppo_params["ent_coef"],
        policy_kwargs={"net_arch": [128, 128]}, verbose=0, seed=42
    )
    agent.learn(total_timesteps=40000)

    # Evaluate on unseen Test Set (Days 7-8)
    env_test = MicroserviceAutoScaleEnv(test_df, alpha=reg["alpha"], beta=reg["beta"])
    obs, _ = env_test.reset()
    done = False
    nodes_list, violations = [], 0
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env_test.step(action)
        nodes_list.append(info["active_nodes"])
        violations += info["sla_violated"]
        done = terminated or truncated

    pareto_rows.append({
        "Regime": reg["name"],
        "Avg_Nodes": np.mean(nodes_list),
        "SLA_Violations": violations,
    })

pareto_df = pd.DataFrame(pareto_rows).sort_values("Avg_Nodes")
pareto_df.to_csv("results/pareto_frontier.csv", index=False)

plt.figure(figsize=(8.5, 4.8))
plt.plot(pareto_df["Avg_Nodes"], pareto_df["SLA_Violations"], "-o", color="#6f42c1", linewidth=2.5, markersize=9)
for _, r in pareto_df.iterrows():
    plt.annotate(
        r["Regime"],
        (r["Avg_Nodes"], r["SLA_Violations"]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
        fontweight="bold"
    )
plt.title("FinOps Pareto Frontier: Infrastructure Cost vs. SLA Violations (Test Set)", fontsize=12, fontweight="bold")
plt.xlabel("Average Active Container Nodes (Infrastructure Cost)")
plt.ylabel("Total SLA Violations (Latency > 100 ms)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("results/pareto_frontier.png", dpi=300)
plt.close()
print("  Saved -> results/pareto_frontier.png")
print("\nBonus experiments complete!")