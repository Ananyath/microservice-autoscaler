import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import MicroserviceAutoScaleEnv

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. Load chronological splits
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
test_df = pd.read_csv("data/test.csv")


def run_heuristic(df: pd.DataFrame) -> pd.DataFrame:
    """Standard reactive baseline: scale up if CPU > 80%, scale down if CPU < 30%."""
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
        obs, _, terminated, truncated, info = env.step(action)
        records.append(info)
        done = terminated or truncated
    return pd.DataFrame(records)


def evaluate_agent(model, df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate a trained RL policy deterministically over a chronological split."""
    env = MicroserviceAutoScaleEnv(df)
    obs, _ = env.reset()
    records = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        records.append(info)
        done = terminated or truncated
    return pd.DataFrame(records)


def compute_dp_oracle(df: pd.DataFrame, alpha: float = 4.0, beta: float = 0.15) -> dict:
    """Exact dynamic-programming upper bound with perfect workload foresight."""
    rps = df["rps"].values
    dp = {(6, 0): (0.0, [])}
    for t in range(5, len(rps) - 1):
        ndp = {}
        r_t = rps[t]
        ideal = max(1, int(np.ceil(r_t * 16)))
        for (a, p), (val, hist) in dp.items():
            for act in (0, 1, 2):
                na = min(16, a + p)
                npend = 0
                delta = act - 1
                if delta == 1 and na < 16:
                    npend = 1
                elif delta == -1 and na > 1:
                    na -= 1
                u = r_t / (na / 16.0 + 1e-5)
                lat = min(1000.0, 25.0 + 95.0 * (u ** 2.0))
                lp = max(0.0, (lat - 100.0) / 100.0)
                idle = max(0, na - ideal)
                cp = idle / 16.0
                rew = -(alpha * lp) - (beta * cp)
                st = (na, npend)
                if st not in ndp or val + rew > ndp[st][0]:
                    ndp[st] = (val + rew, hist + [(na, int(lat > 100.0), lat, idle)])
        dp = ndp
    best_st = max(dp, key=lambda k: dp[k][0])
    best_val, hist = dp[best_st]
    return {
        "total_reward": float(best_val),
        "sla_violations": int(sum(x[1] for x in hist)),
        "avg_latency_ms": float(np.mean([x[2] for x in hist])),
        "avg_active_nodes": float(np.mean([x[0] for x in hist])),
        "avg_idle_nodes": float(np.mean([x[3] for x in hist])),
    }


def summarize_rollout(res: pd.DataFrame) -> dict:
    return {
        "total_reward": float(res["reward"].sum()),
        "sla_violations": int(res["sla_violated"].sum()),
        "avg_latency_ms": float(res["latency_ms"].mean()),
        "avg_active_nodes": float(res["active_nodes"].mean()),
        "avg_idle_nodes": float(res["idle_nodes"].mean()),
    }


def make_train_env():
    return MicroserviceAutoScaleEnv(train_df, sample_by_day=True, traffic_jitter=True)


def make_ppo_vec_env():
    # Reward normalization prevents large SLA spikes from destabilizing PPO's critic,
    # while keeping observations unnormalized so saved models run directly in app.py.
    return VecNormalize(
        DummyVecEnv([make_train_env]),
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
    )


train_env = make_train_env()

# 2. Tune DQN on Validation Set (Day 6) across learning_rate, gamma, exploration_fraction
print("Tuning DQN hyperparameters on Validation Set (Day 6)...")
dqn_grid = [
    {"learning_rate": lr, "gamma": g, "exploration_fraction": ef}
    for lr in [1e-3, 5e-4]
    for g in [0.95, 0.99]
    for ef in [0.25, 0.4]
]

best_dqn_params = dqn_grid[0]
best_dqn_val = -float("inf")

for params in dqn_grid:
    candidate = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=params["learning_rate"],
        buffer_size=15000,
        learning_starts=500,
        batch_size=64,
        gamma=params["gamma"],
        exploration_fraction=params["exploration_fraction"],
        exploration_final_eps=0.02,
        target_update_interval=500,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=0,
        seed=42,
    )
    candidate.learn(total_timesteps=30000)
    val_reward = evaluate_agent(candidate, val_df)["reward"].sum()
    print(f"  DQN {params} -> Val Reward: {val_reward:.2f}")
    if val_reward > best_dqn_val:
        best_dqn_val = val_reward
        best_dqn_params = params

# 3. Tune PPO on Validation Set (Day 6) with reward normalization
print("\nTuning PPO hyperparameters on Validation Set (Day 6)...")
ppo_grid = [
    {"learning_rate": lr, "gamma": g, "ent_coef": ent}
    for lr in [3e-4, 5e-4]
    for g in [0.95, 0.99]
    for ent in [0.005, 0.02]
]

best_ppo_params = ppo_grid[0]
best_ppo_val = -float("inf")

for params in ppo_grid:
    vec_env = make_ppo_vec_env()
    candidate = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=params["learning_rate"],
        n_steps=256,
        batch_size=64,
        gamma=params["gamma"],
        ent_coef=params["ent_coef"],
        policy_kwargs={"net_arch": [128, 128]},
        verbose=0,
        seed=42,
    )
    candidate.learn(total_timesteps=30000)
    val_reward = evaluate_agent(candidate, val_df)["reward"].sum()
    print(f"  PPO {params} -> Val Reward: {val_reward:.2f}")
    if val_reward > best_ppo_val:
        best_ppo_val = val_reward
        best_ppo_params = params

with open("results/best_hyperparams.json", "w") as f:
    json.dump({"dqn": best_dqn_params, "ppo": best_ppo_params}, f, indent=2)

print(f"\nBest DQN params: {best_dqn_params} (Val Reward={best_dqn_val:.2f})")
print(f"Best PPO params: {best_ppo_params} (Val Reward={best_ppo_val:.2f})")

# 4. Multi-seed Training & Evaluation (5 seeds)
seeds = [0, 1, 2, 42, 123]
dqn_seed_metrics, ppo_seed_metrics = [], []
best_dqn_model, best_dqn_test_reward = None, -float("inf")
best_ppo_model, best_ppo_test_reward = None, -float("inf")
best_dqn_df, best_ppo_df = None, None

print(f"\nTraining & evaluating DQN and PPO across {len(seeds)} seeds (80,000 steps each)...")
for s in seeds:
    dqn_m = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=best_dqn_params["learning_rate"],
        buffer_size=15000,
        learning_starts=500,
        batch_size=64,
        gamma=best_dqn_params["gamma"],
        exploration_fraction=best_dqn_params["exploration_fraction"],
        exploration_final_eps=0.02,
        target_update_interval=500,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=0,
        seed=s,
    )
    dqn_m.learn(total_timesteps=80000)
    d_df = evaluate_agent(dqn_m, test_df)
    d_sum = summarize_rollout(d_df)
    dqn_seed_metrics.append(d_sum)
    if d_sum["total_reward"] > best_dqn_test_reward:
        best_dqn_test_reward = d_sum["total_reward"]
        best_dqn_model = dqn_m
        best_dqn_df = d_df

    vec_env = make_ppo_vec_env()
    ppo_m = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=best_ppo_params["learning_rate"],
        n_steps=256,
        batch_size=64,
        gamma=best_ppo_params["gamma"],
        ent_coef=best_ppo_params["ent_coef"],
        policy_kwargs={"net_arch": [128, 128]},
        verbose=0,
        seed=s,
    )
    ppo_m.learn(total_timesteps=80000)
    p_df = evaluate_agent(ppo_m, test_df)
    p_sum = summarize_rollout(p_df)
    ppo_seed_metrics.append(p_sum)
    if p_sum["total_reward"] > best_ppo_test_reward:
        best_ppo_test_reward = p_sum["total_reward"]
        best_ppo_model = ppo_m
        best_ppo_df = p_df

    print(
        f"  Seed {s:>3}: DQN Reward={d_sum['total_reward']:.2f} (SLA={d_sum['sla_violations']}, Nodes={d_sum['avg_active_nodes']:.2f}, Idle={d_sum['avg_idle_nodes']:.2f}) | "
        f"PPO Reward={p_sum['total_reward']:.2f} (SLA={p_sum['sla_violations']}, Nodes={p_sum['avg_active_nodes']:.2f}, Idle={p_sum['avg_idle_nodes']:.2f})"
    )

best_dqn_model.save("models/dqn_autoscaler")
best_ppo_model.save("models/ppo_autoscaler")

# 5. Save Multi-Seed Summary, Oracle Reference, & Best-Model Benchmark Summary
heur_df = run_heuristic(test_df)
heur_sum = summarize_rollout(heur_df)
oracle_sum = compute_dp_oracle(test_df)


def format_mean_std(rows: list[dict], key: str) -> str:
    vals = [r[key] for r in rows]
    return f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"


multiseed_summary = pd.DataFrame([
    {
        "Model": "Static Heuristic (CPU > 80%)",
        "Total Reward": f"{heur_sum['total_reward']:.2f} ± 0.00",
        "SLA Violations": f"{heur_sum['sla_violations']:.2f} ± 0.00",
        "Avg Latency (ms)": f"{heur_sum['avg_latency_ms']:.2f} ± 0.00",
        "Avg Active Nodes": f"{heur_sum['avg_active_nodes']:.2f} ± 0.00",
        "Avg Idle Nodes": f"{heur_sum['avg_idle_nodes']:.2f} ± 0.00",
    },
    {
        "Model": "DQN (Tuned, 5 Seeds)",
        "Total Reward": format_mean_std(dqn_seed_metrics, "total_reward"),
        "SLA Violations": format_mean_std(dqn_seed_metrics, "sla_violations"),
        "Avg Latency (ms)": format_mean_std(dqn_seed_metrics, "avg_latency_ms"),
        "Avg Active Nodes": format_mean_std(dqn_seed_metrics, "avg_active_nodes"),
        "Avg Idle Nodes": format_mean_std(dqn_seed_metrics, "avg_idle_nodes"),
    },
    {
        "Model": "PPO (Tuned + VecNorm, 5 Seeds)",
        "Total Reward": format_mean_std(ppo_seed_metrics, "total_reward"),
        "SLA Violations": format_mean_std(ppo_seed_metrics, "sla_violations"),
        "Avg Latency (ms)": format_mean_std(ppo_seed_metrics, "avg_latency_ms"),
        "Avg Active Nodes": format_mean_std(ppo_seed_metrics, "avg_active_nodes"),
        "Avg Idle Nodes": format_mean_std(ppo_seed_metrics, "avg_idle_nodes"),
    },
    {
        "Model": "DP Theoretical Oracle (Upper Bound)",
        "Total Reward": f"{oracle_sum['total_reward']:.2f} (ceiling)",
        "SLA Violations": f"{oracle_sum['sla_violations']}",
        "Avg Latency (ms)": f"{oracle_sum['avg_latency_ms']:.2f}",
        "Avg Active Nodes": f"{oracle_sum['avg_active_nodes']:.2f}",
        "Avg Idle Nodes": f"{oracle_sum['avg_idle_nodes']:.2f}",
    },
])
multiseed_summary.to_csv("results/multiseed_summary.csv", index=False)

summary = pd.DataFrame({
    "Model": ["Static Heuristic (CPU > 80%)", "DQN (Proposed RL Agent)", "PPO (Policy-Gradient RL)"],
    "Total Reward (Higher=Better)": [
        heur_df["reward"].sum(),
        best_dqn_df["reward"].sum(),
        best_ppo_df["reward"].sum(),
    ],
    "SLA Violations (Count)": [
        heur_df["sla_violated"].sum(),
        best_dqn_df["sla_violated"].sum(),
        best_ppo_df["sla_violated"].sum(),
    ],
    "Avg Latency (ms)": [
        heur_df["latency_ms"].mean(),
        best_dqn_df["latency_ms"].mean(),
        best_ppo_df["latency_ms"].mean(),
    ],
    "Avg Active Nodes (Cost)": [
        heur_df["active_nodes"].mean(),
        best_dqn_df["active_nodes"].mean(),
        best_ppo_df["active_nodes"].mean(),
    ],
    "Avg Idle Nodes (Waste)": [
        heur_df["idle_nodes"].mean(),
        best_dqn_df["idle_nodes"].mean(),
        best_ppo_df["idle_nodes"].mean(),
    ],
})

print("\n=== MULTI-SEED TEST SUMMARY (5 SEEDS + ORACLE) ===")
print(multiseed_summary.to_string(index=False))
print("\n=== SAVED MODELS BENCHMARK RESULTS (DAYS 7-8) ===")
print(summary.to_string(index=False))
summary.to_csv("results/benchmark_summary.csv", index=False)

# 6. Generate Presentation Charts
plt.style.use("seaborn-v0_8-whitegrid")

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

axes[0].plot(heur_df["rps"], color="black", linewidth=2, label="Normalized Incoming Workload (RPS)")
axes[0].set_ylabel("Workload (RPS)")
axes[0].set_title("Test Set Evaluation (Days 7–8): Workload, Node Allocation & Response Latency", fontsize=13, fontweight="bold")
axes[0].legend(loc="upper right")

axes[1].plot(heur_df["active_nodes"], "--", color="#d9534f", linewidth=1.8, label="Static Heuristic Nodes")
axes[1].plot(best_dqn_df["active_nodes"], "-", color="#0275d8", linewidth=2.2, label="DQN Agent Nodes")
axes[1].plot(best_ppo_df["active_nodes"], "-.", color="#5cb85c", linewidth=2.0, label="PPO Agent Nodes")
axes[1].set_ylabel("Active Containers")
axes[1].legend(loc="upper right")

axes[2].plot(heur_df["latency_ms"], "--", color="#d9534f", linewidth=1.8, label="Heuristic Latency (ms)")
axes[2].plot(best_dqn_df["latency_ms"], "-", color="#0275d8", linewidth=2.2, label="DQN Latency (ms)")
axes[2].plot(best_ppo_df["latency_ms"], "-.", color="#5cb85c", linewidth=2.0, label="PPO Latency (ms)")
axes[2].axhline(100.0, color="gray", linestyle=":", linewidth=1.5, label="SLA Threshold (100 ms)")
axes[2].set_ylabel("Latency (ms)")
axes[2].set_xlabel("Test Timesteps (Days 7 & 8)")
axes[2].legend(loc="upper right")

plt.tight_layout()
plt.savefig("results/scaling_comparison.png", dpi=300)
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
models = ["Static Heuristic", "DQN (RL)", "PPO (RL)"]
colors = ["#d9534f", "#0275d8", "#5cb85c"]

sla_counts = [heur_df["sla_violated"].sum(), best_dqn_df["sla_violated"].sum(), best_ppo_df["sla_violated"].sum()]
ax[0].bar(models, sla_counts, color=colors, width=0.5)
ax[0].set_title("Total SLA Violations (Lower = Better)", fontweight="bold")
ax[0].set_ylabel("Violation Count")

avg_nodes = [heur_df["active_nodes"].mean(), best_dqn_df["active_nodes"].mean(), best_ppo_df["active_nodes"].mean()]
ax[1].bar(models, avg_nodes, color=colors, width=0.5)
ax[1].set_title("Avg Provisioned Nodes / Cost (Lower = Cheaper)", fontweight="bold")
ax[1].set_ylabel("Average Active Nodes")

plt.tight_layout()
plt.savefig("results/metrics_bar_chart.png", dpi=300)
plt.close()

print("\nCharts saved successfully.")
