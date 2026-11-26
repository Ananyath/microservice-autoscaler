import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from env import MicroserviceAutoScaleEnv

st.set_page_config(page_title="DRL Microservice Auto-Scaler", layout="wide")
st.title("Autonomous Microservice Auto-Scaling via Deep Reinforcement Learning")
st.markdown("**Live Interactive FinOps Simulator** | Alibaba Cluster Trace (Days 7–8 Test Split)")


@st.cache_data
def load_test_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_models():
    return DQN.load("models/dqn_autoscaler"), PPO.load("models/ppo_autoscaler")


# Sidebar controls for live demo
st.sidebar.header("Simulation Stress-Test Controls")
traffic_mult = st.sidebar.slider("Traffic Surge Multiplier (RPS)", min_value=0.5, max_value=1.8, value=1.0, step=0.1)
sla_limit = st.sidebar.slider("SLA Latency Threshold (ms)", min_value=60, max_value=150, value=100, step=10)

tab1, tab2 = st.tabs(["Live Interactive Simulator", "Offline Research & Evaluation Charts"])

with tab1:
    test_path = "data/test.csv"
    if not os.path.exists(test_path):
        st.error("Missing `data/test.csv`. Place the pre-processed test split in `data/test.csv` to run live simulations (see Tab 2 for saved offline charts).")
    else:
        test_df = load_test_data(test_path).copy()
        test_df["rps"] = test_df["rps"] * traffic_mult

        dqn_model, ppo_model = load_models()

        def simulate(policy_type):
            env = MicroserviceAutoScaleEnv(test_df, sla_threshold_ms=float(sla_limit))
            obs, _ = env.reset()
            records = []
            done = False
            while not done:
                if policy_type == "heuristic":
                    sim_cpu = obs[0]
                    action = 2 if sim_cpu > 0.80 else (0 if sim_cpu < 0.30 else 1)
                elif policy_type == "dqn":
                    action, _ = dqn_model.predict(obs, deterministic=True)
                else:
                    action, _ = ppo_model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                records.append(info)
                done = terminated or truncated
            return pd.DataFrame(records)

        heur_res = simulate("heuristic")
        dqn_res = simulate("dqn")
        ppo_res = simulate("ppo")

        heur_sla = int(heur_res["sla_violated"].sum())
        heur_nodes = float(heur_res["active_nodes"].mean())
        heur_idle = float(heur_res["idle_nodes"].mean())
        heur_rew = float(heur_res["reward"].sum())

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Static Heuristic (CPU > 80%)")
            st.metric("SLA Violations", heur_sla)
            st.metric("Avg Active Nodes", f"{heur_nodes:.2f}")
            st.metric("Avg Idle Nodes (Waste)", f"{heur_idle:.2f}")
            st.metric("Avg Latency", f"{heur_res['latency_ms'].mean():.1f} ms")
            st.metric("Total FinOps Reward", f"{heur_rew:.2f}")

        with col2:
            st.subheader("DQN Agent (Value-Based)")
            dqn_sla = int(dqn_res["sla_violated"].sum())
            dqn_nodes = float(dqn_res["active_nodes"].mean())
            dqn_idle = float(dqn_res["idle_nodes"].mean())
            dqn_rew = float(dqn_res["reward"].sum())
            st.metric("SLA Violations", dqn_sla, delta=dqn_sla - heur_sla, delta_color="inverse")
            st.metric("Avg Active Nodes", f"{dqn_nodes:.2f}", delta=f"{dqn_nodes - heur_nodes:.2f}", delta_color="inverse")
            st.metric("Avg Idle Nodes (Waste)", f"{dqn_idle:.2f}", delta=f"{dqn_idle - heur_idle:.2f}", delta_color="inverse")
            st.metric("Avg Latency", f"{dqn_res['latency_ms'].mean():.1f} ms")
            st.metric("Total FinOps Reward", f"{dqn_rew:.2f}", delta=f"{dqn_rew - heur_rew:.2f}")

        with col3:
            st.subheader("PPO Agent (Policy-Gradient)")
            ppo_sla = int(ppo_res["sla_violated"].sum())
            ppo_nodes = float(ppo_res["active_nodes"].mean())
            ppo_idle = float(ppo_res["idle_nodes"].mean())
            ppo_rew = float(ppo_res["reward"].sum())
            st.metric("SLA Violations", ppo_sla, delta=ppo_sla - heur_sla, delta_color="inverse")
            st.metric("Avg Active Nodes", f"{ppo_nodes:.2f}", delta=f"{ppo_nodes - heur_nodes:.2f}", delta_color="inverse")
            st.metric("Avg Idle Nodes (Waste)", f"{ppo_idle:.2f}", delta=f"{ppo_idle - heur_idle:.2f}", delta_color="inverse")
            st.metric("Avg Latency", f"{ppo_res['latency_ms'].mean():.1f} ms")
            st.metric("Total FinOps Reward", f"{ppo_rew:.2f}", delta=f"{ppo_rew - heur_rew:.2f}")

        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        axes[0].plot(heur_res["active_nodes"], "--", color="#d9534f", label="Static Heuristic")
        axes[0].plot(dqn_res["active_nodes"], "-", color="#0275d8", linewidth=2, label="DQN Agent")
        axes[0].plot(ppo_res["active_nodes"], "-.", color="#5cb85c", linewidth=2, label="PPO Agent")
        axes[0].set_ylabel("Active Containers")
        axes[0].set_title("Real-Time Container Allocation & Response Latency")
        axes[0].legend()

        axes[1].plot(heur_res["latency_ms"], "--", color="#d9534f", label="Static Heuristic")
        axes[1].plot(dqn_res["latency_ms"], "-", color="#0275d8", linewidth=2, label="DQN Agent")
        axes[1].plot(ppo_res["latency_ms"], "-.", color="#5cb85c", linewidth=2, label="PPO Agent")
        axes[1].axhline(sla_limit, color="black", linestyle=":", label=f"SLA Limit ({sla_limit} ms)")
        axes[1].set_ylabel("Latency (ms)")
        axes[1].set_xlabel("Test Timesteps (Days 7–8)")
        axes[1].legend()

        st.pyplot(fig)

with tab2:
    if os.path.exists("results/multiseed_summary.csv"):
        st.subheader("Multi-Seed Test Set Evaluation (5 Seeds, Mean ± Std)")
        st.dataframe(pd.read_csv("results/multiseed_summary.csv"), use_container_width=True, hide_index=True)

    st.subheader("Saved Training, FinOps & Explainability Charts")
    charts = [
        ("Training Convergence Curves", "results/learning_curves.png"),
        ("FinOps Cost vs. SLA Pareto Frontier", "results/pareto_frontier.png"),
        ("DQN Decision Boundary & Action Breakdown", "results/policy_heatmap.png"),
        ("Test Set Benchmark Summary", "results/metrics_bar_chart.png"),
    ]
    c1, c2 = st.columns(2)
    for idx, (title, path) in enumerate(charts):
        target_col = c1 if idx % 2 == 0 else c2
        with target_col:
            if os.path.exists(path):
                st.markdown(f"**{title}**")
                st.image(path, use_container_width=True)
            else:
                st.warning(f"Missing `{path}` — run the generator script first.")
