# Autonomous Microservice Auto-Scaling via Deep Reinforcement Learning

**Authors:** Vansh Vansh & Ananya Singh  
**Domain:** Cloud Infrastructure & FinOps  

---

## Overview

Modern cloud-native applications rely on elastic microservice auto-scaling to handle volatile user demand. Traditional rule-based auto-scalers (e.g., Kubernetes Horizontal Pod Autoscaler triggering at `CPU > 80%`) are purely reactive—leading to **cold-start SLA latency breaches** during sudden traffic bursts and **expensive idle over-provisioning** during low-demand lulls.

This project formulates horizontal container scaling as a **Markov Decision Process (MDP)** and benchmarks value-based (**DQN**) and policy-gradient (**PPO**) Deep Reinforcement Learning agents against a standard static heuristic using real-world production telemetry from the **Alibaba Cluster Trace v2022**.

---

## Key Features & Methodology

* **Real-World Telemetry Pipeline:** Streams and filters the top 10 most trafficked microservices across 8 days of the Alibaba v2022 dataset ($5.45\text{M}$ container records aggregated into 1-minute cluster time-series steps).
* **Leakage-Free Time-Series Engineering:** Enforces a strict chronological split (**Days 1–5** Train, **Day 6** Validation, **Days 7–8** Test), applies cyclical $\sin/\cos$ encodings for time-of-day and day-of-week periodicity, and fits Min-Max normalization exclusively on training data.
* **Custom Gymnasium Simulation (`MicroserviceAutoScaleEnv`):**
  * **State Space ($S_t$):** Simulated CPU utilization, memory utilization, normalized active and pending (cold-start) container counts, incoming workload velocity/trend (`rps_delta`, `rps_trend`), cyclical temporal features, and a $k=5$ sliding window of historical incoming workload (RPS).
  * **Action Space ($A_t$):** Discrete scaling actions $\{-1 \text{ (terminate)}, 0 \text{ (hold)}, +1 \text{ (provision)}\}$ incorporating realistic **cold-start container boot delays** and non-linear queuing latency.
  * **Composite FinOps Reward ($R_t$):** Balances Service Level Agreement (SLA) protection against infrastructure waste:
    $$R_t = -\alpha \cdot \max\left(0, \frac{L_t - L_{\text{SLA}}}{L_{\text{SLA}}}\right) - \beta \cdot \left(\frac{N_{\text{idle}, t}}{N_{\max}}\right)$$
* **FinOps Sensitivity & Explainability:** Includes validation tuning across $(\text{lr}, \gamma, \text{exploration/entropy})$, 5-seed robustness evaluation against an exact Dynamic Programming (DP) oracle ceiling, a multi-regime $(\alpha, \beta)$ **Pareto Frontier** sweep, smoothed convergence diagnostics, Q-value decision heatmaps, and an interactive **Streamlit** stress-testing dashboard.

---

## Repository Structure

```text
microservice-autoscaler/
├── data/
│   ├── train.csv                 # Days 1–5 (150 1-minute steps, Min-Max fitted)
│   ├── val.csv                   # Day 6 (30 steps for DQN hyperparameter tuning)
│   └── test.csv                  # Days 7–8 (60 steps unseen evaluation split)
├── models/
│   ├── dqn_autoscaler.zip        # Trained Deep Q-Network policy
│   └── ppo_autoscaler.zip        # Trained Proximal Policy Optimization policy
├── results/
│   ├── benchmark_summary.csv     # Quantitative test set benchmark table
│   ├── multiseed_summary.csv     # 5-seed mean +/- std & DP oracle ceiling
│   ├── best_hyperparams.json     # Validation-tuned DQN & PPO hyperparameters
│   ├── pareto_frontier.csv       # Cost vs. SLA sensitivity sweep data
│   ├── scaling_comparison.png    # Time-series workload, node allocation & latency
│   ├── metrics_bar_chart.png     # SLA violations & average active nodes comparison
│   ├── learning_curves.png       # Smoothed episode reward convergence curves
│   ├── pareto_frontier.png       # FinOps (alpha, beta) cost-latency frontier
│   └── policy_heatmap.png        # Learned DQN decision surface & action breakdown
├── src/
│   ├── download_data.py          # Streams & filters top 10 microservices across 8 days
│   ├── preprocess.py             # Aggregates cluster load, cyclical encodings & splits
│   ├── env.py                    # Custom Gymnasium MDP environment
│   ├── train_and_eval.py         # Tunes DQN, trains DQN/PPO, and runs benchmarks
│   ├── bonus_experiments.py      # Generates training learning curves & Pareto frontier
│   ├── explainability.py         # Generates DQN Q-value decision map & action stats
│   └── app.py                    # Live interactive Streamlit FinOps simulator dashboard
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Experimental Results (Unseen Test Set: Days 7–8)

| Scaling Policy | Total Reward | SLA Violations | Avg Latency (ms) | Avg Active Nodes (Cost) | Avg Idle Nodes (Waste) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Static Heuristic (`CPU > 80%`)** | -112.54 | 21 | 123.38 ms | 14.87 | 4.35 |
| **DQN (Tuned Value-Based RL)** | **-111.40** | 22 | 141.96 ms | **11.57 (-22.2%)** | **1.06 (-75.7%)** |
| **PPO (Tuned + VecNorm RL)** | -112.59 | **21** | **123.28 ms** | 14.98 | 4.46 |
| *DP Theoretical Oracle (Upper Bound)* | *-110.72* | *21* | *145.63 ms* | *11.28* | *0.76* |

### Key Takeaways
1. **Near-Optimal FinOps Efficiency (DQN):** With per-day episode sampling, surge augmentation, and workload velocity/trend features, DQN achieves `-111.40` total reward (within `0.68` of the exact DP oracle ceiling `-110.72`), cutting average active containers by **22.2%** and slashing idle node waste by **75.7%**.
2. **SLA-First Stability (PPO + VecNormalize):** Reward normalization eliminates PPO training collapse across all 5 seeds (`-112.59 ± 0.00`), matching the heuristic's minimal `21` SLA violations (the unavoidable floor during Day 7 bursts where `rps > 0.8885`).
3. **Configurable Trade-Offs:** Adjusting $(\alpha, \beta)$ shifts the learned policy smoothly along the FinOps Pareto frontier, enabling cloud engineers to dial between strict latency guarantees and aggressive cost reduction.

---

## Quickstart & Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/<your-username>/microservice-autoscaler.git
cd microservice-autoscaler

python -m venv venv
# Windows Command Prompt:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline
*(Note: Pre-processed splits in `data/` and trained weights in `models/` are already included; run the scripts below to regenerate from scratch.)*
```bash
python src/download_data.py
python src/preprocess.py
python src/train_and_eval.py
python src/bonus_experiments.py
python src/explainability.py
```

### 3. Launch Live Interactive Dashboard
```bash
streamlit run src/app.py
```
Open `http://localhost:8501` in your browser to adjust the **Traffic Surge Multiplier (RPS)** and **SLA Latency Threshold (ms)** sliders and inspect offline evaluation charts.