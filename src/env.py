import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class MicroserviceAutoScaleEnv(gym.Env):
    """
    Custom Gymnasium Environment for Microservice Auto-Scaling.
    Matches MDP formulation:
      - State Space (S): CPU util, Memory util, active nodes, cyclical time features, and last k RPS values.
      - Action Space (A_t): Discrete {0: -1 terminate node, 1: 0 maintain, 2: +1 provision node}.
      - Reward (R_t): -alpha * (Latency Penalty) - beta * (Idle Node Cost Penalty).
    """
    def __init__(
        self,
        data: pd.DataFrame,
        k_history: int = 5,
        min_nodes: int = 1,
        max_nodes: int = 16,
        sla_threshold_ms: float = 100.0,
        alpha: float = 4.0,
        beta: float = 0.15,
        cold_start_steps: int = 1,
    ):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.k_history = k_history
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.sla_threshold_ms = sla_threshold_ms
        self.alpha = alpha
        self.beta = beta
        self.cold_start_steps = cold_start_steps

        self.action_space = spaces.Discrete(3)
        obs_dim = 7 + self.k_history
        self.observation_space = spaces.Box(
            low=-1.0, high=2.5, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.k_history
        self.active_nodes = 6
        self.pending_nodes = []
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.data.iloc[self.current_step]
        hist_rps = self.data["rps"].iloc[
            self.current_step - self.k_history : self.current_step
        ].values

        capacity_ratio = self.active_nodes / self.max_nodes
        sim_cpu = np.clip(row["rps"] / (capacity_ratio + 1e-5), 0.0, 2.0)
        sim_mem = np.clip(row["mem_util"] / (capacity_ratio + 1e-5), 0.0, 2.0)
        nodes_norm = capacity_ratio

        obs = np.concatenate([
            np.array([
                sim_cpu,
                sim_mem,
                nodes_norm,
                row["tod_sin"],
                row["tod_cos"],
                row["dow_sin"],
                row["dow_cos"],
            ], dtype=np.float32),
            hist_rps.astype(np.float32),
        ])
        return obs

    def step(self, action):
        # 1. Advance cold-start timers
        still_booting = []
        for timer in self.pending_nodes:
            if timer - 1 <= 0:
                self.active_nodes = min(self.max_nodes, self.active_nodes + 1)
            else:
                still_booting.append(timer - 1)
        self.pending_nodes = still_booting

        # 2. Execute discrete scaling action: 0 -> -1, 1 -> 0, 2 -> +1
        delta = int(action) - 1
        if delta == 1 and (self.active_nodes + len(self.pending_nodes)) < self.max_nodes:
            if self.cold_start_steps > 0:
                self.pending_nodes.append(self.cold_start_steps)
            else:
                self.active_nodes += 1
        elif delta == -1 and self.active_nodes > self.min_nodes:
            self.active_nodes -= 1

        # 3. Simulated queuing latency (bounded realistically up to 1000 ms timeout)
        row = self.data.iloc[self.current_step]
        capacity_ratio = self.active_nodes / self.max_nodes
        utilization = row["rps"] / (capacity_ratio + 1e-5)

        sim_latency_ms = min(1000.0, 25.0 + 95.0 * (utilization ** 2.0))
        latency_penalty = max(0.0, (sim_latency_ms - self.sla_threshold_ms) / self.sla_threshold_ms)

        # 4. Over-provisioning (idle node) cost penalty
        ideal_nodes = max(1, int(np.ceil(row["rps"] * self.max_nodes)))
        idle_nodes = max(0, self.active_nodes - ideal_nodes)
        cost_penalty = idle_nodes / self.max_nodes

        # Composite Reward Function: R = -alpha*(Latency Penalty) - beta*(Cost Penalty)
        reward = -(self.alpha * latency_penalty) - (self.beta * cost_penalty)

        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False

        info = {
            "step": self.current_step,
            "rps": float(row["rps"]),
            "cpu_util": float(utilization),
            "active_nodes": int(self.active_nodes),
            "latency_ms": float(sim_latency_ms),
            "sla_violated": int(sim_latency_ms > self.sla_threshold_ms),
            "idle_nodes": int(idle_nodes),
            "reward": float(reward),
        }
        return self._get_obs(), float(reward), terminated, truncated, info