import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

print("Loading data/top10_8days.csv (this takes a few seconds)...")
df = pd.read_csv("data/top10_8days.csv")

# Clean any missing values
df = df.dropna(subset=["timestamp", "cpu_utilization", "memory_utilization"])

# Aggregate container-level rows into 1-minute cluster-level time-series steps
# Total active container instances * average CPU serves as a direct proxy for cluster request load (RPS)
agg_df = (
    df.groupby("timestamp")
    .agg(
        cpu_util=("cpu_utilization", "mean"),
        mem_util=("memory_utilization", "mean"),
        active_containers=("msinstanceid", "count"),
        total_cpu_demand=("cpu_utilization", "sum"),
    )
    .reset_index()
    .sort_values("timestamp")
    .reset_index(drop=True)
)

# Convert timestamp from milliseconds to seconds
ts_sec = agg_df["timestamp"] / 1000.0

# Assign Day 1 through Day 8 (86,400 seconds per day)
agg_df["day"] = (ts_sec // 86400).astype(int) + 1

# Cyclical encodings for time-of-day and day-of-week (as specified in proposal)
sec_in_day = 86400.0
sec_in_week = 7.0 * 86400.0
agg_df["tod_sin"] = np.sin(2 * np.pi * (ts_sec % sec_in_day) / sec_in_day)
agg_df["tod_cos"] = np.cos(2 * np.pi * (ts_sec % sec_in_day) / sec_in_day)
agg_df["dow_sin"] = np.sin(2 * np.pi * (ts_sec % sec_in_week) / sec_in_week)
agg_df["dow_cos"] = np.cos(2 * np.pi * (ts_sec % sec_in_week) / sec_in_week)

# Chronological Split: Days 1-5 Train, Day 6 Val, Days 7-8 Test
train_df = agg_df[agg_df["day"].isin([1, 2, 3, 4, 5])].copy()
val_df = agg_df[agg_df["day"] == 6].copy()
test_df = agg_df[agg_df["day"].isin([7, 8])].copy()

# Fit Min-Max Scaler ONLY on Training data to prevent data leakage
scale_cols = ["cpu_util", "mem_util", "total_cpu_demand"]
scaler = MinMaxScaler()

train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
val_df[scale_cols] = scaler.transform(val_df[scale_cols])
test_df[scale_cols] = scaler.transform(test_df[scale_cols])

# Rename total_cpu_demand to 'rps' (normalized incoming workload rate) for clarity
for d in (train_df, val_df, test_df):
    d.rename(columns={"total_cpu_demand": "rps"}, inplace=True)

# Save clean splits
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print(f"Preprocessing complete!")
print(f"  Train (Days 1-5): {len(train_df)} timesteps -> saved to data/train.csv")
print(f"  Val   (Day 6):    {len(val_df)} timesteps -> saved to data/val.csv")
print(f"  Test  (Days 7-8): {len(test_df)} timesteps -> saved to data/test.csv")