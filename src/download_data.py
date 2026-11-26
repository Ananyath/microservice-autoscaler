import os
import glob
import urllib.request
import tarfile
import pandas as pd

os.makedirs("data/MSMetrics", exist_ok=True)
base_url = "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2022MicroservicesTraces"

def process_and_delete_csvs(top_services=None):
    chunks = []
    for csv_path in glob.glob("data/MSMetrics/*.csv"):
        print(f"  Filtering {os.path.basename(csv_path)}...")
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        
        if top_services is None:
            top_services = (
                df.groupby("msname")["msinstanceid"]
                .count()
                .nlargest(10)
                .index.tolist()
            )
        
        filtered = df[df["msname"].isin(top_services)].copy()
        chunks.append(filtered)
        os.remove(csv_path)
    return chunks, top_services

all_chunks = []
top_10 = None

if glob.glob("data/MSMetrics/*.csv"):
    print("Cleaning up already downloaded CSVs to free space...")
    existing_chunks, top_10 = process_and_delete_csvs(top_10)
    all_chunks.extend(existing_chunks)

CHUNKS_PER_DAY = int(os.environ.get("CHUNKS_PER_DAY", "1"))
# Alibaba v2022 splits each day into 48 30-minute files (0..47)
step = max(1, 48 // CHUNKS_PER_DAY)

for day in range(2, 8):
    for offset in range(24 % step, 48, step)[:CHUNKS_PER_DAY]:
        ms_idx = day * 48 + offset
        ms_url = f"{base_url}/MSMetricsUpdate/MSMetricsUpdate_{ms_idx}.tar.gz"
        ms_tar = f"data/MSMetrics/MSMetrics_{ms_idx}.tar.gz"

        print(f"[Day {day+1}/8 | Chunk {offset}] Downloading MSMetrics_{ms_idx}.tar.gz...")
        urllib.request.urlretrieve(ms_url, ms_tar)
        with tarfile.open(ms_tar, "r:gz") as tar:
            tar.extractall("data/MSMetrics")
        os.remove(ms_tar)

        day_chunks, top_10 = process_and_delete_csvs(top_10)
        all_chunks.extend(day_chunks)

final_df = pd.concat(all_chunks, ignore_index=True)
final_df.to_csv("data/top10_8days.csv", index=False)
print(f"\nDone! Saved {len(final_df)} rows to data/top10_8days.csv. Disk space freed!")