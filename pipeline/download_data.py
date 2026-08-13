# -*- coding: utf-8 -*-
"""Download real open datasets for the shoe QA demo.

Leg 1: VisA (CC BY 4.0) cashew subset  -> real industrial anomaly benchmark
Leg 2: ipogorelov/sneakers (MIT)       -> real sneaker photos (normal refs + demo samples)
"""
import os, sys, io, json
from huggingface_hub import hf_hub_download

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(ROOT, exist_ok=True)

VISA_REPO = "imaadd05/visa-anomaly-detection"
N_NORMAL = 160   # memory bank + normal test
N_ANOM = 60      # anomaly test subset

def dl(repo, fname, subdir):
    return hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset",
                           local_dir=os.path.join(ROOT, subdir))

def leg1_visa():
    print("== VisA cashew subset ==", flush=True)
    for i in range(N_NORMAL):
        dl(VISA_REPO, f"cashew/Data/Images/Normal/{i:03d}.JPG", "visa")
        if i % 20 == 0: print(f"normal {i}/{N_NORMAL}", flush=True)
    for i in range(N_ANOM):
        dl(VISA_REPO, f"cashew/Data/Images/Anomaly/{i:03d}.JPG", "visa")
        dl(VISA_REPO, f"cashew/Data/Masks/Anomaly/{i:03d}.png", "visa")
        if i % 10 == 0: print(f"anomaly {i}/{N_ANOM}", flush=True)
    print("VisA done", flush=True)

def leg2_sneakers():
    print("== sneakers parquet ==", flush=True)
    p = dl("ipogorelov/sneakers", "dataset_batch_01.parquet", "sneakers")
    print("parquet at", p, flush=True)
    import pyarrow.parquet as pq
    t = pq.read_file if False else pq.read_table(p)
    print("schema:", t.schema, flush=True)
    print("rows:", t.num_rows, flush=True)

if __name__ == "__main__":
    leg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if leg in ("all", "visa"): leg1_visa()
    if leg in ("all", "sneakers"): leg2_sneakers()
