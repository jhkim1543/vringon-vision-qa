# -*- coding: utf-8 -*-
"""Extract sneaker images from parquet, group by model, report stats."""
import os, io, hashlib, collections
import pyarrow.parquet as pq
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(ROOT, "sneakers", "img")
os.makedirs(OUT, exist_ok=True)

t = pq.read_table(os.path.join(ROOT, "sneakers", "dataset_batch_01.parquet")).to_pandas()
print(t.groupby(["brand", "model"]).size().sort_values(ascending=False).head(20))

sizes = collections.Counter()
saved = 0
for i, row in t.iterrows():
    img = Image.open(io.BytesIO(row["image"]))
    sizes[img.size] += 1
    if i < 60:  # save first 60 for visual inspection
        key = f"{row['brand']}_{row['model']}".replace(" ", "-").replace("/", "-")[:40]
        img.convert("RGB").save(os.path.join(OUT, f"{i:04d}_{key}.jpg"), quality=92)
        saved += 1
print("size histogram (top):", sizes.most_common(8))
print("saved", saved)
