import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("strong_results.csv")
df = df.astype({"M":"int64","N":"int64","K":"int64","P":"int64","T":"float64"})

series = []
for (M,N,K), g in df.groupby(["M","N","K"]):
    g = g.sort_values("P")
    base = g[g["P"]==1]
    if base.empty: 
        continue
    T1 = float(base["T"].iloc[0])
    series.append({
        "label": f"STRONG {M}x{N}x{K}",
        "P": g["P"].to_numpy(),
        "T": g["T"].to_numpy(),
        "S": (T1/g["T"]).to_numpy()
    })

plt.figure(figsize=(8,6))
for s in series:
    plt.plot(s["P"], s["T"], "-o", label=s["label"])
plt.xlabel("Threads (P)")
plt.ylabel("Time (s)")
plt.title("Time vs Threads — STRONG")
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend()
plt.savefig("strong_time.png", dpi=180, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8,6))
if len(df):
    Pmax = int(df["P"].max())
    x = np.arange(1, Pmax+1)
    plt.plot(x, x, ":", label="ideal S=P")
for s in series:
    plt.plot(s["P"], s["S"], "-o", label=s["label"])
plt.xlabel("Threads (P)")
plt.ylabel("Speedup S = T1 / TP")
plt.title("Speedup vs Threads — STRONG")
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend()
plt.savefig("strong_speedup.png", dpi=180, bbox_inches="tight")
plt.close()
