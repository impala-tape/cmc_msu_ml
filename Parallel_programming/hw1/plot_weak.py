import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("weak_results.csv")
df = df.astype({"M":"int64","N":"int64","K":"int64","P":"int64","T":"float64","SERIES":"string"})

series = []
for name, g in df.groupby("SERIES"):
    g = g.sort_values("P")
    Pmin = int(g["P"].min())
    T0 = float(g.loc[g["P"]==Pmin, "T"].iloc[0])
    series.append({
        "label": name,
        "P": g["P"].to_numpy(),
        "T": g["T"].to_numpy(),
        "Sg": g["P"].to_numpy() * (T0 / g["T"].to_numpy())
    })

plt.figure(figsize=(8,6))
for s in series:
    plt.plot(s["P"], s["T"], "-o", label=s["label"])
plt.xlabel("Threads (P)")
plt.ylabel("Time (s)")
plt.title("Time vs Threads — WEAK (constant work per thread)")
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend()
plt.savefig("weak_time.png", dpi=180, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8,6))
if len(df):
    Pmax = int(df["P"].max())
    x = np.arange(1, Pmax+1)
    plt.plot(x, x, ":", label="ideal S=P")
for s in series:
    plt.plot(s["P"], s["Sg"], "-o", label=s["label"])
plt.xlabel("Threads (P)")
plt.ylabel("Scaled speedup S = P · T0 / TP")
plt.title("Speedup vs Threads — WEAK (Gustafson)")
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend()
plt.savefig("weak_speedup.png", dpi=180, bbox_inches="tight")
plt.close()
