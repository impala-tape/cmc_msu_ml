def check(x: str, file: str):
    counts = {}
    for w in x.split():
        w = w.lower()
        counts[w] = counts.get(w, 0) + 1
    lines = [f"{w} {counts[w]}" for w in sorted(counts)]
    with open(file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")