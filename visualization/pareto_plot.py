from pathlib import Path
import matplotlib.pyplot as plt

def plot_tradeoff(rows, output_path: str | Path, title="Exact Cost-Emission Trade-off"):
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["cost"])
    plt.figure(figsize=(8, 6))
    plt.plot([r["cost"] for r in rows], [r["emission"] for r in rows], marker="o")
    for r in rows:
        label = r.get("label", "")
        plt.annotate(label, (r["cost"], r["emission"]))
    plt.xlabel("Distribution Cost")
    plt.ylabel("Emission Proxy")
    plt.title(title)
    plt.grid(True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
