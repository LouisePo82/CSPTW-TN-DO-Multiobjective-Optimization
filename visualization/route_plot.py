from pathlib import Path
import matplotlib.pyplot as plt

def plot_routes(instance, solution, output_path: str | Path, title: str):
    nodes = instance["nodes"]
    plt.figure(figsize=(10, 8))

    for nid, n in nodes.items():
        marker = "o"
        if n["type"].startswith("depot"): marker = "s"
        elif n["type"] == "tn": marker = "^"
        elif n["type"] == "adp": marker = "P"
        elif n["type"] == "od_origin": marker = "<"
        elif n["type"] == "od_destination": marker = ">"
        plt.scatter(n["x"], n["y"], marker=marker)
        plt.annotate(nid, (n["x"], n["y"]))

    for vehicle, route in solution.dv_routes.items():
        if route:
            xs = [nodes[i]["x"] for i in route]
            ys = [nodes[i]["y"] for i in route]
            plt.plot(xs, ys, marker="o", label=f"{vehicle} DV")
    for driver, route in solution.od_routes.items():
        if route:
            xs = [nodes[i]["x"] for i in route]
            ys = [nodes[i]["y"] for i in route]
            plt.plot(xs, ys, marker="o", linestyle="--", label=f"{driver} OD")

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
