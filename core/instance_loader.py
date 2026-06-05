from __future__ import annotations
from pathlib import Path
import json
import math
import pandas as pd

REQUIRED_FILES = [
    "nodes.csv", "vehicles.csv", "adp_compatibility.csv", "parameters.csv"
]

def load_instance(instance_path: str | Path) -> dict:
    folder = Path(instance_path)
    missing = [name for name in REQUIRED_FILES if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {folder}: {missing}")

    nodes_df = pd.read_csv(folder / "nodes.csv")
    vehicles_df = pd.read_csv(folder / "vehicles.csv")
    gamma_df = pd.read_csv(folder / "adp_compatibility.csv")
    params_df = pd.read_csv(folder / "parameters.csv")

    nodes = {}
    for _, row in nodes_df.iterrows():
        node_id = str(row["id"]).strip()
        nodes[node_id] = {
            "id": node_id,
            "type": str(row["type"]).strip().lower(),
            "customer_type": int(row["customer_type"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "demand": float(row["demand"]),
            "tw_start": float(row["tw_start"]),
            "tw_end": float(row["tw_end"]),
        }

    vehicles = {}
    for _, row in vehicles_df.iterrows():
        vid = str(row["id"]).strip()
        vehicles[vid] = {
            "id": vid,
            "type": str(row["type"]).strip().lower(),
            "capacity": int(row["capacity"]),
            "origin": str(row["origin"]).strip(),
            "destination": str(row["destination"]).strip(),
            "earliest": float(row["earliest"]),
            "latest": float(row["latest"]),
            "emission_factor": float(row["emission_factor"]),
        }

    gamma = {
        (str(r["customer_id"]).strip(), str(r["adp_id"]).strip()): int(r["gamma"])
        for _, r in gamma_df.iterrows()
    }
    params = {str(r["parameter"]): float(r["value"]) for _, r in params_df.iterrows()}

    ids = list(nodes)
    distance = {i: {} for i in ids}
    travel_time = {i: {} for i in ids}
    for i in ids:
        for j in ids:
            d = 0.0 if i == j else math.hypot(
                nodes[i]["x"] - nodes[j]["x"],
                nodes[i]["y"] - nodes[j]["y"],
            )
            distance[i][j] = d
            travel_time[i][j] = d

    start = [i for i, x in nodes.items() if x["type"] == "depot_start"]
    end = [i for i, x in nodes.items() if x["type"] == "depot_end"]
    if len(start) != 1 or len(end) != 1:
        raise ValueError("Exactly one depot_start and one depot_end are required.")

    customers = [i for i, x in nodes.items() if x["type"] == "customer"]
    instance = {
        "nodes": nodes,
        "vehicles": vehicles,
        "distance": distance,
        "travel_time": travel_time,
        "gamma": gamma,
        "rho": float(params["rho"]),
        "service_time_per_weight": float(params["service_time_per_weight"]),
        "big_m_time": float(params["big_m_time"]),
        "start_depot": start[0],
        "end_depot": end[0],
        "customers": customers,
        "type1": [i for i in customers if nodes[i]["customer_type"] == 1],
        "type2": [i for i in customers if nodes[i]["customer_type"] == 2],
        "type3": [i for i in customers if nodes[i]["customer_type"] == 3],
        "tns": [i for i, x in nodes.items() if x["type"] == "tn"],
        "adps": [i for i, x in nodes.items() if x["type"] == "adp"],
        "dvs": [v for v, x in vehicles.items() if x["type"] == "dv"],
        "ods": [v for v, x in vehicles.items() if x["type"] == "od"],
        "pickup_points": [start[0]] + [i for i, x in nodes.items() if x["type"] == "tn"],
        "instance_path": str(folder),
    }
    metadata_path = folder / "metadata.json"
    instance["metadata"] = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return instance
