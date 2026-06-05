from __future__ import annotations
from pathlib import Path
from datetime import datetime
import csv, json, shutil, yaml

class OutputManager:
    def __init__(self, root: str, experiment_name: str):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(root) / f"{stamp}_{experiment_name}"
        for d in ["summary", "solutions", "charts", "logs", "instance_snapshot"]:
            (self.run_dir / d).mkdir(parents=True, exist_ok=True)

    def save_manifest(self, payload: dict):
        (self.run_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def save_config(self, raw: dict):
        (self.run_dir / "config_used.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    def snapshot_instance(self, instance_path: str):
        source = Path(instance_path)
        dest = self.run_dir / "instance_snapshot"
        for f in source.iterdir():
            if f.is_file():
                shutil.copy2(f, dest / f.name)

    def save_solution(self, label: str, solution):
        folder = self.run_dir / "solutions" / label
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "solution.json").write_text(
            json.dumps(solution.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        self._save_routes(folder, solution)
        self._save_assignments(folder, solution)

    def _save_routes(self, folder, solution):
        with (folder / "routes.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["vehicle", "vehicle_type", "sequence"])
            for k, r in solution.dv_routes.items():
                writer.writerow([k, "DV", " -> ".join(r)])
            for k, r in solution.od_routes.items():
                writer.writerow([k, "OD", " -> ".join(r)])

    def _save_assignments(self, folder, solution):
        with (folder / "assignments.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["customer", "mode", "vehicle_or_driver", "pickup_or_adp"])
            for c, a in solution.assignments.items():
                writer.writerow([
                    c, a.get("mode"),
                    a.get("vehicle", a.get("driver")),
                    a.get("pickup", a.get("adp")),
                ])

    def save_rows(self, relative_path: str, rows: list[dict]):
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
