from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
import csv
from reporting.experiment_plots import create_lambda_summary,plot_cost_emission_tradeoff,plot_metric_by_lambda

def main():
    p=ArgumentParser(); p.add_argument("--output-root",default="outputs/production_runs"); a=p.parse_args()
    root=Path(a.output_root); files=sorted(root.rglob("run_results.csv"))
    if not files: raise SystemExit(f"No run_results.csv found under {root}")
    rows=[]
    for path in files:
        with path.open("r",encoding="utf-8") as f: row=next(csv.DictReader(f))
        row["source_run_results"]=str(path); row["run_timestamp_folder"]=path.parent.name; rows.append(row)
    consolidated=root/"benchmark_outputs_consolidate.csv"
    with consolidated.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary=create_lambda_summary(consolidated,root/"lambda_summary.csv")
    plot_cost_emission_tradeoff(summary,root/"cost_emission_tradeoff.png")
    plot_metric_by_lambda(summary,"mean_cost",root/"mean_cost_by_lambda.png","Mean total cost")
    plot_metric_by_lambda(summary,"mean_emission",root/"mean_emission_by_lambda.png","Mean emission proxy")
    plot_metric_by_lambda(summary,"mean_runtime_seconds",root/"mean_runtime_by_lambda.png","Mean runtime (seconds)")
    print(f"Consolidated: {consolidated}"); print(f"Summary: {root/'lambda_summary.csv'}")

if __name__=="__main__": main()
