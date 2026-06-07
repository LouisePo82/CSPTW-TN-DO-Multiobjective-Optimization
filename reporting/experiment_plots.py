from __future__ import annotations
from pathlib import Path
import csv, statistics
import matplotlib.pyplot as plt

def _read(path):
    with path.open("r",encoding="utf-8") as f: return list(csv.DictReader(f))

def create_lambda_summary(consolidated_csv:Path,output_csv:Path):
    rows=_read(consolidated_csv); grouped={}
    for r in rows: grouped.setdefault(float(r["lambda"]),[]).append(r)
    out=[]
    for lam in sorted(grouped):
        g=grouped[lam]
        costs=[float(r["best_cost"]) for r in g]
        emissions=[float(r["best_emission"]) for r in g]
        objs=[float(r["best_F_lambda"]) for r in g]
        runtimes=[float(r["runtime_seconds"]) for r in g]
        valid=[str(r["validation_pass"]).lower()=="true" for r in g]
        out.append({
            "lambda":lam,"n_runs":len(g),
            "mean_cost":statistics.fmean(costs),
            "std_cost":statistics.stdev(costs) if len(costs)>1 else 0.0,
            "min_cost":min(costs),
            "mean_emission":statistics.fmean(emissions),
            "std_emission":statistics.stdev(emissions) if len(emissions)>1 else 0.0,
            "min_emission":min(emissions),
            "mean_F_lambda":statistics.fmean(objs),
            "mean_runtime_seconds":statistics.fmean(runtimes),
            "valid_run_rate":sum(valid)/len(valid),
        })
    output_csv.parent.mkdir(parents=True,exist_ok=True)
    with output_csv.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    return out

def plot_cost_emission_tradeoff(rows,output_path:Path):
    fig,ax=plt.subplots(figsize=(8,6))
    xs=[float(r["mean_cost"]) for r in rows]; ys=[float(r["mean_emission"]) for r in rows]
    ax.scatter(xs,ys)
    for r,x,y in zip(rows,xs,ys):
        ax.annotate(f"λ={float(r['lambda']):g}",(x,y),xytext=(5,5),textcoords="offset points")
    ax.set_title("Cost–emission trade-off"); ax.set_xlabel("Mean total cost"); ax.set_ylabel("Mean environmental impact")
    ax.grid(True,alpha=.25); fig.tight_layout(); output_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output_path,dpi=220,bbox_inches="tight"); plt.close(fig); return output_path

def plot_metric_by_lambda(rows,metric,output_path:Path,ylabel):
    ordered=sorted(rows,key=lambda r:float(r["lambda"]))
    xs=[float(r["lambda"]) for r in ordered]; ys=[float(r[metric]) for r in ordered]
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot(xs,ys,marker="o")
    ax.set_title(f"{ylabel} by λ"); ax.set_xlabel("λ"); ax.set_ylabel(ylabel); ax.grid(True,alpha=.25)
    fig.tight_layout(); output_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output_path,dpi=220,bbox_inches="tight"); plt.close(fig); return output_path
