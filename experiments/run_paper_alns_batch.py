from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
import csv, subprocess, sys

def parse_args():
    p=ArgumentParser()
    p.add_argument("--instance",required=True); p.add_argument("--instance-id",required=True)
    p.add_argument("--lambdas",default="0,0.2,0.4,0.6,0.8,1.0"); p.add_argument("--seeds",required=True)
    p.add_argument("--iterations",type=int,required=True)
    p.add_argument("--cost-min",type=float,required=True); p.add_argument("--cost-max",type=float,required=True)
    p.add_argument("--emission-min",type=float,required=True); p.add_argument("--emission-max",type=float,required=True)
    p.add_argument("--dv-emission-factor",type=float,default=1.0); p.add_argument("--od-emission-factor",type=float,default=1.0)
    p.add_argument("--output-root",default="outputs/production_runs")
    return p.parse_args()

def main():
    a=parse_args(); lambdas=[float(x) for x in a.lambdas.split(",") if x.strip()]; seeds=[int(x) for x in a.seeds.split(",") if x.strip()]
    failures=[]; total=len(lambdas)*len(seeds); done=0
    for lam in lambdas:
        for seed in seeds:
            done+=1; print(f"[{done}/{total}] lambda={lam:g}, seed={seed}")
            cmd=[sys.executable,"-m","experiments.run_paper_alns_once",
                 "--instance",a.instance,"--instance-id",a.instance_id,
                 "--lambda-value",str(lam),"--seed",str(seed),"--iterations",str(a.iterations),
                 "--cost-min",str(a.cost_min),"--cost-max",str(a.cost_max),
                 "--emission-min",str(a.emission_min),"--emission-max",str(a.emission_max),
                 "--dv-emission-factor",str(a.dv_emission_factor),"--od-emission-factor",str(a.od_emission_factor),
                 "--output-root",a.output_root]
            r=subprocess.run(cmd)
            if r.returncode: failures.append({"lambda":lam,"seed":seed,"returncode":r.returncode})
    if failures:
        path=Path(a.output_root)/"batch_failures.csv"; path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=["lambda","seed","returncode"]); w.writeheader(); w.writerows(failures)
        raise SystemExit(f"{len(failures)} runs failed. See {path}")
    print(f"All {total} runs completed successfully.")

if __name__=="__main__": main()
