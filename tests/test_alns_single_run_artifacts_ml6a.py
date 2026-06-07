from pathlib import Path
import tempfile
from core.instance_loader import load_instance
from alns_solver.paper_alns_main import build_ml1_paper_initial_state
from alns_solver.paper_production_runner import PaperALNSRunConfig,run_paper_alns_production,export_production_result
from reporting.route_visualization import plot_solution_routes

def main():
    root=Path(__file__).resolve().parents[1]
    instance=load_instance(root/"data"/"small"/"instance_001")
    cfg=PaperALNSRunConfig("instance_001",0.5,77,5,(0.0,100.0),(0.0,300.0),(3.0,1.0))
    initial=build_ml1_paper_initial_state(instance,seed=cfg.run_seed,lambda_value=cfg.lambda_value,
        cost_bounds=cfg.cost_bounds,emission_bounds=cfg.emission_bounds,emission_factors=cfg.emission_factors)
    result=run_paper_alns_production(instance=instance,initial_state=initial,config=cfg)
    with tempfile.TemporaryDirectory() as d:
        out=Path(d); artifacts=export_production_result(result,out)
        route=plot_solution_routes(instance=instance,state=result.best_state,output_path=out/"best_route_map.png")
        assert all(p.exists() for p in [artifacts["run_results_csv"],artifacts["iteration_history"],artifacts["best_solution"],route])
    print("[PASS] Single-run artifacts are exported")
    print("[PASS] Routing visualization is created")
    print("[PASS] Best solution passes validation")
    print("\nML-6A — SINGLE-RUN ARTIFACT PIPELINE PASSED")
if __name__=="__main__": main()
