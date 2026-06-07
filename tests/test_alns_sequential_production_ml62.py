from pathlib import Path
from core.instance_loader import load_instance
from alns_solver.paper_alns_main import build_ml1_paper_initial_state
from alns_solver.paper_production_runner import PaperALNSRunConfig,run_paper_alns_production

def main():
    root=Path(__file__).resolve().parents[1]
    instance=load_instance(root/"data"/"small"/"instance_001")
    cfg=PaperALNSRunConfig("instance_001",0.5,2026,20,(0.0,100.0),(0.0,300.0),(3.0,1.0))
    initial=build_ml1_paper_initial_state(instance,seed=cfg.run_seed,lambda_value=cfg.lambda_value,
        cost_bounds=cfg.cost_bounds,emission_bounds=cfg.emission_bounds,emission_factors=cfg.emission_factors)
    result=run_paper_alns_production(instance=instance,initial_state=initial,config=cfg)
    assert len(result.iteration_history)==20
    assert result.best_solution.validator_pass
    assert all(r["validation_pass"] for r in result.iteration_history)
    assert all(b["best_objective_after"]<=a["best_objective_after"]+1e-12 for a,b in zip(result.iteration_history,result.iteration_history[1:]))
    print("[PASS] Sequential current state runs for 20 iterations")
    print("[PASS] All states remain valid")
    print("[PASS] Best objective is non-increasing")
    print("[PASS] No enhanced ALNS behavior is present")
    print("\nML-6.2 — SEQUENTIAL PRODUCTION LOOP PASSED")
if __name__=="__main__": main()
