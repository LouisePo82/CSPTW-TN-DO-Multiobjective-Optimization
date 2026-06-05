from tests.manual_reference import enumerate_micro_reference
from reporting import OutputManager
from .solver_factory import create_solver

def run_micro_validation(config, instance):
    solver = create_solver("exact")
    sol = solver.solve(
        instance,
        {
            "mode": "cost", "lambda_value": 0.0,
            "emission_factor_dv": 1.0, "emission_factor_od": 1.0,
        },
        vars(config.exact),
    )
    manual = enumerate_micro_reference(instance)
    diff = None if sol.cost is None else sol.cost - manual["best"]["cost"]
    if sol.status != "OPTIMAL" or not sol.validator_pass or abs(diff) > 1e-5:
        raise RuntimeError(f"Micro validation failed: status={sol.status}, diff={diff}, errors={sol.validation_errors}")
    out = OutputManager(config.output.root_dir, config.experiment_name)
    out.save_config(config.raw)
    out.snapshot_instance(config.instance_path)
    out.save_solution("micro_exact", sol)
    out.save_manifest({"manual_reference": manual, "exact_solution": sol.to_dict(), "difference": diff})
    return out.run_dir
