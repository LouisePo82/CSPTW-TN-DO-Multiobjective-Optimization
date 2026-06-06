from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from core.objective import recompute_objectives
from core.schedule import recompute_earliest_schedule
from core.solution import Solution
from core.validator import validate_solution


@dataclass
class ALNSSolutionState:
    """
    Mutable solution representation used inside the ALNS search.

    This class deliberately stores only routing and assignment decisions.
    Cost, emission, schedules, loads, and feasibility are recomputed through
    the shared `core` layer so that ALNS and exact MILP use one source of truth.
    """

    dv_routes: dict[str, list[str]] = field(default_factory=dict)
    od_routes: dict[str, list[str]] = field(default_factory=dict)
    assignments: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Search-specific information.
    unassigned_customers: set[str] = field(default_factory=set)
    removed_customers: list[str] = field(default_factory=list)
    operator_history: list[dict[str, Any]] = field(default_factory=list)

    # Cached evaluation values. These are invalidated after mutation.
    cached_cost: float | None = None
    cached_emission: float | None = None
    cached_objective: float | None = None
    cached_dv_distance: float | None = None
    cached_od_extra_distance: float | None = None
    cached_validator_pass: bool | None = None
    cached_validation_errors: list[str] = field(default_factory=list)

    def copy(self) -> "ALNSSolutionState":
        """Return a deep copy suitable for destroy/repair operations."""
        return deepcopy(self)

    def invalidate_cache(self) -> None:
        """Invalidate all derived values after any route/assignment change."""
        self.cached_cost = None
        self.cached_emission = None
        self.cached_objective = None
        self.cached_dv_distance = None
        self.cached_od_extra_distance = None
        self.cached_validator_pass = None
        self.cached_validation_errors = []

    def normalize_routes(self, instance: dict) -> None:
        """
        Ensure all vehicles exist in the state and remove accidental
        consecutive duplicate nodes.

        Unused vehicles are represented by an empty list, not S -> T.
        """
        for vehicle in instance["dvs"]:
            route = list(self.dv_routes.get(vehicle, []))
            self.dv_routes[vehicle] = self._remove_consecutive_duplicates(route)

        for driver in instance["ods"]:
            route = list(self.od_routes.get(driver, []))
            self.od_routes[driver] = self._remove_consecutive_duplicates(route)

        self.invalidate_cache()

    @staticmethod
    def _remove_consecutive_duplicates(route: list[str]) -> list[str]:
        if not route:
            return []
        cleaned = [route[0]]
        for node in route[1:]:
            if node != cleaned[-1]:
                cleaned.append(node)
        return cleaned

    def mark_customer_unassigned(self, customer_id: str) -> None:
        """
        Remove the assignment record and mark the customer for repair.

        Route-node removal is operator-specific and therefore is not performed
        automatically here.
        """
        self.assignments.pop(customer_id, None)
        self.unassigned_customers.add(customer_id)
        if customer_id not in self.removed_customers:
            self.removed_customers.append(customer_id)
        self.invalidate_cache()

    def assign_customer(
        self,
        customer_id: str,
        assignment: dict[str, Any],
    ) -> None:
        """Set or replace one customer assignment."""
        self.assignments[customer_id] = deepcopy(assignment)
        self.unassigned_customers.discard(customer_id)
        self.invalidate_cache()

    def register_operator_event(
        self,
        operator_type: str,
        operator_name: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.operator_history.append(
            {
                "operator_type": operator_type,
                "operator_name": operator_name,
                "details": deepcopy(details or {}),
            }
        )

    def basic_structure_errors(self, instance: dict) -> list[str]:
        """
        Fast structural checks used before expensive shared validation.

        These checks do not replace `core.validator.validate_solution`.
        """
        errors: list[str] = []
        expected_customers = set(instance["customers"])
        assigned_customers = set(self.assignments)

        unknown = assigned_customers - expected_customers
        if unknown:
            errors.append(f"Unknown assigned customers: {sorted(unknown)}")

        duplicated_dv_visits: dict[str, list[str]] = {}
        for vehicle, route in self.dv_routes.items():
            customer_visits = [n for n in route if n in expected_customers]
            duplicates = sorted({n for n in customer_visits if customer_visits.count(n) > 1})
            if duplicates:
                duplicated_dv_visits[vehicle] = duplicates

        duplicated_od_visits: dict[str, list[str]] = {}
        for driver, route in self.od_routes.items():
            customer_visits = [n for n in route if n in expected_customers]
            duplicates = sorted({n for n in customer_visits if customer_visits.count(n) > 1})
            if duplicates:
                duplicated_od_visits[driver] = duplicates

        if duplicated_dv_visits:
            errors.append(f"Duplicate customer visits in DV routes: {duplicated_dv_visits}")
        if duplicated_od_visits:
            errors.append(f"Duplicate customer visits in OD routes: {duplicated_od_visits}")

        overlap = assigned_customers & self.unassigned_customers
        if overlap:
            errors.append(
                f"Customers cannot be both assigned and unassigned: {sorted(overlap)}"
            )

        return errors

    def evaluate(
        self,
        instance: dict,
        lambda_value: float,
        cost_bounds: tuple[float, float] | None,
        emission_bounds: tuple[float, float] | None,
        emission_factors: tuple[float, float] = (1.0, 1.0),
    ) -> dict[str, float]:
        """
        Recompute cost, emission, and weighted objective through shared core.

        During destroy/repair, a partial state can be structurally incomplete.
        This method only evaluates route distances; full feasibility is checked
        when converting to `core.solution.Solution`.
        """
        metrics = recompute_objectives(
            instance,
            self.dv_routes,
            self.od_routes,
            emission_factors=emission_factors,
        )

        cost = metrics["cost"]
        emission = metrics["emission"]

        if cost_bounds is None or emission_bounds is None:
            objective = (1.0 - lambda_value) * cost + lambda_value * emission
        else:
            z_min, z_max = cost_bounds
            e_min, e_max = emission_bounds

            if z_max <= z_min:
                raise ValueError("Cost normalization range must be positive.")
            if e_max <= e_min:
                raise ValueError("Emission normalization range must be positive.")

            normalized_cost = (cost - z_min) / (z_max - z_min)
            normalized_emission = (emission - e_min) / (e_max - e_min)
            objective = (
                (1.0 - lambda_value) * normalized_cost
                + lambda_value * normalized_emission
            )

        self.cached_cost = cost
        self.cached_emission = emission
        self.cached_objective = objective
        self.cached_dv_distance = metrics["dv_distance"]
        self.cached_od_extra_distance = metrics["od_extra_distance"]

        return {
            **metrics,
            "objective": objective,
        }

    def to_core_solution(
        self,
        instance: dict,
        lambda_value: float,
        objective_mode: str = "weighted",
        cost_bounds: tuple[float, float] | None = None,
        emission_bounds: tuple[float, float] | None = None,
        emission_factors: tuple[float, float] = (1.0, 1.0),
        runtime_sec: float = 0.0,
        metadata: dict[str, Any] | None = None,
        require_complete: bool = True,
    ) -> Solution:
        """
        Convert the mutable ALNS state into the shared immutable result schema.

        The shared validator expects the same raw result structure produced by
        the exact solver, so this method creates that adapter explicitly.
        """
        self.normalize_routes(instance)

        structure_errors = self.basic_structure_errors(instance)

        if require_complete:
            missing = set(instance["customers"]) - set(self.assignments)
            if missing:
                structure_errors.append(
                    f"Missing customer assignments: {sorted(missing)}"
                )
            if self.unassigned_customers:
                structure_errors.append(
                    f"Unassigned customers remain: "
                    f"{sorted(self.unassigned_customers)}"
                )

        metrics = self.evaluate(
            instance=instance,
            lambda_value=lambda_value,
            cost_bounds=cost_bounds,
            emission_bounds=emission_bounds,
            emission_factors=emission_factors,
        )

        # Build a provisional raw result for shared schedule/validator modules.
        provisional = {
            "status": "FEASIBLE" if not structure_errors else "INVALID",
            "cost": metrics["cost"],
            "emission": metrics["emission"],
            "objective": metrics["objective"],
            "dv_distance": metrics["dv_distance"],
            "od_extra_distance": metrics["od_extra_distance"],
            "dv_routes": deepcopy(self.dv_routes),
            "od_routes": deepcopy(self.od_routes),
            "routes": {**deepcopy(self.dv_routes), **deepcopy(self.od_routes)},
            "assignments": deepcopy(self.assignments),
            "times": {
                "dv": {},
                "od_customer": {},
                "od_pickup": {},
            },
            "tn_demand": self._compute_tn_demands(instance),
            "adp_load": self._compute_adp_loads(instance),
            "vehicle_load": self._compute_vehicle_loads(instance),
        }

        # Earliest schedules are deterministic and meaningful for ALNS output.
        earliest = recompute_earliest_schedule(instance, provisional)
        provisional["times"] = {
            "dv": earliest["dv"],
            "od_customer": earliest["od_customer"],
            "od_pickup": earliest["od_pickup"],
        }

        if structure_errors:
            validation = {
                "valid": False,
                "errors": structure_errors,
                "warnings": [],
            }
        else:
            validation = validate_solution(
                instance,
                provisional,
                emission_factors=emission_factors,
            )

        self.cached_validator_pass = validation["valid"]
        self.cached_validation_errors = list(validation["errors"])

        return Solution(
            status="FEASIBLE" if validation["valid"] else "INVALID",
            solver_name="alns",
            objective_mode=objective_mode,
            lambda_value=lambda_value,
            cost=metrics["cost"],
            emission=metrics["emission"],
            objective=metrics["objective"],
            dv_distance=metrics["dv_distance"],
            od_extra_distance=metrics["od_extra_distance"],
            dv_routes=deepcopy(self.dv_routes),
            od_routes=deepcopy(self.od_routes),
            assignments=deepcopy(self.assignments),
            arrival_times=earliest,
            vehicle_loads=provisional["vehicle_load"],
            tn_demands=provisional["tn_demand"],
            adp_loads=provisional["adp_load"],
            runtime_sec=runtime_sec,
            optimality_gap=None,
            validator_pass=validation["valid"],
            validation_errors=list(validation["errors"]),
            metadata={
                "operator_history": deepcopy(self.operator_history),
                "removed_customers": list(self.removed_customers),
                **(metadata or {}),
            },
        )

    def _compute_tn_demands(self, instance: dict) -> dict[str, float]:
        demand = {p: 0.0 for p in instance["pickup_points"]}
        for customer, assignment in self.assignments.items():
            if assignment.get("mode") != "OD_HOME":
                continue
            pickup = assignment.get("pickup")
            if pickup in demand:
                demand[pickup] += instance["nodes"][customer]["demand"]
        return demand

    def _compute_adp_loads(self, instance: dict) -> dict[str, dict[str, float]]:
        loads = {
            vehicle: {adp: 0.0 for adp in instance["adps"]}
            for vehicle in instance["dvs"]
        }
        for customer, assignment in self.assignments.items():
            if assignment.get("mode") != "ADP":
                continue
            vehicle = assignment.get("vehicle")
            adp = assignment.get("adp")
            if vehicle in loads and adp in loads[vehicle]:
                loads[vehicle][adp] += instance["nodes"][customer]["demand"]
        return loads

    def _compute_vehicle_loads(self, instance: dict) -> dict[str, float]:
        loads = {vehicle: 0.0 for vehicle in instance["dvs"]}

        # Direct-home and ADP demand.
        for customer, assignment in self.assignments.items():
            mode = assignment.get("mode")
            vehicle = assignment.get("vehicle")
            if mode in {"DV_HOME", "ADP"} and vehicle in loads:
                loads[vehicle] += instance["nodes"][customer]["demand"]

        # TN demand belongs to the DV route that visits the selected TN.
        for customer, assignment in self.assignments.items():
            if assignment.get("mode") != "OD_HOME":
                continue
            pickup = assignment.get("pickup")
            if pickup not in instance["tns"]:
                continue

            visiting_dvs = [
                vehicle
                for vehicle, route in self.dv_routes.items()
                if pickup in route
            ]
            if len(visiting_dvs) == 1:
                loads[visiting_dvs[0]] += instance["nodes"][customer]["demand"]

        return loads
