# An Emission-Aware Extension of the Crowd-Shipping Problem with Time Windows, Transshipment Nodes, and Delivery Options

This repository contains the source code, project documents, retained experimental evidence, and reproducibility materials developed for the course **Optimization Models and Applications**.

The project studies the **Crowd-Shipping Problem with Time Windows, Transshipment Nodes, and Delivery Options (CSPTW-TN-DO)**. The system integrates dedicated vehicles, occasional drivers, customer time windows, alternative delivery points, transshipment nodes, vehicle capacities, customer delivery preferences, and synchronized parcel handoffs.

The work was developed through two connected course stages:

1. a paper presentation that analyzed the original CSPTW-TN-DO study and proposed an emission-aware extension; and
2. a final optimization project that reconstructed the paper-based solution framework and evaluated the proposed extension computationally.

The original reference problem minimizes total distribution cost. This project extends the objective layer by adding estimated operational emissions from dedicated-vehicle routes and occasional-driver detours.

The final implementation includes:

- a paper-based CSPTW-TN-DO model reconstruction;
- an exact mixed-integer optimization formulation;
- a paper-based Adaptive Large Neighborhood Search reconstruction;
- dedicated-vehicle operational-emission calculations;
- occasional-driver incremental-emission calculations;
- empirical objective normalization;
- normalized weighted scalarization;
- numerical safeguards for normalized objectives;
- shared feasibility and consistency validation;
- transformed C101-derived and R101-derived benchmark instances;
- controlled multi-seed ALNS experiments;
- route, convergence, and statistical visualizations;
- automated output consolidation;
- configuration and artifact management;
- reproducibility packages for Windows execution.

Project documents:

- [Original paper presentation](docs/OMA_Group3_PaperPresentation.pdf)
- [Final project report](docs/OMA_Group3_FinalProject_Report.pdf)
- [Final project presentation](docs/OMA_Group3_FinalProject_Presentation.pdf)

The submitted report and presentation files are retained unchanged as course-project evidence.

---

## Course context

**Course:** Optimization Models and Applications  
**Project type:** Paper presentation and final optimization project  
**Institution:** National Taiwan University of Science and Technology  

The course work developed progressively from academic paper analysis to optimization-model implementation.

The overall requirements included:

- selecting a peer-reviewed optimization paper;
- explaining the practical problem and its importance;
- identifying the research gap and paper contribution;
- describing the optimization model and major decisions;
- reviewing the exact and heuristic solution methods;
- analyzing the computational results;
- critically evaluating the paper;
- proposing a meaningful model or method extension;
- defining a final optimization problem;
- implementing an appropriate solution method;
- designing benchmark instances and experiments;
- evaluating solution quality and feasibility;
- presenting results, limitations, and managerial implications;
- retaining sufficient evidence for reproducibility.

The final project was therefore developed as a direct continuation of the earlier paper presentation rather than as an unrelated implementation.

---

## Stage 1 — Paper selection and presentation

The selected reference paper was:

> Yu, V. F., Jodiawan, P., and Redi, A. A. N. P. (2022).  
> *Crowd-shipping problem with time windows, transshipment nodes, and delivery options*.  
> Transportation Research Part E: Logistics and Transportation Review, 157, 102545.  
> https://doi.org/10.1016/j.tre.2021.102545

The paper introduces the Crowd-Shipping Problem with Time Windows, Transshipment Nodes, and Delivery Options.

During the paper-presentation stage, the group:

- explained the last-mile delivery problem;
- described the role of dedicated vehicles and occasional drivers;
- distinguished transshipment nodes from alternative delivery points;
- explained the three customer delivery types;
- identified the main routing and assignment decisions;
- reviewed the original cost-minimization objective;
- grouped the mathematical constraints into operational categories;
- explained the exact formulation and linearization;
- reviewed the paper-based ALNS;
- examined the Solomon-derived benchmark design;
- analyzed the exact-solver and ALNS results;
- discussed managerial findings;
- identified assumptions and limitations;
- proposed adding environmental impact to the original cost-only objective.

The proposed improvement was summarized conceptually as:

```text
Original paper:
Minimize distribution cost

Proposed extension:
Minimize distribution cost and operational emissions
```

This proposal became the starting point of the final project.

---

## Stage 2 — Final optimization project

The final-project stage transformed the conceptual emission-aware proposal into an executable optimization framework.

The project work included:

- reconstructing the main CSPTW-TN-DO delivery structure;
- reconstructing the paper-based exact formulation;
- reconstructing the reported ALNS framework;
- implementing dedicated-vehicle operational emissions;
- implementing occasional-driver incremental emissions;
- defining empirical normalization anchors;
- combining cost and emissions through a normalized scalar objective;
- implementing a sign-safe local-search eligibility safeguard;
- transforming Solomon VRPTW data into controlled CSPTW-TN-DO instances;
- implementing shared feasibility validation;
- checking numerical consistency;
- configuring exact and ALNS experiments;
- executing matched experiments using multiple random seeds;
- comparing clustered and randomly distributed spatial structures;
- consolidating experimental outputs;
- generating route and statistical visualizations;
- preserving tagged production configurations and evidence packages.

The final project therefore represents the computational continuation of the improvement proposed during the paper presentation.

---

## Project development path

```text
Course paper-presentation requirements
        ↓
Selection of a peer-reviewed optimization paper
        ↓
Analysis of the original CSPTW-TN-DO model
        ↓
Review of the exact formulation and ALNS
        ↓
Critical evaluation of the cost-only objective
        ↓
Proposal of a cost–emission extension
        ↓
Final-project problem definition
        ↓
Paper-based exact and ALNS reconstruction
        ↓
Emission-model implementation
        ↓
Objective normalization and scalarization
        ↓
Benchmark transformation
        ↓
Fidelity, feasibility, and numerical checks
        ↓
Controlled exact and ALNS experiments
        ↓
Result analysis and visualization
        ↓
Final report, presentation, and reproducibility packages
```

This development path is important for interpreting the repository correctly.

The original routing problem, customer-service logic, exact formulation, and main ALNS structure come from the reference study.

The emission-aware objective, normalization procedure, numerical safeguard, shared validator, transformed benchmark design, multi-seed experiment, and reproducibility workflow are project-specific additions.

---

## Main project objective

The main objective is to investigate how routing and delivery decisions change when both distribution cost and operational emissions are considered in the CSPTW-TN-DO system.

The project addresses the following research questions:

1. Can the reconstructed framework generate feasible and internally consistent solutions?
2. How can cost and emissions be combined without allowing one raw numerical scale to dominate?
3. How do cost, emissions, and solution variability change under different environmental-priority weights?
4. Does a clustered customer structure behave differently from a randomly distributed structure?
5. What conclusions are supported by the exact-solver and ALNS evidence?
6. What claims remain outside the available computational evidence?

---

## Main project contribution

The contribution is organized into three levels.

| Contribution level | Project contribution |
| --- | --- |
| Modeling | Extension of the original cost-only CSPTW-TN-DO objective to a normalized cost–emission objective |
| Computational | Paper-based exact and ALNS reconstruction used to evaluate routing, cost, emissions, feasibility, and runtime |
| Reliability | Shared feasibility validation, numerical consistency checks, deterministic benchmark generation, matched seeds, frozen configurations, and reproducible output packages |

The project does not propose a new ALNS algorithm.

Its main contribution lies in applying an emission-aware objective to a paper-based CSPTW-TN-DO framework and supporting the computational study with explicit validation, controlled comparisons, and reproducibility evidence.

---

## Team members

### 1. Dong Boi Thi

Main responsibilities:

- reference-paper analysis;
- optimization-model interpretation;
- final-project formulation planning;
- exact-solver reconstruction;
- ALNS framework reconstruction;
- emission-aware objective development;
- objective-normalization implementation;
- scalarization implementation;
- numerical-safeguard implementation;
- feasibility-validator development;
- benchmark transformation;
- experiment configuration;
- experiment execution;
- output consolidation;
- result analysis;
- route and statistical visualization;
- report evidence preparation;
- repository structure;
- release and reproducibility preparation.

### 2. To Nguyen Tan Phuong

Main responsibilities:

- reference-paper analysis;
- methodological review;
- review of the connection between the reference paper and the proposed extension;
- result checking and verification;
- documentation review;
- report proofreading;
- presentation preparation;
- presentation proofreading.

---

## Relationship to the reference paper

The reference paper serves as both:

- the academic foundation of the initial paper presentation; and
- the methodological foundation of the final implementation project.

It provides the original CSPTW-TN-DO structure, including:

- dedicated vehicles;
- occasional drivers;
- customer time windows;
- transshipment nodes;
- alternative delivery points;
- multiple customer delivery preferences;
- the original cost objective;
- an exact mathematical formulation;
- linearization for exact solving;
- an ALNS solution approach.

This project does not claim to reproduce every computational experiment or every undocumented implementation detail from the paper.

The project-specific extension includes:

- dedicated-vehicle operational emissions;
- occasional-driver incremental emissions;
- empirical objective normalization;
- normalized weighted scalarization;
- exact and ALNS evaluation under five objective priorities;
- transformed C101-derived and R101-derived instances;
- shared feasibility validation;
- numerical consistency checks;
- multi-seed experimental design;
- artifact generation and output consolidation;
- Windows reproducibility packages.

The project uses a **paper-based reconstruction** of the reported solution framework.

It should not be interpreted as a source-code-level reproduction based on the original authors’ implementation.

---

## Original CSPTW-TN-DO system

The implemented framework includes the following entities.

### Depot

The depot is:

- the starting and ending location of dedicated vehicles;
- a possible parcel pick-up point for occasional drivers.

### Dedicated vehicles

Dedicated vehicles are company-operated vehicles that can:

- serve eligible customers directly;
- deliver parcels to alternative delivery points;
- deliver parcels to transshipment nodes;
- provide parcels for occasional-driver handoff.

### Occasional drivers

Occasional drivers are external drivers who already have an origin and destination.

They may accept delivery tasks when:

- the assigned detour is feasible;
- capacity is available;
- customer and driver time windows are satisfied;
- parcel pick-up is available at the depot or a transshipment node.

### Alternative delivery points

Alternative delivery points are locations where eligible customers collect their parcels.

They allow several customer deliveries to be consolidated into fewer physical delivery stops.

### Transshipment nodes

Transshipment nodes support parcel handoff between:

- a dedicated vehicle; and
- an occasional driver.

The dedicated vehicle must deliver the parcel to the transshipment node before the occasional driver collects it.

### Customers

Customers have:

- parcel demand;
- service times;
- time windows;
- delivery preferences;
- possible alternative delivery-point assignments.

---

## Customer delivery types

The CSPTW-TN-DO system uses three customer types.

| Customer type | Delivery requirement | Available service |
| --- | --- | --- |
| Type 1 | Home delivery only | Dedicated vehicle or occasional driver |
| Type 2 | ADP delivery only | Dedicated vehicle delivers to the assigned ADP |
| Type 3 | Home delivery or ADP | Dedicated vehicle, occasional driver, or assigned ADP |

Type 3 customers provide the greatest decision flexibility.

The model may select either:

- home delivery; or
- delivery through an assigned alternative delivery point.

The selected option must still satisfy routing, capacity, time-window, and synchronization requirements.

---

## Main decisions

The optimization framework makes several linked decisions:

1. which delivery option is selected for each customer;
2. whether a home-delivery customer is assigned to a dedicated vehicle or occasional driver;
3. which alternative delivery point is used for an eligible customer;
4. which parcel pick-up point is assigned to an occasional driver;
5. how dedicated-vehicle routes are sequenced;
6. how occasional-driver routes are sequenced;
7. how vehicle loads are managed;
8. how arrival and service times are coordinated;
9. whether each transshipment handoff is feasible.

These decisions are interdependent.

For example, assigning a customer to an occasional driver may require:

- a valid pick-up point;
- sufficient capacity;
- an acceptable additional detour;
- customer time-window feasibility;
- driver availability;
- a valid transshipment-node handoff time.

---

## Main constraint groups

The detailed formulation is organized into five operational constraint groups.

| Constraint group | Main control | Operational meaning |
| --- | --- | --- |
| Delivery-option constraints | Home delivery or ADP selection | Each customer receives one valid delivery option |
| Vehicle-assignment constraints | DV or OD assignment | Each home-delivery customer is served once |
| Dedicated-vehicle constraints | Route flow, capacity, load, and time | Each DV route remains feasible |
| Occasional-driver constraints | Pick-up point, route, capacity, and availability | Each OD route remains feasible |
| TN synchronization constraints | DV supply before OD collection | Parcels must reach the TN before collection |

A route may be short in distance but infeasible because:

- the vehicle exceeds capacity;
- a time window is violated;
- the occasional driver is unavailable;
- the parcel has not reached the transshipment node before collection.

---

## Original cost objective

The original paper minimizes total distribution cost.

Conceptually:

```text
Total distribution cost
=
Dedicated-vehicle travel cost
+
Occasional-driver compensation
```

Dedicated-vehicle cost is based on the selected route arcs.

Occasional-driver compensation is based on the additional distance created by the delivery task.

Only the additional occasional-driver distance is attributed to the delivery operation because the original personal trip already existed before the parcel assignment.

---

## Emission-aware extension

The final project retains the original delivery structure and feasibility conditions.

The main extension changes how a feasible solution is evaluated.

Instead of considering only distribution cost, the extended model considers:

1. distribution cost;
2. dedicated-vehicle operational emissions;
3. occasional-driver incremental emissions.

---

## Dedicated-vehicle operational emissions

Dedicated-vehicle emissions are based on the complete distance traveled for the delivery operation.

```text
DV emissions = DV route distance * DV emission factor
```

The complete dedicated-vehicle route is included because the route is performed specifically for the logistics operation.

---

## Occasional-driver incremental emissions

Occasional drivers already have an original trip from an origin to a destination.

Only the additional travel caused by the parcel assignment is attributed to the crowd-shipping operation.

```text
OD additional distance
=
max(0, OD delivery-route distance - OD original direct-trip distance)
```

Then:

```text
OD emissions = OD additional distance * OD emission factor
```

This avoids assigning the emissions of the original personal trip to the logistics operation.

---

## Total operational emissions

```text
Total emissions = DV emissions + OD emissions
```

The emission model uses comparative distance-based factors.

It does not represent a complete lifecycle carbon inventory.

---

## Why normalization is required

Distribution cost and operational emissions have:

- different units;
- different numerical ranges;
- different interpretations.

A direct raw objective such as:

```text
Cost + Emissions
```

could be dominated by whichever component has the larger numerical scale.

Normalization converts both objectives into dimensionless values before they are combined.

This allows the environmental weight to represent the relative emphasis on the two normalized measures rather than the accidental scale of the raw values.

---

## Multi-objective formulation

The production experiments use the following normalized scalar objective:

```text
F_lambda = (1 - lambda) * C_hat + lambda * E_hat
```

where:

- `C_hat` is the normalized distribution cost;
- `E_hat` is the normalized operational emissions;
- `lambda = 0` represents a cost-oriented objective;
- `lambda = 1` represents an emission-oriented objective.

The normalized cost is calculated as:

```text
C_hat = (C - C_ideal) / (C_nadir - C_ideal)
```

The normalized emissions are calculated as:

```text
E_hat = (E - E_ideal) / (E_nadir - E_ideal)
```

The tested lambda values are:

```text
0.00, 0.25, 0.50, 0.75, 1.00
```

Their interpretation is:

| Lambda | Main objective emphasis |
| ---: | --- |
| 0.00 | Cost-oriented |
| 0.25 | Higher emphasis on normalized cost |
| 0.50 | Equal weight on normalized cost and emissions |
| 0.75 | Higher emphasis on normalized emissions |
| 1.00 | Emission-oriented |

The setting `lambda = 0.50` gives equal weight to the normalized objectives.

It does not imply that one raw cost unit equals one raw emission unit.

---

## Empirical normalization anchors

The normalization values are empirical anchors obtained from validated calibration solutions.

They are not claimed to be globally proven theoretical ideal or nadir points.

Separate anchors are used for each benchmark instance because the C101-derived and R101-derived cases have different cost and emission scales.

| Instance | Cost ideal | Cost nadir | Emission ideal | Emission nadir |
| --- | ---: | ---: | ---: | ---: |
| C101-derived | 138.620394 | 140.106516 | 445.860718 | 450.764046 |
| R101-derived | 184.011763 | 326.629585 | 614.888095 | 1037.248717 |

The C101-derived anchors were generated from objective-specific exact-solver incumbents under a fixed time limit.

The R101-derived anchors were obtained from a feasible-solution calibration pool.

The anchors were frozen before the final production experiment.

---

## Normalized values outside zero and one

Because the normalization anchors are empirical, later solutions may produce:

- normalized values below zero; or
- normalized values above one.

This does not automatically indicate an invalid solution.

A negative normalized value means that the solution improved beyond the frozen empirical ideal reference.

A normalized value above one means that the solution fell outside the opposite-boundary reference range.

---

## Numerical safeguard for negative objectives

The reference ALNS applies local search when a working solution is sufficiently close to the best solution found so far.

A direct multiplicative threshold works normally when the objective is nonnegative.

However, empirical normalization can produce a negative scalar objective.

For a negative best objective, multiplying by a value greater than one makes the threshold more negative and may incorrectly reject even the current best solution.

The project therefore uses a sign-safe relative-gap condition.

This safeguard:

- is identical to the original rule for nonnegative objectives;
- modifies only the negative-objective case;
- preserves the intended local-search eligibility logic;
- is treated as numerical compatibility logic rather than an algorithmic enhancement.

---

## Optimization methods

The repository contains two main optimization paths:

1. an exact mixed-integer formulation;
2. a paper-based ALNS reconstruction.

---

## Exact MILP solver

The exact formulation was evaluated on the 25-customer C101-derived instance under a fixed computational budget.

It supports:

- cost-oriented anchor configurations;
- emission-oriented anchor configurations;
- normalized weighted-sum objectives;
- epsilon-constraint configurations;
- configurable time limits;
- configurable target MIP gaps;
- solver status reporting;
- lower-bound reporting;
- incumbent reporting;
- relative MIP-gap reporting;
- post-solution feasibility validation;
- route and visualization artifact export.

A feasible incumbent is not reported as a zero-gap optimum unless the solver provides sufficient certification.

A solver status such as `OPTIMAL` under a configured nonzero gap tolerance should not automatically be interpreted as a mathematical zero-gap result.

---

## Role of the exact solver

The exact formulation is used to:

- represent the detailed optimization model;
- generate empirical C101-derived normalization anchors;
- evaluate weighted objectives on a controlled instance;
- observe solver behavior under the fixed computational limit;
- provide small-instance formulation evidence.

It is not used to claim that all final ALNS results have certified optimality gaps.

---

## Paper-based ALNS reconstruction

The ALNS implementation includes:

- an initial feasible solution;
- paper-defined destroy-operator families;
- paper-defined repair logic;
- feasible insertion filtering;
- best insertion;
- perturbed best insertion;
- 2-regret insertion;
- perturbed 2-regret insertion;
- 3-regret insertion;
- perturbed 3-regret insertion;
- eight local-search operators;
- adaptive destroy-operator weights;
- adaptive repair-operator weights;
- simulated-annealing acceptance;
- deterministic random seeds;
- shared final feasibility validation;
- iteration histories;
- operator-selection statistics;
- route visualizations.

The production implementation does not use:

- enhanced search behavior;
- fallback repair;
- operator substitution;
- operator resampling.

---

## ALNS workflow

The main ALNS sequence is:

```text
Generate a complete feasible initial solution
        ↓
Select a destroy operator
        ↓
Remove part of the current solution
        ↓
Select a repair operator
        ↓
Reinsert the removed elements
        ↓
Check feasibility
        ↓
Apply local search when eligible
        ↓
Evaluate cost, emissions, and scalar objective
        ↓
Accept or reject using simulated annealing
        ↓
Update operator scores
        ↓
Update adaptive operator weights
        ↓
Continue until the stopping condition
        ↓
Validate and export the best solution
```

---

## Initial solution construction

The initial solution follows the three-stage logic reported in the reference study.

### Stage 1 — Occasional-driver insertion

Home-delivery customers are considered for insertion into occasional-driver routes.

If an unused occasional driver is selected:

- a depot or transshipment-node pick-up point is assigned;
- route feasibility is checked;
- customer and driver time windows are checked;
- driver capacity is checked.

### Stage 2 — Transshipment-node insertion

Transshipment nodes required by occasional-driver routes are inserted into dedicated-vehicle routes.

When a transshipment node cannot be supplied feasibly:

- the related occasional driver may switch to the depot; or
- affected customers may return to the unassigned-customer list.

### Stage 3 — Remaining-customer insertion

Remaining customers are inserted into feasible:

- dedicated-vehicle routes;
- occasional-driver routes;
- alternative delivery-point assignments.

The initial construction must produce a complete feasible solution before ALNS begins.

---

## Destroy operators

The destroy stage includes twelve selectable entries derived from the reported operator families.

These include removal logic related to:

- random customers;
- alternative delivery points;
- transshipment nodes;
- worst customers;
- worst alternative delivery points;
- complete routes;
- related customers;
- historical performance;
- spatial neighborhoods;
- node neighborhoods.

The purpose of the destroy stage is to remove part of the current solution and allow the search to reconstruct it differently.

---

## Repair operators

The repair stage contains six variants:

1. best insertion;
2. perturbed best insertion;
3. 2-regret insertion;
4. perturbed 2-regret insertion;
5. 3-regret insertion;
6. perturbed 3-regret insertion.

Best insertion selects the feasible position with the lowest incremental objective.

Regret insertion also considers the cost of delaying a customer’s insertion.

Customers with a high regret value receive higher priority because losing the best insertion opportunity may lead to a much worse alternative.

---

## Local search

The ALNS contains eight local-search entries based on move and swap operations.

The operators act:

- within dedicated-vehicle routes;
- between dedicated-vehicle routes;
- within occasional-driver routes;
- between route types where permitted.

Every accepted move must preserve:

- delivery-option feasibility;
- capacity feasibility;
- time-window feasibility;
- occasional-driver availability;
- transshipment-node synchronization.

---

## Adaptive operator weights

Destroy and repair operators initially receive equal weights.

Their selection probabilities are updated according to recent performance.

Operators receive scores when they:

- find a new global-best solution;
- improve the current solution;
- produce a worse but accepted solution.

The production parameter values are:

```text
Weight-update segment: 300 iterations
Reaction factor: 0.1
Operator scores: 33 / 15 / 9
```

After each full segment, operators with stronger recent performance receive higher future selection probabilities.

---

## Simulated-annealing acceptance

A working solution that improves the current solution is accepted directly.

A worse solution may also be accepted according to a simulated-annealing probability.

This mechanism helps the search:

- escape local optima;
- explore alternative regions;
- avoid becoming fully greedy too early.

The temperature decreases after each iteration using the production cooling rate:

```text
Cooling rate: 0.9994
```

As the temperature decreases, worse solutions become less likely to be accepted.

---

## Benchmark instances

The project uses two transformed 25-customer benchmark instances:

- **C101-derived:** clustered customer structure;
- **R101-derived:** randomly distributed customer structure.

The source customer data were taken from the Solomon VRPTW benchmark repository:

- customer coordinates;
- customer demand;
- service times;
- customer time windows.

Source repository:

https://www.sintef.no/projectweb/top/vrptw/100-customers/

---

## Transformation from Solomon VRPTW

The original Solomon data describe a classical Vehicle Routing Problem with Time Windows.

They do not contain:

- occasional drivers;
- occasional-driver origins;
- occasional-driver destinations;
- occasional-driver capacities;
- driver availability periods;
- transshipment nodes;
- alternative delivery points;
- customer delivery types;
- customer-to-ADP assignments;
- emission parameters.

These elements were added during the project instance-construction process.

The final cases are therefore described as:

```text
Solomon-derived CSPTW-TN-DO instances
```

They are not unchanged Solomon VRPTW instances.

Their objective values should not be compared directly with published Solomon VRPTW best-known solutions.

---

## Why C101 and R101 were selected

The two families represent different spatial structures.

### C101-derived

The C101-derived instance has a clustered customer structure.

It may provide stronger opportunities for:

- short local connections;
- route consolidation;
- alternative delivery-point consolidation;
- shorter occasional-driver detours.

### R101-derived

The R101-derived instance has a more randomly distributed customer structure.

It may create:

- greater route dispersion;
- more varied detours;
- weaker consolidation opportunities;
- higher stochastic variation.

Only one transformed instance from each family was tested.

The results therefore support a controlled instance comparison rather than a general conclusion about all clustered and random routing problems.

---

## Final matched ALNS experiment

| Setting | C101-derived | R101-derived |
| --- | ---: | ---: |
| Customers | 25 | 25 |
| Lambda values | 5 | 5 |
| Seeds | 2029–2040 | 2029–2040 |
| Seeds per lambda | 12 | 12 |
| Iterations per run | 301 | 301 |
| Runs per instance | 60 | 60 |
| Final validation | 100% pass | 100% pass |

The final matched experiment contains:

```text
2 instances
*
5 lambda values
*
12 seeds
=
120 ALNS runs
```

Both instance families used:

- the same algorithm source;
- the same lambda values;
- the same seed set;
- the same 301-iteration limit;
- the same hardware environment;
- separate output directories.

---

## Why 301 ALNS iterations were used

The reference study uses a much larger search budget.

The course project uses 301 iterations per run.

This budget allows each run to complete:

1. one full 300-iteration adaptive scoring segment;
2. one operator-weight update;
3. one additional iteration using the updated weights.

The 301-iteration budget is a controlled course-project setting.

It is not a reproduction of the paper’s full 30,000-iteration search depth.

---

## Exact-solver experiment

The exact formulation was evaluated on the C101-derived 25-customer instance.

The configuration included:

| Parameter | Value |
| --- | --- |
| Primary objective mode | Normalized weighted scalar objective |
| Lambda values | 0, 0.25, 0.50, 0.75, 1.00 |
| Weighted-objective packages | 2 |
| Weighted runs per package | 5 |
| Retained anchor runs | 2 |
| Time limit | 300 seconds per solve |
| Target relative MIP gap | 5% |
| Feasible incumbent required | Yes |
| Shared-validator acceptance required | Yes |
| Zero-gap optimality required | No |

A separate epsilon-constraint stage was attempted.

The first epsilon-constrained solve returned `NOT_SOLVED` within the fixed time limit and did not provide a retained feasible incumbent.

No epsilon-constraint result was included in the final analysis.

---

## Experiment inclusion rules

An ALNS run was included in the final analysis only when:

1. it belonged to the final instance–lambda–seed matrix;
2. it used the frozen production configuration;
3. it reached the required stopping condition;
4. the final feasibility validator passed;
5. required output files were generated;
6. no duplicate instance–lambda–seed combination was present;
7. reported objectives agreed with the independent evaluator.

Pilot outputs generated before the final feasibility alignment were excluded.

---

## Validation and reliability gates

The project applied several verification gates before the final experiment.

| Gate | What was checked | Final status |
| --- | --- | --- |
| Structural fidelity | Operator inventory, parameter values, and solution representation | Pass |
| Behavioral feasibility | Delivery options, capacity, time windows, and TN synchronization | Pass |
| Numerical consistency | Cost, emissions, normalization, and scalar objective | Pass |
| Reproducibility | Instance, seed, lambda, iteration count, and configuration | Pass |
| Production readiness | Output completeness, duplicates, and validator status | Pass |

These checks do not prove that every undocumented implementation detail is identical to the original authors’ implementation.

They show that the reported paper structure was reconstructed and tested against the computational contract used in this project.

---

## Feasibility-consistency correction

During pilot testing, an inconsistency was identified between:

- repair-stage capacity checking; and
- the final feasibility validator.

The issue concerned transshipment-node handoff demand when multiple dedicated-vehicle routes visited the same transfer node.

The repair-stage feasibility logic was aligned with the shared validator.

This correction:

- did not add a new operator;
- did not create an enhanced search mode;
- changed candidate feasibility interpretation;
- could affect stochastic search trajectories.

For this reason:

- mixed-provenance pilot outputs were excluded;
- C101-derived and R101-derived production experiments were rerun;
- both experiments used the same corrected source revision;
- the same seeds and iteration limits were applied.

---

## Final experiment completion

| Experiment | Planned | Completed | Included | Validation |
| --- | ---: | ---: | ---: | --- |
| Exact weighted package 1 | 5 | 5 | 5 | All retained incumbents passed |
| Exact weighted package 2 | 5 | 5 | 5 | All retained incumbents passed |
| Exact anchor solves | 2 | 2 | 2 | Both passed |
| Epsilon-constraint stage | 2 levels | 1 attempted | 0 | No incumbent retained |
| ALNS C101-derived | 60 | 60 | 60 | 100% pass |
| ALNS R101-derived | 60 | 60 | 60 | 100% pass |
| Total ALNS | 120 | 120 | 120 | 100% pass |

No final ALNS instance–lambda–seed combination was missing or duplicated.

---

## Main ALNS results

### C101-derived results

| Lambda | Runs | Mean cost | Mean emissions | Mean runtime (s) |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 12 | 143.573 | 461.267 | 149.2 |
| 0.25 | 12 | 141.990 | 456.464 | 149.3 |
| 0.50 | 12 | 141.355 | 454.591 | 147.8 |
| 0.75 | 12 | 141.894 | 454.200 | 168.9 |
| 1.00 | 12 | 142.534 | 454.768 | 152.3 |

For the C101-derived instance:

- every nonzero lambda produced lower mean cost than the lambda-zero baseline;
- every nonzero lambda produced lower mean emissions than the baseline;
- the lowest mean cost occurred at lambda 0.50;
- the lowest mean emissions occurred at lambda 0.75;
- the pattern was beneficial but not fully monotonic.

### R101-derived results

| Lambda | Runs | Mean cost | Mean emissions | Mean runtime (s) |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 12 | 190.157 | 631.999 | 148.8 |
| 0.25 | 12 | 190.251 | 626.375 | 143.9 |
| 0.50 | 12 | 189.744 | 628.133 | 143.7 |
| 0.75 | 12 | 193.015 | 634.044 | 150.7 |
| 1.00 | 12 | 189.890 | 625.983 | 150.9 |

For the R101-derived instance:

- the response was weaker and non-monotonic;
- lambda 1.00 produced the lowest mean emissions;
- lambda 0.50 produced a small mean cost reduction;
- lambda 0.75 increased both mean cost and mean emissions relative to the baseline.

---

## Improvement from the initial solution

| Instance | Mean cost improvement | Mean emission improvement |
| --- | --- | --- |
| C101-derived | Approximately 33.67%–34.69% | Approximately 31.38%–32.43% |
| R101-derived | Approximately 28.2%–29.4% | Approximately 27.3%–28.2% |

ALNS improved the initial feasible construction under every tested instance and lambda setting.

The larger improvement observed for C101-derived may reflect stronger consolidation opportunities in the clustered structure.

However, the experiment uses only one transformed instance from each family.

---

## Main interpretation

The results show that cost and emissions are partly aligned because both are strongly influenced by travel distance.

A shorter dedicated-vehicle route can reduce:

- travel cost;
- operational emissions.

A shorter occasional-driver detour can reduce:

- compensation;
- incremental emissions.

For this reason, a higher emission weight does not necessarily increase cost.

Some lambda settings improved both measures.

However, the observed points were not monotonic because:

- each lambda configuration was optimized separately;
- ALNS is stochastic;
- the search budget was limited;
- the spatial structures differ;
- empirical anchors affect the scalar objective.

---

## Repository structure

```text
.
├── main.py
├── requirements.txt
├── README.md
├── alns_solver/
├── exact_solver/
├── core/
├── experiments/
├── visualization/
├── configs/
│   └── experiments/
├── data/
│   └── solomon/
├── tests/
├── docs/
│   ├── OMA_Group3_PaperPresentation.pdf
│   ├── OMA_Group3_FinalProject_Report.pdf
│   └── OMA_Group3_FinalProject_Presentation.pdf
├── outputs/
└── .vscode/
    └── launch.json
```

---

## Main repository components

### `main.py`

Primary project entry point.

It loads the selected configuration and launches the required exact or ALNS experiment.

### `core/`

Contains shared domain and evaluation logic, including:

- configuration handling;
- instance loading;
- objective evaluation;
- scheduling and route state;
- shared solution representations;
- feasibility-related utilities.

### `exact_solver/`

Contains the exact optimization path, including:

- mathematical model construction;
- weighted-objective configuration;
- anchor solving;
- epsilon-constraint support;
- exact-solver output handling;
- solver-status and MIP-gap reporting.

### `alns_solver/`

Contains the paper-based ALNS implementation, including:

- ALNS control logic;
- destroy operators;
- repair operators;
- local-search operators;
- adaptive operator weighting;
- simulated-annealing acceptance;
- solution-state management.

### `experiments/`

Contains controlled experiment launchers and result-processing utilities.

These modules support:

- batch execution;
- seed allocation;
- result consolidation;
- lambda-level summaries;
- production artifact generation.

### `visualization/`

Contains route and experiment visualization functions.

### `configs/experiments/`

Contains reusable YAML experiment configurations.

### `data/solomon/`

Contains source Solomon data and transformed CSPTW-TN-DO benchmark files.

### `tests/`

Contains controlled checks for:

- algorithm structure;
- feasibility;
- output contracts;
- experiment boundaries;
- numerical behavior.

### `docs/`

Contains the paper presentation, final report, and final presentation.

### `outputs/`

Contains generated experiment results.

### `.vscode/launch.json`

Contains prepared Visual Studio Code run configurations.

---

## One-click Windows reproducibility packages

Prepackaged Windows bundles are available from the repository’s [Releases page](https://github.com/LouisePo82/CSPTW-TN-DO-Multiobjective-Optimization/releases).

The release contains:

1. Exact C101-25 solver and final-report evidence
2. ALNS C101-25 experiment and final-report evidence
3. ALNS R101-25 experiment and final-report evidence

Each package includes a `RUN_EXPERIMENT.bat` file that:

- checks whether Python is available;
- creates a local virtual environment;
- installs required dependencies;
- runs the selected experiment;
- writes reproduced outputs to a separate directory;
- preserves the frozen evidence used in the final report.

ALNS is stochastic, so reproduced objective values may differ from the frozen report results.

Runtime also depends on hardware and the operating environment.

---

# Running the project

## Option A — One-click Windows package

1. Open the [Releases page](https://github.com/LouisePo82/CSPTW-TN-DO-Multiobjective-Optimization/releases).
2. Download one of the ZIP packages.
3. Extract the ZIP completely.
4. Double-click `RUN_EXPERIMENT.bat`.
5. Wait until the command window reports completion.
6. Review generated files under `outputs_reproduced/`.

The first execution may require an internet connection to install Python and project dependencies.

---

## Option B — Visual Studio Code

### 1. Install the required software

Install:

- Python 3.11 or 3.12;
- Visual Studio Code;
- the Microsoft Python extension for VS Code;
- Git.

### 2. Clone the repository

```bash
git clone https://github.com/LouisePo82/CSPTW-TN-DO-Multiobjective-Optimization.git
cd CSPTW-TN-DO-Multiobjective-Optimization
```

### 3. Create a virtual environment

#### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Select the interpreter in VS Code

1. Open the project folder in VS Code.
2. Press `Ctrl+Shift+P` on Windows or `Cmd+Shift+P` on macOS.
3. Select **Python: Select Interpreter**.
4. Select the interpreter inside `.venv`.

### 6. Run a prepared configuration

1. Open **Run and Debug**.
2. Select one of the prepared configurations:
   - `Final ALNS — C101-25`
   - `Final ALNS — R101-25`
   - `Exact solver — C101-25`
3. Press the green Run button or press `F5`.

The `.vscode/launch.json` file supplies the required configuration argument to `main.py`.

---

## Option C — Terminal execution

### Final C101-derived ALNS experiment

```bash
python main.py \
  --config configs/experiments/alns_c101_25_mac_final_12seeds.yaml
```

### Final R101-derived ALNS experiment

```bash
python main.py \
  --config configs/experiments/alns_r101_25_mac_12seeds.yaml
```

### Exact C101-derived experiment

```bash
python main.py \
  --config configs/experiments/exact_c101_25_production_5lambda.yaml
```

On macOS, long experiments can be protected from system sleep:

```bash
caffeinate -dimsu \
python main.py \
  --config configs/experiments/alns_c101_25_mac_final_12seeds.yaml
```

---

## Option D — Advanced module execution

Examples:

```bash
python -m experiments.run_paper_alns_batch --help
```

```bash
python -m experiments.consolidate_paper_alns_results --help
```

```bash
python -m tests.test_alns_segment_production_ml63
```

Direct module execution is mainly intended for:

- diagnostics;
- custom seed allocation;
- controlled testing;
- artifact consolidation;
- development.

---

## Output structure

A typical ALNS run produces:

```text
run_config.json
run_results.json
run_results.csv
iteration_history.csv
operator_statistics.csv
best_solution.json
best_route_map.png
vehicle_routes/
artifact_manifest.json
```

A completed ALNS batch also produces:

```text
benchmark_outputs_consolidate.csv
lambda_summary.csv
launcher_manifest.json
```

The output files record:

- instance identity;
- benchmark family;
- lambda value;
- random seed;
- execution timestamp;
- active configuration;
- algorithm parameters;
- total distribution cost;
- operational emissions;
- normalized cost;
- normalized emissions;
- scalarized objective;
- validator status;
- iteration count;
- runtime;
- route details;
- visualization artifacts.

---

## Typical exact-solver outputs

The exact-solver path may produce files containing:

- solver status;
- incumbent objective;
- lower bound;
- relative MIP gap;
- solve time;
- selected lambda;
- normalization anchors;
- total cost;
- total emissions;
- normalized objectives;
- route decisions;
- post-solution validator result;
- visualization artifacts.

A time-limited feasible incumbent should not be interpreted as a zero-gap optimum.

---

## Validation and reproducibility

The repository includes tests for:

- solution-state consistency;
- destroy operators;
- repair operators;
- occasional-driver insertion strategies;
- local-search move operators;
- local-search swap operators;
- local-search control logic;
- adaptive operator-weight updates;
- simulated-annealing behavior;
- 301-iteration segment boundaries;
- production artifact generation;
- Solomon-derived instance contracts;
- exact-production output contracts;
- feasibility consistency between repair candidates and the final validator.

The final matched production configuration is identified by:

```text
Algorithm and feasibility-alignment commit: 1079caf
Final matched configuration commit: 5fa541d
Git tag: final-c101-r101-production
```

These identifiers support traceability between:

- the code revision;
- experiment configuration;
- final output set;
- report evidence.

---

## Reproducibility interpretation

Reproducibility in this repository means that a user can:

- use the same source structure;
- load the same benchmark instances;
- use the same configuration files;
- apply the same lambda values;
- use the same deterministic seed set;
- execute the same iteration budget;
- generate the same output structure;
- verify final feasibility.

Because ALNS is stochastic, reproduced objective values may still differ when:

- random-seed handling changes;
- the Python environment changes;
- dependency versions change;
- the operating system changes;
- the hardware changes;
- parallel execution affects ordering.

The retained production evidence is preserved so that the exact results used in the submitted report remain available.

---

## Interpretation notes

- ALNS is stochastic, so repeated execution may produce different objective values.
- Computational time depends on hardware and the operating environment.
- The exact solver may return feasible incumbents without proving zero-gap optimality.
- A solver status reached under a configured MIP-gap tolerance should be interpreted together with the reported numerical gap.
- Exact and ALNS outputs are not reported as certified heuristic optimality gaps unless both methods use equivalent feasible domains and objective-evaluation rules.
- Observed mean cost–emission points are not claimed to form a complete Pareto frontier.
- Empirical normalization anchors are not theoretical global ideal or nadir points.
- A normalized value outside zero and one is possible when later solutions fall outside the frozen empirical range.
- The transformed instances are controlled project benchmarks, not unchanged Solomon VRPTW instances.
- The C101–R101 comparison is based on one transformed instance from each family.
- The 301-iteration budget is much smaller than the reference paper’s search budget.
- Emission factors are comparative modeling parameters rather than a complete real-world carbon inventory.

---

## Project scope

The repository focuses on:

- crowd-shipping optimization;
- dedicated-vehicle routing;
- occasional-driver routing;
- alternative delivery-point assignment;
- transshipment-node handoff;
- customer time windows;
- vehicle capacities;
- route synchronization;
- exact mixed-integer optimization;
- paper-based ALNS reconstruction;
- operational-emission modeling;
- multi-objective normalization;
- weighted scalarization;
- benchmark transformation;
- controlled multi-seed experiments;
- feasibility validation;
- result consolidation;
- route visualization;
- reproducibility management.

---

## Areas outside the implemented scope

The following areas are outside the current implementation scope:

- complete reproduction of every computational experiment in the reference paper;
- direct reproduction from the original authors’ source code;
- proof that all undocumented paper details were reconstructed identically;
- complete Pareto-front generation;
- certified zero-gap optimality for all exact runs;
- real-time urban traffic;
- stochastic travel times;
- load-dependent fuel consumption;
- road-gradient effects;
- detailed vehicle-engine technology;
- electric-vehicle battery models;
- lifecycle emissions;
- parcel damage or service-reliability objectives;
- dynamic customer requests;
- real-time driver participation;
- endogenous pricing and compensation;
- large-scale industrial deployment.

---

## Main limitations

The final project has the following limitations:

1. The implementation was reconstructed from the published methodology rather than the original source code.
2. The final benchmark contains only two transformed 25-customer instances.
3. Each ALNS run uses 301 iterations rather than the paper’s 30,000 iterations.
4. The emission model uses simplified distance-based factors.
5. Load, traffic, speed, gradient, and detailed vehicle technologies are not modeled.
6. The normalization anchors are empirical.
7. Weighted-sum scalarization may not identify every nondominated solution.
8. The exact weighted-sum runs did not certify zero-gap optimality.
9. Exact and ALNS feasible domains were not fully proven equivalent.
10. The C101–R101 findings may also depend on the instance-generation procedure.
11. Each lambda configuration was optimized independently.
12. The results should not be generalized to all crowd-shipping systems.

---

## Possible future extensions

Future work could include:

- longer ALNS runs;
- multiple adaptive-weight segments;
- additional C, R, and RC benchmark instances;
- larger customer sets;
- broader seed sets;
- Pareto-based multi-objective search;
- successful epsilon-constraint experiments;
- heterogeneous vehicles;
- load-dependent emissions;
- congestion-dependent emissions;
- time-dependent travel;
- electric-vehicle energy models;
- dynamic occasional-driver participation;
- real company data;
- canonical exact–ALNS evaluators;
- shared feasible-domain verification;
- larger-scale reproducibility experiments;
- statistical significance testing.

---

## Main learning outcomes

The project involved:

- optimization-problem interpretation;
- mathematical optimization modeling;
- mixed-integer programming;
- exact-solver configuration;
- Adaptive Large Neighborhood Search;
- destroy-and-repair operators;
- regret insertion;
- local search;
- simulated annealing;
- adaptive operator selection;
- multi-objective optimization;
- empirical normalization;
- weighted scalarization;
- emission modeling;
- benchmark transformation;
- deterministic instance generation;
- feasibility validation;
- numerical consistency checking;
- controlled experiment design;
- multi-seed execution;
- statistical result consolidation;
- route visualization;
- runtime interpretation;
- configuration management;
- source-version traceability;
- artifact management;
- Windows packaging;
- reproducibility documentation;
- academic report development;
- presentation preparation.

---

## Academic interpretation

This repository should be interpreted as a course-project implementation built from a published optimization study.

It demonstrates how a proposed paper-presentation improvement can be developed into:

- a formal model extension;
- an executable objective function;
- exact and metaheuristic solution paths;
- transformed benchmark instances;
- controlled experiments;
- feasibility checks;
- reproducible computational evidence.

The repository does not claim:

- a new general ALNS algorithm;
- direct source-code reproduction of the paper;
- state-of-the-art benchmark performance;
- a complete Pareto frontier;
- universal superiority of one lambda value;
- certified global optimality for all retained solutions.

---

## References

Yu, V. F., Jodiawan, P., and Redi, A. A. N. P. (2022).  
*Crowd-shipping problem with time windows, transshipment nodes, and delivery options*.  
Transportation Research Part E: Logistics and Transportation Review, 157, 102545.  
https://doi.org/10.1016/j.tre.2021.102545

Solomon, M. M. (1987).  
*Algorithms for the vehicle routing and scheduling problems with time window constraints*.  
Operations Research, 35(2), 254–265.

Ropke, S., and Pisinger, D. (2006).  
*An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows*.  
Transportation Science, 40(4), 455–472.

Pisinger, D., and Ropke, S. (2007).  
*A general heuristic for vehicle routing problems*.  
Computers & Operations Research, 34(8), 2403–2435.

Kirkpatrick, S., Gelatt, C. D., and Vecchi, M. P. (1983).  
*Optimization by simulated annealing*.  
Science, 220(4598), 671–680.

Demir, E., Bektas, T., and Laporte, G. (2011).  
*A comparative analysis of several vehicle emission models for road freight transportation*.  
Transportation Research Part D: Transport and Environment, 16(5), 347–357.

Demir, E., Bektas, T., and Laporte, G. (2012).  
*An adaptive large neighborhood search heuristic for the pollution-routing problem*.  
European Journal of Operational Research, 223(2), 346–359.

Solomon VRPTW benchmark repository:  
https://www.sintef.no/projectweb/top/vrptw/100-customers/

---

## Final note

The tagged production configuration and released evidence packages are retained for reproducibility.

The final project should be understood as:

```text
Paper analysis
+
Proposed emission-aware improvement
+
Paper-based optimization reconstruction
+
Controlled exact and ALNS evaluation
+
Validation and reproducibility evidence
```

This structure preserves the complete development path from the original paper presentation to the final computational project.
