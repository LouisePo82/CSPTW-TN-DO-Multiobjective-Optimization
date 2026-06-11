# Multi-Objective CSPTW-TN-DO Optimization

This repository contains the exact and metaheuristic optimization framework developed for the final project of the course **Optimization Models and Applications**.

The project is based on:

> Yu, V. F., Jodiawan, P., and Redi, A. A. N. P. (2022).  
> *Crowd-shipping problem with time windows, transshipment nodes, and delivery options*.  
> Transportation Research Part E: Logistics and Transportation Review, 157, 102545.  
> https://doi.org/10.1016/j.tre.2021.102545

The reference study introduces the Crowd-Shipping Problem with Time Windows, Transshipment Nodes, and Delivery Options (CSPTW-TN-DO), formulates an exact mathematical model, and develops an Adaptive Large Neighborhood Search (ALNS) with the main objective of minimizing total distribution cost.

This course project reconstructs the paper-based solution framework and extends the objective layer by adding operational emissions. The final experiments examine how routing and delivery decisions change under different cost and environmental priorities.

Project documents:

- [Final report](docs/OMA_Group3_FinalProject_Report.docx)
- [Final presentation](docs/OMA_Group3_FinalProject_Presentation.pdf)

---

## Course context

**Course:** Optimization Models and Applications  
**Project type:** Final group project  

### Team members

1. Dong Boi Thi
2. To Nguyen Tan Phuong

---

## Relationship to the reference paper

The reference paper provides the original CSPTW-TN-DO structure, including:

- dedicated vehicles;
- occasional drivers;
- customer time windows;
- transshipment nodes;
- alternative delivery points;
- multiple customer delivery preferences;
- the original cost objective;
- an exact mathematical formulation;
- an ALNS solution approach.

This project does not claim to reproduce every computational experiment or every undocumented implementation detail from the paper.

The main project extension includes:

- dedicated-vehicle operational emissions;
- occasional-driver incremental emissions;
- empirical objective normalization;
- normalized weighted scalarization;
- exact and ALNS evaluation under five objective priorities;
- transformed C101-derived and R101-derived benchmark instances;
- shared feasibility validation and experiment automation.

The project does not propose a new ALNS. It uses a paper-based reconstruction of the reported search framework and adds the emission-aware objective, numerical safeguards, validation logic, and final experimental design.

---

## Problem components

The implemented framework includes:

- dedicated vehicles;
- occasional drivers;
- customer home delivery;
- customer pickup at alternative delivery points;
- transshipment-node parcel handoff;
- customer time windows;
- vehicle-capacity constraints;
- synchronization between dedicated vehicles and occasional drivers;
- distribution-cost and operational-emission objectives.

---

## Optimization methods

### Exact MILP solver

The exact formulation was evaluated on the 25-customer C101-derived instance under a fixed computational budget.

It supports:

- cost-oriented and emission-oriented anchor configurations;
- normalized weighted-sum objectives;
- epsilon-constraint configurations;
- solver status, bound, MIP gap, and runtime reporting;
- post-solution feasibility validation;
- route and visualization artifact export.

A feasible incumbent is not reported as an optimal solution unless the solver certifies optimality.

### Paper-based ALNS reconstruction

The ALNS implementation includes:

- the paper-defined destroy-operator families;
- the paper-defined repair structure;
- feasible insertion filtering;
- eight local-search operators;
- adaptive operator weights;
- simulated-annealing acceptance;
- deterministic random seeds;
- shared final feasibility validation;
- iteration histories and operator statistics;
- combined and per-vehicle route visualizations.

The production configuration does not use:

- enhanced search behavior;
- fallback repair;
- operator substitution;
- operator resampling.

---

## Multi-objective formulation

The production experiments use the following normalized scalar objective:

(1-\lambda)\widehat{C}
+
\lambda\widehat{E}
$$

where:

$\widehat{C}$ is the normalized distribution cost;
$\widehat{E}$ is the normalized operational emissions;
$\lambda = 0$ represents a cost-oriented objective;
$\lambda = 1$ represents an emission-oriented objective.

The normalized cost is calculated as:

\frac{C-C^{\mathrm{ideal}}}
{C^{\mathrm{nadir}}-C^{\mathrm{ideal}}}
$$

The normalized emissions are calculated as:

\frac{E-E^{\mathrm{ideal}}}
{E^{\mathrm{nadir}}-E^{\mathrm{ideal}}}
$$

The tested lambda values are:

0.00, 0.25, 0.50, 0.75, 1.00

The normalization values are empirical anchors obtained from validated calibration solutions. They are not claimed to be globally proven theoretical ideal or nadir points.

---

## Benchmark instances

The project uses two transformed 25-customer benchmark instances:

- **C101-derived:** clustered customer structure;
- **R101-derived:** randomly distributed customer structure.

The source customer coordinates, demands, service times, and time windows were taken from the Solomon VRPTW benchmark repository:

- https://www.sintef.no/projectweb/top/vrptw/100-customers/

The original Solomon data do not contain occasional drivers, transshipment nodes, alternative delivery points, delivery types, or emission parameters. These elements were added during the project instance-construction process.

The final cases are therefore described as **Solomon-derived CSPTW-TN-DO instances**, not unchanged Solomon VRPTW instances. Their objective values are not directly comparable with the published Solomon VRPTW results.

---

## Final matched ALNS experiment

| Setting            | C101-derived | R101-derived |
| ------------------ | -----------: | -----------: |
| Customers          |           25 |           25 |
| Lambda values      |            5 |            5 |
| Seeds              |    2029–2040 |    2029–2040 |
| Seeds per lambda   |           12 |           12 |
| Iterations per run |          301 |          301 |
| Runs per instance  |           60 |           60 |
| Final validation   |    100% pass |    100% pass |

The final matched experiment contains **120 ALNS runs**.

Both instance families used:

- the same algorithm source;
- the same lambda values;
- the same seed set;
- the same 301-iteration limit;
- the same execution environment;
- separate output directories.

The 301-iteration budget completes one full 300-iteration adaptive segment and one additional post-update iteration. It is a controlled course-project budget, not a reproduction of the paper's 30,000-iteration search depth.

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
│   ├── OMA_Group3_FinalProject_Report.docx
│   └── OMA_Group3_FinalProject_Presentation.pdf
├── outputs/
└── .vscode/
    └── launch.json
```

---

## One-click Windows reproducibility packages

Prepackaged Windows bundles are available from the repository's [Releases page](https://github.com/LouisePo82/CSPTW-TN-DO-Multiobjective-Optimization/releases).

The release contains:

1. Exact C101-25 solver and final-report evidence
2. ALNS C101-25 experiment and final-report evidence
3. ALNS R101-25 experiment and final-report evidence

Each package includes a `RUN_EXPERIMENT.bat` file that:

- checks whether Python is available;
- creates a local virtual environment;
- installs the required dependencies;
- runs the selected experiment;
- writes reproduced outputs to a separate directory;
- preserves the frozen evidence used in the final report.

ALNS is stochastic, so reproduced objective values may differ from the frozen report results. Runtime also depends on hardware and the operating environment.

---

# Running the project

## Option A — One-click Windows package

1. Open the [Releases page](https://github.com/LouisePo82/CSPTW-TN-DO-Multiobjective-Optimization/releases).
2. Download one of the ZIP packages.
3. Extract the ZIP completely.
4. Double-click `RUN_EXPERIMENT.bat`.
5. Wait until the command window reports completion.
6. Review the generated files under `outputs_reproduced/`.

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

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

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

### 6. Run the prepared configuration

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

Direct module execution is mainly intended for diagnostics, custom seed allocation, controlled testing, and development.

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
- lambda value;
- random seed;
- execution timestamp;
- configuration;
- objective values;
- validation status;
- solution details;
- route visualizations.

---

## Validation and reproducibility

The repository includes tests for:

- solution-state consistency;
- destroy operators;
- repair operators;
- occasional-driver insertion strategies;
- local-search move and swap operators;
- local-search control;
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

---

## Interpretation notes

- ALNS is stochastic, so repeated execution may produce different objective values.
- Computational time depends on hardware and the operating environment.
- The exact solver may return feasible incumbents without proving optimality.
- Exact and ALNS outputs are not reported as certified heuristic optimality gaps unless both methods use equivalent feasible domains and objective-evaluation rules.
- Observed mean cost-emission points are not claimed to form a complete Pareto frontier.
- Empirical normalization anchors are not theoretical global ideal or nadir points.
- The transformed instances are controlled project benchmarks, not unchanged Solomon instances.

---

## Main learning outcomes

The project involved:

- mathematical optimization modeling;
- exact mixed-integer optimization;
- adaptive large neighborhood search;
- multi-objective optimization;
- simulated annealing;
- adaptive operator selection;
- benchmark transformation;
- feasibility validation;
- experiment automation;
- statistical result consolidation;
- data visualization;
- reproducibility and configuration management.

The tagged production configuration and released evidence packages are retained for reproducibility.

---

## References

Yu, V. F., Jodiawan, P., and Redi, A. A. N. P. (2022).  
*Crowd-shipping problem with time windows, transshipment nodes, and delivery options*.  
Transportation Research Part E: Logistics and Transportation Review, 157, 102545.  
https://doi.org/10.1016/j.tre.2021.102545

Solomon VRPTW benchmark repository:  
https://www.sintef.no/projectweb/top/vrptw/100-customers/
