# CSPTW-TN-DO Multiobjective Optimization

This repository provides the algorithm implementation, experimental configurations, and reproducibility materials for a multi-objective extension of the CSPTW-TN-DO problem.

The study includes:

- an exact MILP formulation for the C101-derived 25-customer instance;
- a paper-faithful Adaptive Large Neighborhood Search implementation;
- cost-emission scalarization across five lambda settings;
- matched C101-derived and R101-derived experiments;
- final computational evidence and reproducibility packages.

## Final experimental design

- Instances: C101-derived and R101-derived
- Customers: 25
- Lambda values: 0, 0.25, 0.50, 0.75, 1.00
- ALNS iterations: 301 per run
- Seeds: 2029–2040
- Runs: 60 per instance
- Total ALNS runs: 120

## Reproducibility packages

The executable and frozen-evidence packages will be uploaded to the Releases section.

## Citation

This repository accompanies the final course project report on emission-aware CSPTW-TN-DO optimization.
