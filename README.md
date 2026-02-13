# Routing Optimization with Gurobi (TSP & VRPTW)

This repository contains a routing optimization project developed using Python and the Gurobi optimizer.  
The project focuses on modeling and solving routing and vehicle routing problems under realistic operational constraints.

The main goal is to minimize total driven distance while analyzing the impact of:
- routing formulations,
- capacity and time window constraints,
- driver working time limits,
- and different modeling approaches.

---

## Problem Setting

A logistics company operates from a central depot and serves multiple customers.  
Each customer has:
- a location,
- a demand,
- a service time,
- and a delivery time window.

The routing problem is studied under different levels of realism:
- Basic routing (TSP),
- Vehicle Routing with Time Windows (VRPTW),
- Capacity constraints,
- Driver working time limits,
- Sensitivity analysis on time window width,
- Alternative formulation using a time-expanded network.

---

## Models & Methods

**TSP formulations**
- MTZ formulation  
- DFJ formulation (full)  
- DFJ with lazy subtour elimination (cut generation)

**VRPTW extensions**
- Capacity constraints  
- Time window constraints  
- Vehicle count penalty in the objective  
- Driver working time constraint (7 hours)

**Advanced modeling**
- Time window width sensitivity analysis  
- Time-expanded network formulation (discrete-time)

All models are implemented as Mixed-Integer Programming (MIP) formulations and solved with Gurobi.

---

## Tech Stack

- Python  
- Gurobi Optimizer  
- Pandas  
- Matplotlib  

---
.
├── src/ # Python implementations of all models
├── results/ # Selected plots / outputs
├── report/ # Final report (PDF)
├── C101_025.xlsx # Input dataset (Nodes, Requests, Fleet)
└── README.md
---

## Example Output

**Objective value per iteration for DFJ with lazy cut generation (25 customers):**

![Objective per iteration](results/objective_value_per_iteration_25_customers.png)

This plot illustrates how the objective value evolves as violated subtour elimination constraints are iteratively added to the model.

---

## How to Run

1. Make sure `C101_025.xlsx` is placed in the repository root.  
2. Install required dependencies:
   - Gurobi  
   - pandas  
   - matplotlib  
3. Run any script from the `src/` folder:
   ```bash
   python src/<script_name>.py
   ```
   Note: A valid Gurobi license is required to run the optimization models.

## Report

The full analysis, experimental results, and tables are available in the report:
```
report/assignment_report.pdf
```
## Context

This project was developed as part of a graduate-level course on Decision Making in Transport and Mobility at Eindhoven University of Technology (TU/e).

The repository is shared as a portfolio project to demonstrate practical experience with:

- mathematical optimization,

- vehicle routing problems,

- and solver-based modeling in Python.




