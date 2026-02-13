import gurobipy as gp
from gurobipy import GRB
import math
import pandas as pd


# QUESTION 1.1(b)
# MTZ


NUM_CUSTOMERS = 15  # Change this to 5, 10, 15, 20, or 25
# Parameters
path = "C101_025.xlsx"
nodes_df = pd.read_excel(path, sheet_name="Nodes")

nodes_df = nodes_df.head(NUM_CUSTOMERS + 1)  # +1 for depot

N = nodes_df["id"].tolist()
coords = {row["id"]: (row["cx"], row["cy"]) for _, row in nodes_df.iterrows()}
n = len(N)

# Distance matrix (Euclidean, rounded to nearest integer)
c = {}
for i in N:
    for j in N:
        if i == j:
            c[(i, j)] = 0
        else:
            xi, yi = coords[i]
            xj, yj = coords[j]
            dist = math.sqrt((xj - xi) ** 2 + (yj - yi) ** 2)
            c[(i, j)] = round(dist)

print(f"Testing MTZ with {NUM_CUSTOMERS} customers ({len(N)} nodes total)")

# Model
m = gp.Model("MTZ")
m.setParam('TimeLimit', 600)  # 10 minute limit
# Decision Variables
x = m.addVars(N, N, vtype=GRB.BINARY, name="x")

# u variables: position in tour (depot is fixed at 1)
u = m.addVars(N, vtype=GRB.CONTINUOUS, name="u")
u[N[0]].LB = 1
u[N[0]].UB = 1

# Set bounds for other nodes
for i in N[1:]:  # All nodes except depot
    u[i].LB = 2
    u[i].UB = n

# Constraints

# Each node has exactly one outgoing arc
for i in N:
    m.addConstr(gp.quicksum(x[i, j] for j in N if i != j) == 1, name=f"out_{i}")

# Each node has exactly one incoming arc
for j in N:
    m.addConstr(gp.quicksum(x[i, j] for i in N if i != j) == 1, name=f"in_{j}")

# MTZ subtour elimination constraints
# Only for non-depot nodes to avoid redundancy
for i in N[1:]:  # Exclude depot from i
    for j in N[1:]:  # Exclude depot from j
        if i != j:
            m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name=f"mtz_{i}_{j}")

# Objective: minimize total distance
m.setObjective(gp.quicksum(c[i, j] * x[i, j] for i in N for j in N), GRB.MINIMIZE)

m.update()
print(f"Number of variables: {m.NumVars}")
print(f"Number of constraints: {m.NumConstrs}")

print("Starting optimization...")

# Solve
m.optimize()

# Output solution
if m.status == GRB.OPTIMAL:
    print(f"\n{'=' * 50}")
    print(f"\nOptimal solution found!")
    print(f"{'=' * 50}")
    print(f"Total distance: {m.objVal}")
    print(f"Solution time: {m.Runtime:.2f} seconds")
    print("\nRoute:")

    # Reconstruct tour
    tour = [N[0]]  # Start at depot
    current = N[0]
    while len(tour) < n:
        for j in N:
            if x[current, j].X > 0.5:  # Arc is used
                tour.append(j)
                current = j
                break
    tour.append(N[0])  # Return to depot
    print(" -> ".join(map(str, tour)))
elif m.status == GRB.TIME_LIMIT:  # ⭐ ADD THIS
    print(f"\nTime limit reached!")
    if m.SolCount > 0:
        print(f"Best solution found: {m.objVal}")
        print(f"Solution time: {m.Runtime:.2f} seconds")
    else:
        print("No feasible solution found within time limit")

elif m.status == GRB.INFEASIBLE:  # ⭐ ADD THIS
    print("\nModel is infeasible!")
else:
    print(f"Optimization failed with status {m.status}")
