import gurobipy as gp
from gurobipy import GRB
import math
import pandas as pd
from itertools import combinations


# QUESTION 1.1(C)
# DFJ


NUM_CUSTOMERS = 10  # Change this to 5, 10, 15, 20, or 25

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

print(f"Testing with {NUM_CUSTOMERS} customers ({len(N)} nodes total)")

# Model
m = gp.Model("DFJ")

m.setParam('TimeLimit', 600)  # 10 minute limit

# Decision Variables

x = m.addVars(N,N, vtype=GRB.BINARY, name="x")

# Constraints

for i in N:
    m.addConstr(gp.quicksum(x[i,j] for j in N if i != j ) == 1, name=f"out_{i}")
for j in N:
    m.addConstr(gp.quicksum(x[i, j] for i in N if i != j) == 1, name=f"in_{j}")

#subtour elimination constraints

customers = N[1:] # get customers except the depot

print("Generating SEC constraints...")
constraint_count = 0

#for each possible subset size
for k in range(2,len(customers)+1):
    #generate all subsets of size k
    for S in combinations(customers, k):
        # add subtour constraint for this subset S
        m.addConstr(gp.quicksum(x[i,j] for  i in S for j in S if i != j ) <= len(S)-1, name=f"sec_{S}")
        constraint_count += 1

print(f"Added {constraint_count} SEC constraints")


m.update()
# Objective Function

m.setObjective(gp.quicksum(x[i,j]*c[i,j] for i in N for j in N), GRB.MINIMIZE)
m.update()

print("Starting optimization...")

m.optimize()

if m.status == GRB.OPTIMAL:
    print(f"\n{'=' * 50}")
    print(f"OPTIMAL SOLUTION FOUND")
    print(f"{'=' * 50}")
    print(f"Total distance: {m.objVal}")
    print(f"Solution time: {m.Runtime:.2f} seconds")

    depot = N[0]
    tour = [depot]
    current = depot

    while len(tour) < len(N):
        for j in N:
            if x[current, j].X > 0.5:
                tour.append(j)
                current = j
                break
    tour.append(depot)

    print(f"Route: {' -> '.join(map(str, tour))}")

elif m.status == GRB.TIME_LIMIT:
    print(f"\nTime limit reached!")
    if m.SolCount > 0:
        print(f"Best solution found: {m.objVal}")
    else:
        print("No feasible solution found")

elif m.status == GRB.INFEASIBLE:
    print("\nModel is infeasible!")

else:
    print(f"\nOptimization ended with status {m.status}")
