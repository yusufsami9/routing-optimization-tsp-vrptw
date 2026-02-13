import gurobipy as gp
from gurobipy import GRB
import pandas as  pd
import math
from q_1_2_a import find_subtours
import time


# QUESTION 1.2(b)

NUM_CUSTOMERS = 25

path = "C101_025.xlsx"
nodes_df = pd.read_excel(path, sheet_name="Nodes")
nodes_df = nodes_df.head(NUM_CUSTOMERS + 1)
N = nodes_df["id"].tolist()
coords = {row["id"]: (row["cx"], row["cy"]) for _, row in nodes_df.iterrows()}
n = len(N)

c = {}
for i in N:
    for j in N:
        if i == j:
            c[(i,j)] = 0
        else:
            xi,yi = coords[i]
            xj,yj = coords[j]
            dist = math.sqrt((xj-xi)**2 + (yj-yi)**2)
            c[(i,j)] = round(dist)

print(f"Testing with {NUM_CUSTOMERS} customers ({len(N)} nodes total)")

m = gp.Model("DFJ_improved")

m.setParam('TimeLimit', 600)

x = m.addVars(N,N, vtype=GRB.BINARY, name="x")

for i in N:
    m.addConstr(gp.quicksum(x[i,j] for j in N if i != j ) == 1, name=f"out_{i}")
for j in N:
    m.addConstr(gp.quicksum(x[i, j] for i in N if i != j) == 1, name=f"in_{j}")

m.update()

m.setObjective(gp.quicksum(x[i,j]*c[i,j] for i in N for j in N), GRB.MINIMIZE)
m.update()

# Iterative cut generation
iteration = 0
objectives = []
start_time = time.time()
print("\nStarting iterative cut generation...")
print("="*60)

while True:
    iteration += 1
    print(f"\nIteration {iteration}:")

    m.optimize()

    if m.status != GRB.OPTIMAL:
        print(f"  ❌ Solver failed with status {m.status}")
        break

    print(f"  Objective: {m.objVal}")
    objectives.append(m.objVal)

    #check for subtours
    subtours, has_subtour = find_subtours(x,N)

    if not has_subtour:
        print("  ✅ No subtours found - Optimal solution!")
        break

    print(f"  ❌ Found {len(subtours)} subtour(s):")
    for st in subtours:
        print(f"     {' → '.join(map(str, st))} → {st[0]}")

    # add subtour elimination constraints for found subtours
    for subtour in subtours:
        m.addConstr(gp.quicksum(x[i,j] for i in subtour for j in subtour if i != j ) <= len(subtour) - 1, name= f"sec_{tuple(subtour)}")
    m.update()

# Final results
end_time = time.time()
total_time = end_time - start_time
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Converged in {iteration} iterations")
print(f"Total time: {total_time:.2f} seconds")
print(f"Final objective: {m.objVal}")
print(f"Total SEC constraints added: {sum(1 for c in m.getConstrs() if 'sec' in c.ConstrName)}")

# Print tour
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
print(f"Optimal tour: {' → '.join(map(str, tour))}")
print("=" * 60)


# Question 1.2(c): Plot objective value per iteration
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(objectives) + 1), objectives, marker='o', linewidth=2, markersize=8)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Objective Value (Total Distance)', fontsize=12)
plt.title(f'Objective Value per Iteration - {NUM_CUSTOMERS} Customers', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(range(1, len(objectives) + 1))

# Annotate each point with its value
for i, obj in enumerate(objectives, 1):
    plt.annotate(f'{obj:.0f}', (i, obj), textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.savefig(f'objective_plot_{NUM_CUSTOMERS}_customers.png', dpi=300)
plt.show()
print(f"\nPlot saved as: objective_plot_{NUM_CUSTOMERS}_customers.png")
