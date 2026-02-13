import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import math


# QUESTION 1.2(a)


def find_subtours(x,N):
    """
    Find all subtours in the solution
    Parameters:
    - x: Gurobi decision variables (x[i,j] binary variables)
    - N: list of node IDs
    Returns:
    - subtours: list of subtours (each subtour is a list of nodes)
    - has_subtour: boolean indicating if subtours exist
    """
    # Step 1: Build successor dictionary from solution
    successors = {}

    for i in N:
        for j in N:
            if i != j and x[i,j].X > 0.5: # arc is used in solution
                successors[i] = j
                break # each node has exactly one outgoing arc

    # Step 2: Find all cycles by following successors
    unvisited = set(N)
    tours = []

    while unvisited:
        start = next(iter(unvisited))
        current = start
        tour = []

        while True:
            tour.append(current)
            unvisited.discard(current) # mark as visited
            current = successors[current] # move to next node

            if current == start: # completed cycle
                break
        tours.append(tour)

    # Step 3: Identify subtours (tours that don't contain the depot)
    depot = N[0]
    subtours = []
    for tour in tours:
        if depot not in tour:
            subtours.append(tour)
    has_subtour = len(subtours) > 0
    return subtours, has_subtour


def print_subtours(subtours):
    """Print the subtours found"""
    if not subtours:
        print("✅ No subtours found - solution is valid!")
    else:
        print(f"❌ Found {len(subtours)} subtour(s):")
        for i, subtour in enumerate(subtours, 1):
            tour_str = " → ".join(map(str, subtour))
            tour_str += f" → {subtour[0]}"  # Show it's a cycle
            print(f"   Subtour {i}: {tour_str}")


# Test the function with a simple DFJ model
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Subtour Detection Function")
    print("=" * 60)


    NUM_CUSTOMERS = 5
    path = "C101_025.xlsx"
    nodes_df = pd.read_excel(path, sheet_name="Nodes")
    nodes_df = nodes_df.head(NUM_CUSTOMERS + 1)

    N = nodes_df["id"].tolist()
    coords = {row["id"]: (row["cx"], row["cy"]) for _, row in nodes_df.iterrows()}
    n = len(N)

    # Distance matrix
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

    print(f"\nTesting with {NUM_CUSTOMERS} customers ({n} nodes)\n")

    # Create a simple DFJ model without subtour elimination constraints

    m = gp.Model("DFJ_Test")
    #m.setParam('OutputFlag', 0)  # Suppress Gurobi output

    # Variables
    x = m.addVars(N, N, vtype=GRB.BINARY, name="x")

    # Flow constraints only (NO SEC constraints yet!)
    for i in N:
        m.addConstr(gp.quicksum(x[i, j] for j in N if i != j) == 1, name=f"out_{i}")
    for j in N:
        m.addConstr(gp.quicksum(x[i, j] for i in N if i != j) == 1, name=f"in_{j}")

    # Objective
    m.setObjective(gp.quicksum(c[i, j] * x[i, j] for i in N for j in N), GRB.MINIMIZE)

    # Solve
    print("Solving DFJ without SEC constraints...")
    m.optimize()

    if m.status == GRB.OPTIMAL:
        print(f"✅ Solution found with objective: {m.objVal}\n")

        # Print the solution
        print("Solution arcs:")
        for i in N:
            for j in N:
                if i != j and x[i, j].X > 0.5:
                    print(f"   {i} → {j} (distance: {c[i, j]})")

        print("\n" + "=" * 60)
        print("Checking for subtours...")
        print("=" * 60 + "\n")

        # TEST OUR FUNCTION!
        subtours, has_subtour = find_subtours(x, N)
        print_subtours(subtours)

        if has_subtour:
            print(f"\n📊 Summary:")
            print(f"   Total tours found: {len(subtours) + 1}")  # +1 for main tour
            print(f"   Subtours (illegal): {len(subtours)}")
            print(f"   Total nodes in subtours: {sum(len(st) for st in subtours)}")

            # Show which nodes are in subtours
            subtour_nodes = set()
            for st in subtours:
                subtour_nodes.update(st)
            print(f"   Nodes in subtours: {sorted(subtour_nodes)}")
    else:
        print(f"❌ Optimization failed with status {m.status}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
