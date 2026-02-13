import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import math
import time

# ============================================================================
# Question 2.2(b): Time Window Width Trade-off Analysis
# ============================================================================

# PARAMETER: Change this for each run!
WINDOW_MULTIPLIER = 7


print(f"\n{'=' * 80}")
print(f"QUESTION 2.2(b): TIME WINDOW WIDTH ANALYSIS")
print(f"Window Multiplier: {WINDOW_MULTIPLIER} ({WINDOW_MULTIPLIER * 100:.0f}% of original)")
print(f"{'=' * 80}\n")

# ============================================================================
# DATA LOADING
# ============================================================================

path = "C101_025.xlsx"
nodes_df = pd.read_excel(path, sheet_name="Nodes")
requests_df = pd.read_excel(path, sheet_name="Requests")
fleet_df = pd.read_excel(path, sheet_name="Fleet")

N = nodes_df["id"].tolist()
depot = 0
customers = [i for i in N if i != 0]

# ============================================================================
# COORDINATES
# ============================================================================

coords = {}
for _, row in nodes_df.iterrows():
    node_id = row["id"]
    x_coord = row["cx"]
    y_coord = row["cy"]
    coords[node_id] = (x_coord, y_coord)

# ============================================================================
# DISTANCE MATRIX
# ============================================================================

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

t = c.copy()  # Travel time = distance

# ============================================================================
# CUSTOMER DATA WITH ADJUSTED TIME WINDOWS
# ============================================================================

demand = {}
ready_time = {}
due_time = {}
service_time = {}

# Depot data
demand[depot] = 0
ready_time[depot] = 0
service_time[depot] = 0

max_travel_time = fleet_df.loc[0, "max_travel_time"]
due_time[depot] = max_travel_time

# Customer data with adjusted time windows
for _, row in requests_df.iterrows():
    node_id = int(row["node"])

    demand[node_id] = row["quantity"]

    # Get original time window
    original_ready = row["start"]
    original_due = row["end"]
    original_width = original_due - original_ready

    # Calculate center of window
    center = (original_ready + original_due) / 2

    # Adjust width based on multiplier, keeping center fixed
    new_width = original_width * WINDOW_MULTIPLIER
    ready_time[node_id] = center - new_width / 2
    due_time[node_id] = center + new_width / 2

    service_time[node_id] = row["service_time"]

# Calculate average widths
avg_original_width = sum((requests_df.loc[i - 1, "end"] - requests_df.loc[i - 1, "start"])
                         for i in customers) / len(customers)
avg_new_width = avg_original_width * WINDOW_MULTIPLIER

print(f"Time Window Adjustment:")
print(f"  Original average width: {avg_original_width:.1f} time units")
print(f"  New average width: {avg_new_width:.1f} time units")
print(f"  Examples:")
print(f"    Customer 1: [{ready_time[1]:.1f}, {due_time[1]:.1f}] (width: {due_time[1] - ready_time[1]:.1f})")
print(f"    Customer 5: [{ready_time[5]:.1f}, {due_time[5]:.1f}] (width: {due_time[5] - ready_time[5]:.1f})")

# ============================================================================
# CAPACITY INFO (Not enforced in Question 2.2)
# ============================================================================

Q = fleet_df.loc[0, "capacity"]
total_demand = sum(demand[i] for i in customers)

print(f"\n⚠️  NOTE: Question 2.2 requirement - Ignoring vehicle capacity constraints")
print(f"  Vehicle capacity: {Q} (NOT enforced)")
print(f"  Total demand: {total_demand}")

# ============================================================================
# BIG-M CALCULATION
# ============================================================================

M = max(due_time.values()) + max(service_time.values()) + max(c.values())

# ============================================================================
# MODEL CREATION
# ============================================================================

m = gp.Model(f"VRPTW_NoCapacity_Width{WINDOW_MULTIPLIER}")
m.setParam('TimeLimit', 600)
m.setParam('OutputFlag', 1)

# ============================================================================
# DECISION VARIABLES
# ============================================================================

# Binary arc variables
x = m.addVars(N, N, vtype=GRB.BINARY, name="x")

# No self-loops
for i in N:
    m.addConstr(x[i, i] == 0, name=f"no_self_{i}")

# NO u variables (capacity constraints removed)
# Time variables
T = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="T")

m.update()

# ============================================================================
# FLOW CONSERVATION CONSTRAINTS
# ============================================================================

for i in customers:
    m.addConstr(
        gp.quicksum(x[i, j] for j in N if j != i) == 1,
        name=f"out_{i}"
    )

for j in customers:
    m.addConstr(
        gp.quicksum(x[i, j] for i in N if i != j) == 1,
        name=f"in_{j}"
    )

u = m.addVars(customers, vtype=GRB.CONTINUOUS, lb=1, ub=len(customers), name="u")

num_subtour_constraints = 0
for i in customers:
    for j in customers:
        if i != j:
            m.addConstr(
                u[j] >= u[i] + 1 - len(customers) * (1 - x[i, j]),
                name=f"mtz_{i}_{j}"
            )
            num_subtour_constraints += 1

# ============================================================================
# CAPACITY CONSTRAINTS - DISABLED FOR QUESTION 2.2

# All capacity constraints are removed

print("\nCapacity constraints: DISABLED (as required by Question 2.2)")

# ============================================================================
# TIME WINDOW CONSTRAINTS
# ============================================================================

# Time propagation
for i in N:
    for j in customers:
        if i != j:
            m.addConstr(
                T[j] >= T[i] + service_time[i] + t[i, j] - M * (1 - x[i, j]),
                name=f"time_{i}_{j}"
            )

# Time window bounds
for i in customers:
    m.addConstr(T[i] >= ready_time[i], name=f"ready_{i}")
    m.addConstr(T[i] <= due_time[i], name=f"due_{i}")

# Depot time
m.addConstr(T[depot] == 0, name="depot_time")

m.update()


# OBJECTIVE FUNCTION


m.setObjective(
    gp.quicksum(c[i, j] * x[i, j] for i in N for j in N if i != j),
    GRB.MINIMIZE
)

m.update()


print("\n" + "=" * 80)
print("SOLVING MODEL...")
print("=" * 80)

start_time = time.time()
m.optimize()
end_time = time.time()
solve_time = end_time - start_time

# ============================================================================
# RESULTS
# ============================================================================

if m.status == GRB.OPTIMAL:
    print("\n✅ OPTIMAL SOLUTION FOUND!")
    print("=" * 80)
    print(f"Objective value (total distance): {m.objVal:.2f}")
    print(f"Solution time: {solve_time:.2f} seconds")

    # Count number of vehicles used
    num_vehicles = sum(1 for j in customers if x[depot, j].X > 0.5)
    print(f"Number of vehicles used: {num_vehicles}")

    # Extract all used arcs
    used_arcs = [(i, j) for i in N for j in N if x[i, j].X > 0.5]
    print(f"Total arcs used: {len(used_arcs)}")

    print("\n" + "=" * 80)
    print("VEHICLE ROUTES")
    print("=" * 80)

    # Reconstruct routes for each vehicle
    remaining_arcs = set(used_arcs)
    vehicle_num = 1

    while any(arc[0] == depot for arc in remaining_arcs):
        # Find a route starting from depot
        route = [depot]
        current = depot
        route_arcs = []

        while True:
            # Find next node
            next_node = None
            for (i, j) in remaining_arcs:
                if i == current:
                    next_node = j
                    remaining_arcs.remove((i, j))
                    route_arcs.append((i, j))
                    break

            if next_node is None:
                break

            route.append(next_node)
            current = next_node

            # If back at depot, route is complete
            if current == depot:
                break

        # Calculate route statistics
        route_distance = sum(c[i, j] for (i, j) in route_arcs)
        route_customers = [node for node in route if node != depot]

        # Get time of last customer visit
        if len(route_customers) > 0:
            last_customer = route_customers[-1]
            completion_time = T[last_customer].X + service_time[last_customer]
        else:
            completion_time = 0

        print(f"\nVehicle {vehicle_num}:")
        print(f"  Route: {' → '.join(map(str, route))}")
        print(f"  Customers visited: {len(route_customers)}")
        print(f"  Route distance: {route_distance:.2f}")
        print(f"  Completion time: {completion_time:.2f}")

        # Show detailed timing for each customer
        if len(route_customers) > 0:
            print(f"  Customer details:")
            for cust in route_customers:
                arrival = T[cust].X
                tw_start = ready_time[cust]
                tw_end = due_time[cust]
                wait_time = max(0, tw_start - arrival)
                print(
                    f"    Customer {cust}: arrive {arrival:.1f}, TW [{tw_start:.1f}, {tw_end:.1f}], wait {wait_time:.1f}")

        vehicle_num += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Window Multiplier: {WINDOW_MULTIPLIER} ({WINDOW_MULTIPLIER * 100:.0f}%)")
    print(f"Average Window Width: {avg_new_width:.1f} time units")
    print(f"Total Distance: {m.objVal:.2f} km")
    print(f"Vehicles Used: {num_vehicles}")
    print(f"Solution Time: {solve_time:.2f} seconds")
    print("=" * 80)

elif m.status == GRB.TIME_LIMIT:
    print("\n⏰ TIME LIMIT REACHED")
    print(f"Best solution found: {m.objVal:.2f}")
    print(f"Optimality gap: {m.MIPGap * 100:.2f}%")
    print(f"Time: {solve_time:.2f} seconds")

elif m.status == GRB.INFEASIBLE:
    print("\n❌ MODEL IS INFEASIBLE")
    print(f"Window multiplier {WINDOW_MULTIPLIER} makes the problem infeasible!")
    print("Time windows are too tight - no valid solution exists")
    print("\nPossible reasons:")
    print(f"  - Average window width ({avg_new_width:.1f}) too narrow")
    print(f"  - Some customers cannot be reached within their time windows")
    print(f"  - Time window conflicts prevent feasible routing")

    # Compute IIS to help debug
    print("\nComputing IIS (Irreducible Inconsistent Subsystem)...")
    m.computeIIS()
    m.write(f"infeasible_model_width{WINDOW_MULTIPLIER}.ilp")
    print(f"IIS written to infeasible_model_width{WINDOW_MULTIPLIER}.ilp")

else:
    print(f"\n❌ OPTIMIZATION FAILED")
    print(f"Status code: {m.status}")
    print(f"Time: {solve_time:.2f} seconds")

print("\n" + "=" * 80)
print("OPTIMIZATION COMPLETE")
print("=" * 80)
