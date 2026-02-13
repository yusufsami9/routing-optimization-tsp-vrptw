import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import math
import time


# QUESTION 2.3
# Extended MTZ for capacities, time windows, AND driver working time (Question 2.3)

path = "C101_025.xlsx"
nodes_df = pd.read_excel(path, sheet_name="Nodes")
requests_df = pd.read_excel(path, sheet_name="Requests")
fleet_df = pd.read_excel(path, sheet_name="Fleet")

N = nodes_df["id"].tolist()

depot = 0
customers = [i for i in N if i != 0]

coords = {}
for _, row in nodes_df.iterrows():
    node_id = row["id"]
    x_coord = row["cx"]
    y_coord = row["cy"]
    coords[node_id] = (x_coord, y_coord)

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
t = c.copy()

demand = {}
ready_time = {}
due_time = {}
service_time = {}
demand[depot] = 0
ready_time[depot] = 0
service_time[depot] = 0

max_travel_time = fleet_df.loc[0, "max_travel_time"]
due_time[depot] = max_travel_time

for _, row in requests_df.iterrows():
    node_id = int(row["node"])
    demand[node_id] = row["quantity"]
    ready_time[node_id] = row["start"]
    due_time[node_id] = row["end"]
    service_time[node_id] = row["service_time"]

Q = fleet_df.loc[0, "capacity"]
print(f"\nVehicle capacity (Q): {Q}")

total_demand = sum(demand[i] for i in customers)
print(f"Total demand across all customers: {total_demand}")
print(f"Minimum vehicles needed (based on capacity): {math.ceil(total_demand / Q)}")

# Big M
M = max(due_time.values()) + max(service_time.values()) + max(c.values())

# Driver working time limit (Question 2.3)
H = 420  # 7 hours = 420 minutes
print(f"Driver working time limit (H): {H} minutes (7 hours)")

m = gp.Model("VRPTW_MTZ_DriverConstraint")

m.setParam('TimeLimit', 600)
m.setParam('OutputFlag', 1)

# Decision variables
x = m.addVars(N, N, vtype=GRB.BINARY, name="x")
for i in N:
    m.addConstr(x[i, i] == 0, name=f"no_self_{i}")

u = m.addVars(customers, vtype=GRB.CONTINUOUS, lb=0, ub=Q, name="u")
T = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="T")

# R[i]: Return time to depot if route ends at customer i (Question 2.3)
R = m.addVars(customers, vtype=GRB.CONTINUOUS, lb=0, name="R")

m.update()

# Flow Constraints
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

# Capacity constraints
for i in customers:
    for j in customers:
        if i != j:
            m.addConstr(
                u[j] >= u[i] + demand[j] - Q * (1 - x[i, j]),
                name=f"load_{i}_{j}"
            )

for i in customers:
    m.addConstr(u[i] >= demand[i], name=f"load_lb_{i}")
    m.addConstr(u[i] <= Q, name=f"load_ub_{i}")

# Time window constraints
for i in N:
    for j in customers:
        if i != j:
            m.addConstr(
                T[j] >= T[i] + service_time[i] + t[i, j] - M * (1 - x[i, j]),
                name=f"time_{i}_{j}"
            )

for i in customers:
    m.addConstr(T[i] >= ready_time[i], name=f"ready_{i}")
    m.addConstr(T[i] <= due_time[i], name=f"due_{i}")

# Depot constraints
m.addConstr(T[depot] == 0, name="depot_time")

# ============================================================================
# DRIVER WORKING TIME CONSTRAINTS (Question 2.3)
# ============================================================================
# DIAGNOSTIC: Check if original solution would violate H
print("\n" + "=" * 80)
print("DIAGNOSTIC: Checking feasibility of H = 420")
print("=" * 80)

print(f"\nMaximum possible working time (depot due time): {due_time[depot]}")
print(f"Working time limit (H): {H}")

if H < due_time[depot]:
    print(f"⚠️ H ({H}) < depot max time ({due_time[depot]}) - may cause infeasibility!")
else:
    print(f"✅ H ({H}) >= depot max time ({due_time[depot]}) - should be OK")

# Check what happens if we relax H
print(f"\nIf infeasible, try increasing H to at least {due_time[depot]}")
print("=" * 80 + "\n")
print("\nAdding driver working time constraints (Question 2.3)...")

# Use tighter Big-M for driver constraints
M_driver = due_time[depot]  # Maximum possible return time

# D1a: Lower bound on return time
for i in customers:
    m.addConstr(
        R[i] >= T[i] + service_time[i] + t[i, depot] - M_driver * (1 - x[i, depot]),
        name=f"return_lb_{i}"
    )

# D1b: Upper bound on return time
for i in customers:
    m.addConstr(
        R[i] <= T[i] + service_time[i] + t[i, depot] + M_driver * (1 - x[i, depot]),
        name=f"return_ub_{i}"
    )

# D2: Working time limit (7 hours = 420 minutes)
for i in customers:
    m.addConstr(
        R[i] <= H,
        name=f"max_work_{i}"
    )

print(f"Added {3 * len(customers)} driver working time constraints")

m.update()

# Objective Function
m.setObjective(
    gp.quicksum(c[i, j] * x[i, j] for i in N for j in N),
    GRB.MINIMIZE
)
m.update()

print("\n" + "=" * 80)
print("SOLVING MODEL WITH DRIVER WORKING TIME CONSTRAINT...")
print("=" * 80)

start_time = time.time()
m.optimize()
end_time = time.time()
solve_time = end_time - start_time

if m.status == GRB.OPTIMAL:
    print("✅ OPTIMAL SOLUTION FOUND!")
    print("=" * 80)
    print(f"Objective value (total distance): {m.objVal:.2f}")
    print(f"Solution time: {solve_time:.2f} seconds")

    num_vehicles = sum(1 for j in customers if x[depot, j].X > 0.5)
    print(f"Number of vehicles used: {num_vehicles}")

    used_arcs = [(i, j) for i in N for j in N if x[i, j].X > 0.5]

    print("\n" + "=" * 80)
    print("VEHICLE ROUTES WITH WORKING TIME ANALYSIS")
    print("=" * 80)

    remaining_arcs = set(used_arcs)
    vehicle_num = 1

    while any(arc[0] == depot for arc in remaining_arcs):
        route = [depot]
        current = depot
        route_arcs = []

        while True:
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

            if current == depot:
                break

        route_distance = sum(c[i, j] for (i, j) in route_arcs)
        route_customers = [node for node in route if node != depot]
        route_demand = sum(demand[node] for node in route_customers)

        # Get working time for this route
        if len(route_customers) > 0:
            last_customer = route_customers[-1]
            working_time = R[last_customer].X

            # Calculate components
            driving_time = sum(t[route_arcs[k][0], route_arcs[k][1]] for k in range(len(route_arcs)))
            service_time_total = sum(service_time[cust] for cust in route_customers)
            waiting_time = working_time - driving_time - service_time_total
        else:
            working_time = 0
            driving_time = 0
            service_time_total = 0
            waiting_time = 0

        print(f"\nVehicle {vehicle_num}:")
        print(f"  Route: {' → '.join(map(str, route))}")
        print(f"  Customers visited: {len(route_customers)}")
        print(f"  Total demand: {route_demand:.1f} / {Q} (capacity: {route_demand / Q * 100:.1f}%)")
        print(f"  Route distance: {route_distance:.2f} km")
        print(f"  Working time: {working_time:.2f} min")
        print(f"    - Driving: {driving_time:.2f} min")
        print(f"    - Service: {service_time_total:.2f} min")
        print(f"    - Waiting: {waiting_time:.2f} min")

        if working_time > H + 0.1:
            print(f"  ⚠️ WARNING: Exceeds {H} min limit!")
        else:
            print(f"  ✅ Within {H} min limit")

        if len(route_customers) > 0:
            print(f"  Customer timeline:")
            for cust in route_customers:
                arrival = T[cust].X
                tw_start = ready_time[cust]
                tw_end = due_time[cust]
                wait_time = max(0, tw_start - arrival)
                print(
                    f"    Customer {cust}: arrive {arrival:.1f}, TW [{tw_start:.0f}, {tw_end:.0f}], wait {wait_time:.1f}")

        vehicle_num += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total distance: {m.objVal:.2f} km")
    print(f"Vehicles used: {num_vehicles}")
    print(f"Driver working time limit: {H} minutes (7 hours)")
    print(f"All routes satisfy working time constraint: ✅")
    print(f"Solution time: {solve_time:.2f} seconds")
    print("=" * 80)

elif m.status == GRB.INFEASIBLE:
    print("❌ MODEL IS INFEASIBLE")
    print("\nThe 7-hour working time limit may be too restrictive!")
    m.computeIIS()
    m.write("q_2_3_infeasible.ilp")
    print("IIS written to q_2_3_infeasible.ilp")

else:
    print(f"❌ OPTIMIZATION FAILED (Status: {m.status})")

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)
