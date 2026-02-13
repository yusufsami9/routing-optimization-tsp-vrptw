import gurobipy as gp
from gurobipy import GRB
import pandas as  pd
import math
import time


# QUESTION 2.1(c)

# same model with Extended MTZ for capacities and time windows (q_2_1_b) except the objective function



path = "C101_025.xlsx"
nodes_df = pd.read_excel(path, sheet_name="Nodes")
requests_df = pd.read_excel(path, sheet_name="Requests")
fleet_df = pd.read_excel(path, sheet_name="Fleet")


N = nodes_df["id"].tolist()

depot = 0
customers = [i for i in N if i!=0]

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
ready_time = {}      # a_i (earliest time)
due_time = {}        # b_i (latest time)
service_time = {}    # s_i (service duration)
demand[depot] = 0
ready_time[depot] = 0
service_time[depot] = 0

# Read fleet data to get depot's due time
max_travel_time = fleet_df.loc[0, "max_travel_time"]
due_time[depot] = max_travel_time

for _, row in requests_df.iterrows():
    node_id = int(row["node"])

    demand[node_id] = row["quantity"]
    ready_time[node_id] = row["start"]
    due_time[node_id] = row["end"]
    service_time[node_id] = row["service_time"]

# Vehicle capacity from Fleet sheet
Q = fleet_df.loc[0, "capacity"]
print(f"\nVehicle capacity (Q): {Q}")

# Calculate total demand
total_demand = sum(demand[i] for i in customers)
print(f"Total demand across all customers: {total_demand}")
print(f"Minimum vehicles needed (based on capacity): {math.ceil(total_demand / Q)}")



# big M
# #T_i+s_i+t_ij is at most b_i+s_i+t_ij therefore, setting M=b_i+s_i+t_ij  constraint becomes T_ij >=0
M = max(due_time.values()) + max(service_time.values()) + max(c.values())

m = gp.Model("VRPTW_MTZ")

# Set parameters
m.setParam('TimeLimit', 600)  # 10 minutes
m.setParam('OutputFlag', 1)   # Show solver output

# Decision variables

x = m.addVars(N, N, vtype=GRB.BINARY, name="x")
for i in N:
    m.addConstr(x[i, i] == 0, name=f"no_self_{i}")


u = m.addVars(customers, vtype=GRB.CONTINUOUS, lb=0, ub=Q, name="u")

T = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0, name="T")

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

# capacity constraints
num_capacity_constraints = 0
for i in customers:
    for j in customers:
        if i != j:
            m.addConstr(
                u[j] >= u[i] + demand[j] - Q * (1 - x[i, j]),
                name=f"load_{i}_{j}"
            )
            num_capacity_constraints += 1
# Load bounds: load at each customer must be at least their demand
for i in customers:
    m.addConstr(u[i] >= demand[i], name=f"load_lb_{i}")
    m.addConstr(u[i] <= Q, name=f"load_ub_{i}")

# TIME WINDOW CONSTRAINTS
# Time tracking: if we go from i to j, time at j = time at i + service + travel
num_time_constraints = 0
for i in N:
    for j in customers:
        if i != j:
            m.addConstr(
                T[j] >= T[i] + service_time[i] + t[i, j] - M * (1 - x[i, j]),
                name=f"time_{i}_{j}"
            )
            num_time_constraints += 1
# Time window bounds: arrival time must be within [ready_time, due_time]
for i in customers:
    m.addConstr(T[i] >= ready_time[i], name=f"ready_{i}")
    m.addConstr(T[i] <= due_time[i], name=f"due_{i}")

# DEPOT CONSTRAINTS
# Depot departure time is 0
m.addConstr(T[depot] == 0, name="depot_time")

m.update()

# New Objective Function
lambda_penalty = 10

m.setObjective(
    gp.quicksum(c[i, j] * x[i, j] for i in N for j in N) + lambda_penalty * gp.quicksum(x[depot,j] for j in customers),
    GRB.MINIMIZE
)
m.update()

print("\n" + "="*60)
print("SOLVING MODEL...")
print("="*60)

start_time = time.time()

# Solve!
m.optimize()
end_time = time.time()
solve_time = end_time - start_time

if m.status == GRB.OPTIMAL:
    print("✅ OPTIMAL SOLUTION FOUND!")
    print("=" * 60)
    print(f"Objective value (total distance): {m.objVal:.2f}")
    print(f"Solution time: {solve_time:.2f} seconds")

    # Count number of vehicles used
    num_vehicles = sum(1 for j in customers if x[depot, j].X > 0.5)
    print(f"Number of vehicles used: {num_vehicles}")
    print(f"Minimum vehicles needed (by capacity): {math.ceil(total_demand / Q)}")

    # Extract all used arcs
    used_arcs = [(i, j) for i in N for j in N if x[i, j].X > 0.5]
    print(f"\nTotal arcs used: {len(used_arcs)}")

    print("\n" + "=" * 60)
    print("VEHICLE ROUTES")
    print("=" * 60)

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
        route_demand = sum(demand[node] for node in route_customers)

        # Get time of last customer visit (before returning to depot)
        if len(route_customers) > 0:
            last_customer = route_customers[-1]
            completion_time = T[last_customer].X + service_time[last_customer]
        else:
            completion_time = 0

        print(f"\nVehicle {vehicle_num}:")
        print(f"  Route: {' → '.join(map(str, route))}")
        print(f"  Customers visited: {len(route_customers)}")
        print(f"  Total demand: {route_demand:.1f} / {Q} (capacity utilization: {route_demand / Q * 100:.1f}%)")
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
                    f"    Customer {cust}: arrive {arrival:.1f}, TW [{tw_start:.0f}, {tw_end:.0f}], wait {wait_time:.1f}, demand {demand[cust]:.0f}")

        vehicle_num += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total distance: {m.objVal:.2f}")
    print(f"Vehicles used: {num_vehicles}")
    print(f"Average distance per vehicle: {m.objVal / num_vehicles:.2f}")
    print(f"Total demand delivered: {total_demand:.1f}")
    print(f"Average load per vehicle: {total_demand / num_vehicles:.1f}")
    print(f"Solution time: {solve_time:.2f} seconds")
    print("=" * 60)

elif m.status == GRB.TIME_LIMIT:
    print("⏰ TIME LIMIT REACHED")
    print(f"Best solution found: {m.objVal:.2f}")
    print(f"Optimality gap: {m.MIPGap * 100:.2f}%")
    print(f"Time: {solve_time:.2f} seconds")

elif m.status == GRB.INFEASIBLE:
    print("❌ MODEL IS INFEASIBLE")
    print("No solution exists that satisfies all constraints!")
    print("\nPossible reasons:")
    print("  - Time windows too tight")
    print("  - Vehicle capacity too small")
    print("  - Conflicting constraints")

    # Compute IIS to help debug
    print("\nComputing IIS (Irreducible Inconsistent Subsystem)...")
    m.computeIIS()
    m.write("infeasible_model.ilp")
    print("IIS written to infeasible_model.ilp")

else:
    print(f"❌ OPTIMIZATION FAILED")
    print(f"Status code: {m.status}")
    print(f"Time: {solve_time:.2f} seconds")

print("\n" + "=" * 60)
print("OPTIMIZATION COMPLETE")
print("=" * 60)




