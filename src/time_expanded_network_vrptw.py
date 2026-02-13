import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import math
import time

# ============================================================================
# Question 2.2(e): Time-Expanded Network Implementation
# ============================================================================

# PARAMETERS
DELTA_T = 1
WINDOW_MULTIPLIER = 0.5  # Window width multiplier

print(f"\n{'=' * 80}")
print(f"TIME-EXPANDED NETWORK FORMULATION")
print(f"Discretization step: Δt = {DELTA_T}")
print(f"Window multiplier: λ = {WINDOW_MULTIPLIER}")
print(f"{'=' * 80}\n")



path = "C101_025.xlsx"

print("Loading data...")
nodes_df = pd.read_excel(path, sheet_name="Nodes")
requests_df = pd.read_excel(path, sheet_name="Requests")
fleet_df = pd.read_excel(path, sheet_name="Fleet")

N = nodes_df["id"].tolist()
depot = 0
customers = [i for i in N if i != 0]

print(f"Nodes: {len(N)} (1 depot + {len(customers)} customers)")


coords = {}
for _, row in nodes_df.iterrows():
    coords[row["id"]] = (row["cx"], row["cy"])

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

tau = c.copy()  # Travel time = distance


# TIME WINDOWS AND SERVICE TIMES


demand = {}
ready_time = {}
due_time = {}
service_time = {}

# Depot
demand[depot] = 0
ready_time[depot] = 0
service_time[depot] = 0

max_travel_time = fleet_df.loc[0, "max_travel_time"]
due_time[depot] = max_travel_time * WINDOW_MULTIPLIER

# Customers (with window adjustment)
for _, row in requests_df.iterrows():
    node_id = int(row["node"])
    demand[node_id] = row["quantity"]

    # Adjust time windows based on multiplier
    original_ready = row["start"]
    original_due = row["end"]
    original_width = original_due - original_ready
    center = (original_ready + original_due) / 2
    new_width = original_width * WINDOW_MULTIPLIER

    ready_time[node_id] = center - new_width / 2
    due_time[node_id] = center + new_width / 2
    service_time[node_id] = row["service_time"]

print(f"\nTime window info:")
print(f"  Depot max time: {due_time[depot]:.1f}")
avg_width = sum(due_time[i] - ready_time[i] for i in customers) / len(customers)
print(f"  Average customer window width: {avg_width:.1f}")

# ============================================================================
# CREATE TIME PERIODS
# ============================================================================

T_max = int(due_time[depot])
T = list(range(0, T_max + DELTA_T, DELTA_T))

print(f"\nTime discretization:")
print(f"  Time periods: {len(T)}")
print(f"  Range: [0, {T_max}]")
print(f"  Step: {DELTA_T}")

# ============================================================================
# PRE-CHECK: Verify all customers can be served
# ============================================================================

print(f"\nChecking temporal feasibility...")
infeasible_customers = []
for i in customers:
    # Check if customer window overlaps with time horizon
    if ready_time[i] >= T_max:
        infeasible_customers.append(i)
        print(
            f"  ⚠️ Customer {i}: Window [{ready_time[i]:.1f}, {due_time[i]:.1f}] starts after depot closes at {T_max}")
    elif due_time[i] < 0:
        infeasible_customers.append(i)
        print(f"  ⚠️ Customer {i}: Window [{ready_time[i]:.1f}, {due_time[i]:.1f}] ends before depot opens")

if infeasible_customers:
    print(f"\n{'=' * 80}")
    print("❌ MODEL IS INFEASIBLE BY CONSTRUCTION")
    print(f"{'=' * 80}")
    print(f"\n{len(infeasible_customers)} customers have time windows outside the feasible time horizon [0, {T_max}]:")
    print(f"Infeasible customers: {infeasible_customers}")
    print(f"\nReason: With window multiplier λ = {WINDOW_MULTIPLIER}, narrow windows push")
    print(f"some customers' time windows completely outside the depot's operating hours.")
    print(f"\nSuggestions:")
    print(f"  - Use wider time windows (larger λ, e.g., λ = 1.0 or 2.0)")
    print(f"  - Adjust depot operating hours")
    print(f"\n{'=' * 80}")
    print("STOPPING - MODEL NOT SOLVED")
    print(f"{'=' * 80}\n")
    exit(0)

print(f"✅ All customers have windows within time horizon")

# ============================================================================
# CREATE FEASIBLE ARC SET
# ============================================================================

print(f"\nBuilding feasible arc set...")
A = []

for i in N:
    for j in N:
        if i == j:
            continue

        for t in T:
            feasible = False

            # Case 1: Arc to customer j
            if j in customers:
                # Must arrive within time window
                if ready_time[j] <= t <= due_time[j]:
                    if i == depot:
                        # From depot: vehicle leaves at time 0
                        # Arrives at time tau[depot, j]
                        # Allow arrival at any discretized time >= travel time
                        travel_time = tau[(depot, j)]
                        if t >= travel_time:
                            feasible = True
                    else:
                        # From another customer: timing validated by C4
                        feasible = True

            # Case 2: Arc to depot
            elif j == depot:
                if t <= T_max:
                    feasible = True

            if feasible:
                A.append((i, j, t))

print(f"Arcs created: {len(A)}")

# Count arc types for debugging
depot_to_customer = sum(1 for (i, j, t) in A if i == depot and j in customers)
customer_to_customer = sum(1 for (i, j, t) in A if i in customers and j in customers)
customer_to_depot = sum(1 for (i, j, t) in A if i in customers and j == depot)

print(f"  Depot → Customer: {depot_to_customer}")
print(f"  Customer → Customer: {customer_to_customer}")
print(f"  Customer → Depot: {customer_to_depot}")

# ============================================================================
# CREATE MODEL
# ============================================================================

print(f"\nBuilding Gurobi model...")
m = gp.Model("TimeExpandedVRPTW")
m.setParam('TimeLimit', 600)
m.setParam('OutputFlag', 1)
m.setParam('MIPGap', 0.01)  # 1% gap tolerance

# ============================================================================
# DECISION VARIABLES
# ============================================================================

print("Creating variables...")

# x[i,j,t]: arc variables
x = m.addVars(A, vtype=GRB.BINARY, name="x")

# y[i,t]: service start variables (only for valid time windows)
y_indices = [(i, t) for i in customers for t in T
             if ready_time[i] <= t <= due_time[i]]
y = m.addVars(y_indices, vtype=GRB.BINARY, name="y")

m.update()

print(f"Variables created:")
print(f"  Arc variables (x): {len(A)}")
print(f"  Service variables (y): {len(y_indices)}")
print(f"  Total: {len(A) + len(y_indices)}")

# Verify all customers have at least one y variable
customers_with_y = set(i for (i, t) in y_indices)
print(f"\nCustomers with service variables: {len(customers_with_y)}/{len(customers)}")
if len(customers_with_y) < len(customers):
    missing = set(customers) - customers_with_y
    print(f"⚠️ ERROR: Customers without service variables: {missing}")
    print(f"⚠️ This should not happen after pre-check!")

# ============================================================================
# CONSTRAINTS
# ============================================================================

print(f"\nAdding constraints...")
constraint_count = 0

# C1: Each customer served exactly once
print("  C1: Visit each customer once...")
for i in customers:
    valid_times = [t for t in T if ready_time[i] <= t <= due_time[i]]
    if valid_times:
        m.addConstr(
            gp.quicksum(y[(i, t)] for t in valid_times if (i, t) in y_indices) == 1,
            name=f"visit_{i}"
        )
        constraint_count += 1
    else:
        # This should not happen after pre-check, but add safety constraint
        print(f"  ⚠️ Customer {i} has no valid times - adding infeasibility constraint")
        m.addConstr(0 == 1, name=f"impossible_{i}")
        constraint_count += 1

# C3: Arrival triggers service
print("  C3: Arrival → service start...")
for i in customers:
    for t in T:
        if (i, t) in y_indices:
            incoming_arcs = [(j, i, t) for j in N if (j, i, t) in A]
            if incoming_arcs:
                m.addConstr(
                    gp.quicksum(x[arc] for arc in incoming_arcs) == y[(i, t)],
                    name=f"arrival_{i}_{t}"
                )
                constraint_count += 1

# C4: Departure after service (with travel time)
print("  C4: Departure with travel time...")
for i in customers:
    for t in T:
        if (i, t) in y_indices:
            # Find outgoing arcs at correct departure time
            outgoing_arcs = []
            for k in N:
                # Exact departure time
                exact_departure = t + service_time[i] + tau[(i, k)]

                # Find closest time period >= exact departure
                best_t_dep = None
                min_diff = float('inf')
                for t_dep in T:
                    if t_dep >= exact_departure:
                        diff = t_dep - exact_departure
                        if diff < min_diff:
                            min_diff = diff
                            best_t_dep = t_dep

                # If found valid departure time and arc exists
                if best_t_dep is not None and (i, k, best_t_dep) in A:
                    outgoing_arcs.append((i, k, best_t_dep))

            if outgoing_arcs:
                m.addConstr(
                    gp.quicksum(x[arc] for arc in outgoing_arcs) == y[(i, t)],
                    name=f"depart_{i}_{t}"
                )
                constraint_count += 1

# C5a: At least one vehicle departs
print("  C5a: At least one vehicle...")
depot_departures = [(depot, j, t) for j in customers for t in T
                    if (depot, j, t) in A]
if depot_departures:
    m.addConstr(gp.quicksum(x[arc] for arc in depot_departures) >= 1,
                name="min_vehicles")
    constraint_count += 1
else:
    print("  ⚠️ NO DEPOT DEPARTURES! Adding infeasibility constraint")
    m.addConstr(0 == 1, name="no_depot_arcs")
    constraint_count += 1

# C5b: Depot balance
print("  C5b: Depot balance...")
depot_returns = [(i, depot, t) for i in customers for t in T
                 if (i, depot, t) in A]
if depot_departures and depot_returns:
    m.addConstr(
        gp.quicksum(x[arc] for arc in depot_departures) ==
        gp.quicksum(x[arc] for arc in depot_returns),
        name="depot_balance"
    )
    constraint_count += 1

m.update()
print(f"Total constraints: {constraint_count}")

# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

print(f"\nSetting objective...")
m.setObjective(
    gp.quicksum(c[(i, j)] * x[(i, j, t)] for (i, j, t) in A),
    GRB.MINIMIZE
)

# ============================================================================
# SOLVE
# ============================================================================

print(f"\n{'=' * 80}")
print("SOLVING MODEL...")
print(f"{'=' * 80}\n")

start_time = time.time()
m.optimize()
solve_time = time.time() - start_time

# ============================================================================
# RESULTS
# ============================================================================

print(f"\n{'=' * 80}")
print("RESULTS")
print(f"{'=' * 80}\n")

if m.status == GRB.OPTIMAL:
    print("✅ OPTIMAL SOLUTION FOUND")
    print(f"Objective value: {m.objVal:.2f}")
    print(f"Solution time: {solve_time:.2f} seconds")

    # Count vehicles
    num_vehicles = sum(1 for (i, j, t) in A
                       if i == depot and j in customers and x[(i, j, t)].X > 0.5)
    print(f"Number of vehicles: {num_vehicles}")

    # Count customers served
    served_customers = set()
    for (i, t) in y_indices:
        if y[(i, t)].X > 0.5:
            served_customers.add(i)
    print(f"Customers served: {len(served_customers)}/{len(customers)}")

    # Sanity check
    if len(served_customers) < len(customers):
        print(f"\n⚠️ WARNING: Not all customers served!")
        print(f"Missing customers: {set(customers) - served_customers}")
        print(f"This indicates a bug in the model!")

    # Extract routes
    print(f"\n{'=' * 80}")
    print("ROUTES")
    print(f"{'=' * 80}")

    used_arcs = [(i, j, t) for (i, j, t) in A if x[(i, j, t)].X > 0.5]

    # Find routes starting from depot
    remaining_arcs = set(used_arcs)
    vehicle_num = 1
    total_route_distance = 0

    while any(i == depot for (i, j, t) in remaining_arcs):
        # Find depot departure
        depot_arc = next((arc for arc in remaining_arcs if arc[0] == depot), None)
        if not depot_arc:
            break

        route = [depot]
        current = depot_arc[1]  # First customer
        route.append(current)
        remaining_arcs.remove(depot_arc)

        # Follow route
        while current != depot:
            next_arc = next((arc for arc in remaining_arcs if arc[0] == current), None)
            if not next_arc:
                break
            remaining_arcs.remove(next_arc)
            current = next_arc[1]
            route.append(current)

        # Calculate route distance
        route_distance = sum(c[(route[k], route[k + 1])] for k in range(len(route) - 1))
        total_route_distance += route_distance

        print(f"\nVehicle {vehicle_num}:")
        print(f"  Route: {' → '.join(map(str, route))}")
        print(f"  Customers: {len(route) - 2}")
        print(f"  Distance: {route_distance:.2f}")

        vehicle_num += 1

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Configuration:")
    print(f"  Window multiplier (λ): {WINDOW_MULTIPLIER}")
    print(f"  Discretization (Δt): {DELTA_T}")
    print(f"  Time horizon: [0, {T_max}]")
    print(f"\nResults:")
    print(f"  Total distance: {m.objVal:.2f} km")
    print(f"  Route distance sum: {total_route_distance:.2f} km")
    print(f"  Vehicles used: {num_vehicles}")
    print(f"  Customers served: {len(served_customers)}/{len(customers)}")
    print(f"  Solution time: {solve_time:.2f} seconds")
    print(f"  MIP gap: {m.MIPGap * 100:.2f}%")
    print(f"{'=' * 80}")

elif m.status == GRB.TIME_LIMIT:
    print(" TIME LIMIT REACHED")
    if m.SolCount > 0:
        print(f"Best solution found: {m.objVal:.2f}")
        print(f"MIP gap: {m.MIPGap * 100:.2f}%")

        num_vehicles = sum(1 for (i, j, t) in A
                           if i == depot and j in customers and x[(i, j, t)].X > 0.5)
        print(f"Vehicles: {num_vehicles}")

        # Count customers served
        served_customers = set()
        for (i, t) in y_indices:
            if y[(i, t)].X > 0.5:
                served_customers.add(i)
        print(f"Customers served: {len(served_customers)}/{len(customers)}")
    else:
        print("No solution found within time limit")
    print(f"Solution time: {solve_time:.2f} seconds")

elif m.status == GRB.INFEASIBLE:
    print(" MODEL IS INFEASIBLE")
    print("\nThis means the time-expanded formulation with the given discretization")
    print("cannot represent a feasible solution that satisfies all constraints.")
    print("\nPossible reasons:")
    print("  1. Discretization too coarse (Δt too large)")
    print("  2. Time windows too tight (λ too small)")
    print("  3. Temporal constraints cannot be satisfied with discrete time grid")
    print("\nDiagnostics:")
    print(f"  Window multiplier (λ): {WINDOW_MULTIPLIER}")
    print(f"  Discretization (Δt): {DELTA_T}")
    print(f"  Depot arcs: {depot_to_customer}")
    print(f"  Time periods: {len(T)}")
    print(f"  Time horizon: [0, {T_max}]")
    print(f"  Customers with y variables: {len(customers_with_y)}/{len(customers)}")

    print("\nComputing IIS (Irreducible Inconsistent Subsystem)...")
    m.computeIIS()
    ilp_file = f"infeasible_lambda_{WINDOW_MULTIPLIER}_dt_{DELTA_T}.ilp"
    m.write(ilp_file)
    print(f"IIS written to: {ilp_file}")

else:
    print(f"❌ SOLVER STATUS: {m.status}")
    print(f"Status codes: LOADED=1, OPTIMAL=2, INFEASIBLE=3, INF_OR_UNBD=4,")
    print(f"              UNBOUNDED=5, CUTOFF=6, ITERATION_LIMIT=7, NODE_LIMIT=8,")
    print(f"              TIME_LIMIT=9, SOLUTION_LIMIT=10, INTERRUPTED=11,")
    print(f"              NUMERIC=12, SUBOPTIMAL=13, INPROGRESS=14, USER_OBJ_LIMIT=15")
    print(f"Solution time: {solve_time:.2f} seconds")

print(f"\n{'=' * 80}")
print("COMPLETE")
print(f"{'=' * 80}\n")
