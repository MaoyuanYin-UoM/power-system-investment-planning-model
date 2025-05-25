import os
import json
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from network import NetworkClass
from config import NetConfig


def build_full_year_dc_opf_model(
    demand_profile_file="Demand_Profile/normalized_hourly_demand_profile_year.xlsx"
):
    """
    Build a Pyomo model for full‐year DC‐OPF:
      - hours 1..T form a time index
      - demand at hour t = normalized_profile[t] * max_demand per bus
    """
    # 1) Read normalized hourly profile
    df = pd.read_excel(demand_profile_file, header=0)
    norm_profile = df.iloc[:, 0].tolist()
    num_hrs = len(norm_profile)
    if num_hrs not in (8760, 720, 168):
        raise ValueError(f"Expected 8760/720/168 hours, got {num_hrs}")

    # 2) Initialize network (loads branch, gen data; also sets demand_profile_active if you use it)
    net = NetworkClass()

    # 3) Create model and index sets
    model = pyo.ConcreteModel()
    model.Set_bus = pyo.Set(initialize=net.data.net.bus)
    model.Set_bch = pyo.Set(initialize=range(1, len(net.data.net.bch) + 1))
    model.Set_gen = pyo.Set(initialize=range(1, len(net.data.net.gen) + 1))
    model.Set_ts  = pyo.Set(initialize=range(1, num_hrs + 1))

    # 4) Parameters
    # 4.1) demand per (bus, ts)
    demand_dict = {
        (b, t): norm_profile[t-1] * net.data.net.max_demand_active[b-1]
        for b in net.data.net.bus
        for t in model.Set_ts
    }
    model.demand = pyo.Param(
        model.Set_bus, model.Set_ts,
        initialize=demand_dict,
        mutable=False
    )

    # 4.2) generator cost coefficients (a + b·P)
    coef_len = len(net.data.net.gen_cost_coef[0])
    model.gen_cost_coef = pyo.Param(
        model.Set_gen, range(coef_len),
        initialize={
            (i+1, j): net.data.net.gen_cost_coef[i][j]
            for i in range(len(net.data.net.gen_cost_coef))
            for j in range(coef_len)
        },
        mutable=False
    )

    # 4.3) generation capacity bounds
    model.gen_active_max = pyo.Param(
        model.Set_gen,
        initialize={i+1: v for i, v in enumerate(net.data.net.gen_active_max)},
        mutable=False
    )
    model.gen_active_min = pyo.Param(
        model.Set_gen,
        initialize={i+1: v for i, v in enumerate(net.data.net.gen_active_min)},
        mutable=False
    )

    # 4.4) branch susceptance and thermal limits
    model.bch_B = pyo.Param(
        model.Set_bch,
        initialize={i+1: 1.0/net.data.net.bch_X[i] for i in range(len(net.data.net.bch_X))},
        mutable=False
    )
    model.bch_cap = pyo.Param(
        model.Set_bch,
        initialize={i+1: net.data.net.bch_cap[i] for i in range(len(net.data.net.bch_cap))},
        mutable=False
    )

    # 5) Decision variables
    model.P_gen  = pyo.Var(model.Set_gen,  model.Set_ts, within=pyo.NonNegativeReals)
    model.theta  = pyo.Var(model.Set_bus,  model.Set_ts, within=pyo.Reals)
    model.P_flow = pyo.Var(model.Set_bch,  model.Set_ts, within=pyo.Reals)
    model.load_shed = pyo.Var(model.Set_bus, model.Set_ts, within=pyo.NonNegativeReals)

    # 6) Constraints
    # 6.1) Slack bus: zero angle
    slack = net.data.net.slack_bus
    def slack_rule(m, t):
        return m.theta[slack, t] == 0
    model.Constraint_SlackBus = pyo.Constraint(model.Set_ts, rule=slack_rule)

    # 6.2) Power balance at each bus & timestep
    def power_balance_rule(m, b, t):
        gen_out = sum(m.P_gen[g, t]
                      for g in m.Set_gen
                      if net.data.net.gen[g-1] == b)
        inflow  = sum(m.P_flow[l, t]
                      for l in m.Set_bch
                      if net.data.net.bch[l-1][1] == b)
        outflow = sum(m.P_flow[l, t]
                      for l in m.Set_bch
                      if net.data.net.bch[l-1][0] == b)
        return gen_out + inflow - outflow + m.load_shed[b, t] == m.demand[b, t]
    model.Constraint_PowerBalance = pyo.Constraint(
        model.Set_bus, model.Set_ts,
        rule=power_balance_rule
    )

    # 6.3) DC flow definition
    def flow_definition_rule(m, l, t):
        i, j = net.data.net.bch[l-1]
        return m.P_flow[l, t] == m.bch_B[l] * (m.theta[i, t] - m.theta[j, t])
    model.Constraint_FlowDefinition = pyo.Constraint(
        model.Set_bch, model.Set_ts,
        rule=flow_definition_rule
    )

    # 6.4) Thermal limits
    def flow_limit_upper_rule(m, l, t):
        return m.P_flow[l, t] <= m.bch_cap[l]
    def flow_limit_lower_rule(m, l, t):
        return m.P_flow[l, t] >= -m.bch_cap[l]
    model.Constraint_FlowLimit_Upper = pyo.Constraint(
        model.Set_bch, model.Set_ts, rule=flow_limit_upper_rule
    )
    model.Constraint_FlowLimit_Lower = pyo.Constraint(
        model.Set_bch, model.Set_ts, rule=flow_limit_lower_rule
    )

    # 6.5) Generation limits
    def gen_upper_limit_rule(m, g, t):
        return m.P_gen[g, t] <= m.gen_active_max[g]
    def gen_lower_limit_rule(m, g, t):
        return m.P_gen[g, t] >= m.gen_active_min[g]
    model.Constraint_GenUpperLimit = pyo.Constraint(
        model.Set_gen, model.Set_ts, rule=gen_upper_limit_rule
    )
    model.Constraint_GenLowerLimit = pyo.Constraint(
        model.Set_gen, model.Set_ts, rule=gen_lower_limit_rule
    )


    # 8) objective function: gen cost + load_shed cost

    # get load‐shedding cost constants
    net_conf = NetConfig()
    # cost_bus_ls is per‐bus list in config; convert to a Param
    cost_ls_dict = {b: net_conf.data.cost_bus_ls[b - 1]
                    for b in net.data.net.bus}
    model.cost_load_shed = pyo.Param(model.Set_bus,
                                     initialize=cost_ls_dict,
                                     mutable=False)

    def total_cost_rule(m):
        gen_cost = sum(
            m.gen_cost_coef[g, 0]
            + m.gen_cost_coef[g, 1] * m.P_gen[g, t]
            for g in m.Set_gen for t in m.Set_ts
        )
        ls_cost = sum(
            m.cost_load_shed[b] * m.load_shed[b, t]
            for b in m.Set_bus for t in m.Set_ts
        )
        return gen_cost + ls_cost

    model.Objective = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    return model


def run_full_year_dc_opf(
    solver: str = "gurobi",
    out_dir: str = "Scenario_Results",
    out_file: str = "normal_scenario_annual_cost.json"
) -> float:
    """
    Build, solve, and return the annual DC‐OPF cost.
    """
    # 1) Build model
    model = build_full_year_dc_opf_model()

    # 2) Solve
    opt   = SolverFactory(solver)
    results = opt.solve(model, tee=False)
    status  = results.solver.status
    term    = results.solver.termination_condition

    # 3) Check solver
    if not (status == SolverStatus.ok and term == TerminationCondition.optimal):
        raise RuntimeError(f"DC‐OPF failed: {status}, {term}")

    # 4) Extract cost
    annual_cost = pyo.value(model.Objective)
    print(f"Total annual DC‐OPF cost = {annual_cost:.2f}")

    # 5) Write to JSON
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, out_file)
    with open(path, "w") as f:
        json.dump({"annual_operation_cost": annual_cost}, f, indent=4)
    print(f"Saved annual cost to {path}")

    return annual_cost