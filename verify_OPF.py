import pandapower as pp

def create_dc_opf_network():
    """Create a pandapower network for DC OPF"""
    net = pp.create_empty_network()

    # Add buses
    for i in range(5):  # Buses: 1-5
        pp.create_bus(net, vn_kv=110, name=f"Bus {i+1}")

    # Add lines (branches)
    branches = [
        {"from_bus": 0, "to_bus": 1, "x_ohm_per_km": 0.00281, "max_i_ka": 0.4},
        {"from_bus": 0, "to_bus": 3, "x_ohm_per_km": 0.00304, "max_i_ka": 0.4},
        {"from_bus": 0, "to_bus": 4, "x_ohm_per_km": 0.00064, "max_i_ka": 0.4},
        {"from_bus": 1, "to_bus": 2, "x_ohm_per_km": 0.00108, "max_i_ka": 0.4},
        {"from_bus": 2, "to_bus": 3, "x_ohm_per_km": 0.00297, "max_i_ka": 0.4},
        {"from_bus": 3, "to_bus": 4, "x_ohm_per_km": 0.00297, "max_i_ka": 0.4},
    ]
    for branch in branches:
        pp.create_line_from_parameters(
            net,
            from_bus=branch["from_bus"],
            to_bus=branch["to_bus"],
            length_km=1,
            r_ohm_per_km=0,  # DC assumption: R = 0
            x_ohm_per_km=branch["x_ohm_per_km"],
            c_nf_per_km=0,  # Ignore shunt capacitance
            max_i_ka=branch["max_i_ka"]
        )

    # Add generators
    generators = [
        {"bus": 0, "p_mw": 0, "vm_pu": 1.0, "max_p_mw": 200, "min_p_mw": 0, "cost": 10},
        {"bus": 1, "p_mw": 0, "vm_pu": 1.0, "max_p_mw": 150, "min_p_mw": 0, "cost": 15},
        {"bus": 2, "p_mw": 0, "vm_pu": 1.0, "max_p_mw": 300, "min_p_mw": 0, "cost": 20},
        {"bus": 3, "p_mw": 0, "vm_pu": 1.0, "max_p_mw": 100, "min_p_mw": 0, "cost": 25},
        {"bus": 4, "p_mw": 0, "vm_pu": 1.0, "max_p_mw": 250, "min_p_mw": 0, "cost": 5},
    ]
    for gen in generators:
        pp.create_gen(
            net,
            bus=gen["bus"],
            p_mw=gen["p_mw"],
            vm_pu=gen["vm_pu"],  # DC: Voltage fixed to 1.0 p.u.
            max_p_mw=gen["max_p_mw"],
            min_p_mw=gen["min_p_mw"]
        )

    # Add polynomial costs
    for i, gen in enumerate(generators):
        pp.create_poly_cost(
            net,
            element=i,
            et="gen",
            cp1_eur_per_mw=gen["cost"],
            cp2_eur_per_mw2=0,
            cp0_eur=0
        )

    # Add loads
    loads = [
        {"bus": 0, "p_mw": 300},
        {"bus": 1, "p_mw": 200},
        {"bus": 2, "p_mw": 0},
        {"bus": 3, "p_mw": 150},
        {"bus": 4, "p_mw": 0},
    ]
    for load in loads:
        pp.create_load(net, bus=load["bus"], p_mw=load["p_mw"])

    # Add slack bus (ext grid) and cost for the grid
    pp.create_ext_grid(net, bus=2, vm_pu=1.0, va_degree=0.0)
    net.ext_grid.loc[0, "min_p_mw"] = 0  # Prevent exporting power
    net.ext_grid.loc[0, "max_p_mw"] = 1000  # Allow up to 1000 MW import
    pp.create_poly_cost(
        net,
        element=0,  # External grid index
        et="ext_grid",
        cp1_eur_per_mw=1000,  # Cost per MW from the external grid
        cp2_eur_per_mw2=0,
        cp0_eur=0
    )

    return net


def solve_dc_opf(net):
    """Solve DC OPF using pandapower"""
    pp.runopp(net)  # Correct method to run optimal power flow
    return net


def print_opf_results(net):
    """Print results of DC OPF"""
    print("\n=== DC Optimal Power Flow Results ===")
    print("\nBus Voltages (pu and angles):")
    print(net.res_bus[["vm_pu", "va_degree"]])
    print("\nGenerator Outputs (MW):")
    print(net.res_gen[["p_mw"]])
    print("\nLine Flows (MW):")
    print(net.res_line[["p_from_mw", "p_to_mw"]])

    print("\nTotal Generation Cost ($):")
    print(f"{net.res_cost:.2f}")

    # # Calculate and display total generation cost manually
    # total_cost = sum(net.res_gen.loc[i, "p_mw"] * gen["cost"] for i, gen in enumerate(generators))
    # print("\nManually Calculated Total Generation Cost ($):")
    # print(f"{total_cost:.2f}")



# Create and solve the network
net = create_dc_opf_network()
net = solve_dc_opf(net)

# Print the results
print_opf_results(net)
