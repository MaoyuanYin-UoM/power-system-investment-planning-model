# 2. Parameters:
model.demand = pyo.Param(model.Set_bus, initialize={i + 1: d for i, d in
                                                    enumerate(net.data.net.demand_active)})

model.gen_cost_coef = pyo.Param(model.Set_gen, range(len(net.data.net.gen_cost_coef[0])),
                                initialize={
                                    (i + 1, j): self.data.net.gen_cost_coef[i][j]
                                    for i in range(len(net.data.net.gen_cost_coef))
                                    for j in range(len(net.data.net.gen_cost_coef[i]))})

model.gen_active_max = pyo.Param(model.Set_gen, initialize={i + 1: g for i, g in
                                                            enumerate(net.data.net.gen_active_max)})

model.gen_active_min = pyo.Param(model.Set_gen, initialize={i + 1: g for i, g in
                                                            enumerate(net.data.net.gen_active_min)})

model.bch_cap = pyo.Param(model.Set_bch, initialize={i + 1: bc for i, bc in
                                                     enumerate(net.data.net.bch_cap)})

# calculate susceptance B for each line (under the assumption of DC power flow)
model.bch_B = pyo.Param(model.Set_bch, initialize={i + 1: 1 / X for i, X in
                                                   enumerate(net.data.net.bch_X)})