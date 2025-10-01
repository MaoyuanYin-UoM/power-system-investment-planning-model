"""
test_windstorm_opf_debug.py

Standalone script to debug why TN always has zero EENS in windstorm scenarios.
Builds and solves a single windstorm OPF model, then exports all variables to Excel.
"""

import json
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from factories.network_factory import make_network
from pathlib import Path
import numpy as np
from datetime import datetime


def export_model_to_excel(model, filepath="windstorm_opf_debug.xlsx"):
    """
    Export all model variables and key parameters to Excel for inspection.
    """
    print(f"\nExporting model to {filepath}...")

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:

        # 1. Model Info
        info_data = {
            'Item': ['Buses', 'TN Buses', 'DN Buses', 'Branches', 'Generators', 'Timesteps'],
            'Count': [
                len(model.Set_bus),
                len(model.Set_bus_tn),
                len(model.Set_bus_dn),
                len(model.Set_bch),
                len(model.Set_gen),
                len(model.Set_ts)
            ]
        }
        pd.DataFrame(info_data).to_excel(writer, sheet_name='Model_Info', index=False)

        # 2. Load Shedding (Pc) - Most Important
        pc_data = []
        for b in model.Set_bus:
            bus_level = 'TN' if b in model.Set_bus_tn else 'DN'
            for t in model.Set_ts:
                if (b, t) in model.Set_bt:
                    pc_data.append({
                        'Bus': b,
                        'Level': bus_level,
                        'Timestep': t,
                        'Demand_MW': pyo.value(model.Pd[b, t]),
                        'LoadShed_MW': pyo.value(model.Pc[b, t]),
                        'Shed_Pct': (pyo.value(model.Pc[b, t]) / pyo.value(model.Pd[b, t]) * 100)
                                   if pyo.value(model.Pd[b, t]) > 0 else 0
                    })

        df_pc = pd.DataFrame(pc_data)
        df_pc.to_excel(writer, sheet_name='Load_Shedding', index=False)

        # 3. Summary by Level
        summary_data = []
        for level in ['TN', 'DN']:
            level_df = df_pc[df_pc['Level'] == level]
            summary_data.append({
                'Level': level,
                'Total_Demand_MWh': level_df['Demand_MW'].sum(),
                'Total_LoadShed_MWh': level_df['LoadShed_MW'].sum(),
                'Avg_Shed_Pct': level_df['Shed_Pct'].mean(),
                'Buses_With_Shedding': (level_df['LoadShed_MW'] > 0.01).sum(),
                'Total_Bus_Hours': len(level_df)
            })

        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

        # 4. Branch Status (for lines only)
        branch_data = []
        for l in model.Set_bch_lines:
            branch_level = 'TN' if l in model.Set_bch_tn_lines else 'DN'
            for t in model.Set_ts:
                if (l, t) in model.Set_lt_lines:
                    branch_data.append({
                        'Branch': l,
                        'Level': branch_level,
                        'Timestep': t,
                        'Status': pyo.value(model.bch_status[l, t]),
                        'Fail_Prob': pyo.value(model.fail_prob[l, t]),
                        'Fail_Occurs': pyo.value(model.fail_occurs[l, t]),
                        'Wind_Speed': pyo.value(model.wind_gust_speed[t]),
                        'Impact_Flag': pyo.value(model.impact_flag[l, t])
                    })

        df_branch = pd.DataFrame(branch_data)
        df_branch.to_excel(writer, sheet_name='Branch_Status', index=False)

        # 5. Branch Failures Summary
        fail_summary = []
        for level in ['TN', 'DN']:
            level_branches = df_branch[df_branch['Level'] == level]
            fail_summary.append({
                'Level': level,
                'Total_Branch_Hours': len(level_branches),
                'Failed_Branch_Hours': (level_branches['Status'] < 0.5).sum(),
                'Failure_Rate': (level_branches['Status'] < 0.5).sum() / len(level_branches) * 100 if len(level_branches) > 0 else 0,
                'Max_Wind_Speed': level_branches['Wind_Speed'].max(),
                'Avg_Fail_Prob': level_branches['Fail_Prob'].mean()
            })

        pd.DataFrame(fail_summary).to_excel(writer, sheet_name='Failure_Summary', index=False)

        # 6. Generation (Pg)
        gen_data = []
        for g in model.Set_gen:
            gen_bus = model.data.net.gen[g-1]
            bus_level = 'TN' if gen_bus in model.Set_bus_tn else 'DN'
            for t in model.Set_ts:
                if (g, t) in model.Set_gt:
                    gen_data.append({
                        'Generator': g,
                        'Bus': gen_bus,
                        'Level': bus_level,
                        'Timestep': t,
                        'Pg_MW': pyo.value(model.Pg[g, t]),
                        'Pg_max': pyo.value(model.Pg_max[g])
                    })

        pd.DataFrame(gen_data).to_excel(writer, sheet_name='Generation', index=False)

        # 7. Power Flows (sample)
        flow_data = []

        # TN flows
        for l in model.Set_bch_tn[:min(20, len(model.Set_bch_tn))]:  # First 20 TN branches
            for t in model.Set_ts[:min(5, len(model.Set_ts))]:  # First 5 timesteps
                if (l, t) in [(l, t) for l in model.Set_bch_tn for t in model.Set_ts]:
                    flow_data.append({
                        'Branch': l,
                        'Level': 'TN',
                        'Timestep': t,
                        'Pf_MW': pyo.value(model.Pf_tn[l, t]),
                        'Pmax': pyo.value(model.Pmax_tn[l]) if l in model.Set_bch_tn else 0,
                        'Status': pyo.value(model.bch_status[l, t]) if (l, t) in model.Set_lt_lines else 1
                    })

        pd.DataFrame(flow_data).to_excel(writer, sheet_name='Power_Flows_Sample', index=False)

        # 8. Import/Export at Slack
        slack_data = []
        for t in model.Set_ts:
            slack_data.append({
                'Timestep': t,
                'Import_MW': pyo.value(model.Pimp[t]),
                'Export_MW': pyo.value(model.Pexp[t]),
                'Net_Import': pyo.value(model.Pimp[t]) - pyo.value(model.Pexp[t])
            })

        pd.DataFrame(slack_data).to_excel(writer, sheet_name='Slack_Bus', index=False)

    print(f"Export complete: {filepath}")

    # Print quick summary
    print("\nQUICK SUMMARY:")
    print(f"TN Load Shedding: {df_pc[df_pc['Level']=='TN']['LoadShed_MW'].sum():.2f} MWh")
    print(f"DN Load Shedding: {df_pc[df_pc['Level']=='DN']['LoadShed_MW'].sum():.2f} MWh")
    print(f"TN Branch Failures: {(df_branch[df_branch['Level']=='TN']['Status'] < 0.5).sum()} branch-hours")
    print(f"DN Branch Failures: {(df_branch[df_branch['Level']=='DN']['Status'] < 0.5).sum()} branch-hours")


def main():
    # Configuration
    network_preset = "29_bus_GB_transmission_network_with_Kearsley_GSP_group"
    ws_library_path = "../Scenario_Database/Scenarios_Libraries/Filtered_Scenario_Libraries/ws_library_29BusGB-KearsleyGSP_GB_1000scn_s10000_filt_b1_h1_buf15.json"
    scenario_to_test = "ws_0012"  # Change this to test different scenarios

    print("="*70)
    print("WINDSTORM OPF DEBUG TEST")
    print("="*70)

    # 1. Load network
    print(f"\n1. Loading network: {network_preset}")
    net = make_network(network_preset)

    # 2. Load scenario
    print(f"\n2. Loading scenario library: {ws_library_path}")
    with open(ws_library_path, 'r') as f:
        ws_library = json.load(f)

    scenarios = ws_library.get("scenarios", {})

    if scenario_to_test not in scenarios:
        print(f"ERROR: Scenario {scenario_to_test} not found!")
        print(f"Available scenarios: {list(scenarios.keys())[:10]}...")
        return

    scenario_data = scenarios[scenario_to_test]

    # Print scenario info
    print(f"\n3. Testing scenario: {scenario_to_test}")
    print(f"   Events: {len(scenario_data.get('events', []))}")
    if scenario_data.get('events'):
        first_event = scenario_data['events'][0]
        print(f"   First event: Hour {first_event['bgn_hr']}, Duration {first_event['duration']}h")
        print(f"   Max wind speed: {max(first_event['gust_speed']):.1f} m/s")

    # 3. Build model
    print(f"\n4. Building OPF model...")
    model = net.build_combined_opf_model_under_ws_scenarios(
        single_ws_scenario=scenario_data,
        scenario_probability=1.0
    )

    print(f"   Model built with {len(model.Set_bus)} buses, {model.num_timesteps} timesteps")

    # 4. Solve model
    print(f"\n5. Solving model...")
    eens = net.solve_combined_opf_model_under_ws_scenarios(
        model=model,
        solver_name="gurobi",
        mip_gap=1e-8,
        time_limit=300,
        write_xlsx=True,
        output_path=None,  # use default
    )

    print(f"   Solution EENS: {eens:.2f} MWh")

    # 5. Export to Excel
    timestamp = datetime.strftime("%Y%m%d_%H%M%S")
    output_file = f"debug_{scenario_to_test}_{network_preset}_{timestamp}.xlsx"
    export_model_to_excel(model, output_file)

    # 6. Additional diagnostics
    print("\n" + "="*70)
    print("DETAILED DIAGNOSTICS")
    print("="*70)

    # Check for TN branch failures
    tn_branch_failures = 0
    for l in model.Set_bch_tn_lines:
        for t in model.Set_ts:
            if (l, t) in model.Set_lt_lines:
                if pyo.value(model.bch_status[l, t]) < 0.5:
                    tn_branch_failures += 1
                    if tn_branch_failures <= 3:  # Show first few
                        print(f"TN Branch {l} failed at timestep {t}")

    print(f"\nTotal TN branch failures: {tn_branch_failures}")

    # Check TN generation vs demand
    total_tn_demand = sum(pyo.value(model.Pd[b, t])
                         for b in model.Set_bus_tn
                         for t in model.Set_ts)
    total_tn_gen = sum(pyo.value(model.Pg[g, t])
                      for g in model.Set_gen
                      if model.data.net.gen[g-1] in model.Set_bus_tn
                      for t in model.Set_ts)
    total_import = sum(pyo.value(model.Pimp[t]) for t in model.Set_ts)

    print(f"\nTN Power Balance:")
    print(f"  Total TN Demand: {total_tn_demand:.2f} MWh")
    print(f"  Total TN Generation: {total_tn_gen:.2f} MWh")
    print(f"  Total Import at Slack: {total_import:.2f} MWh")
    print(f"  Balance: {total_tn_gen + total_import - total_tn_demand:.2f} MWh")

    print(f"\nResults exported to: {output_file}")
    print("Please check the Excel file for detailed variable values.")


if __name__ == "__main__":
    main()