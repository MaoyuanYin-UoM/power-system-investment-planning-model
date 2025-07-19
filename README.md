# Power System Resilience Planning

A Python framework for multi-stage stochastic MILP modeling of power system resilience-oriented strategic planning under multi-dimensional uncertainties.

## Project Overview

This project implements a multi-stage stochastic mixed-integer linear programming (MILP) model for power system resilience enhancement planning, considering:
- Windstorm scenario generation using spatial-temporal Monte Carlo simulation
- Resilience enhancement investment decisions for line hardening and DER installations
- Operational flexibility from distributed energy resources
- Integration of transmission and distribution network models

## Project Structure

```
power_system_resilience_planning_under_uncertainties/
│
├── core/                          # Core modeling components
│   ├── __init__.py
│   ├── config.py                  # Configuration classes
│   ├── investment_model.py        # Investment optimization model
│   ├── network.py                 # Network modeling and power flow
│   └── windstorm.py               # Windstorm simulation
│
├── data_processing/               # Data reading and processing, scenario generation, analysis
│   ├── __init__.py
│   ├── compute_baseline_yearly_cost.py
│   ├── compute_line_length.py
│   ├── fine_tune_bus_coordinates.py
│   ├── normalize_demand_profile.py
│   ├── scenario_generation_model.py
│   ├── select_and_combine_ws_scenarios.py
│   └── set_dfes_data.py
│
├── factories/                     # Factory pattern implementations
│   ├── __init__.py
│   ├── network_factory.py         # Network creation factory
│   └── windstorm_factory.py       # Windstorm creation factory
│
├── tests/                        # Test scripts (for debugging purposes; ignore this if you are a user)
│   ├── __init__.py
│   ├── temp.py
│   ├── test.py
│   ├── test_network_modelling.py
│   └── verify_OPF.py
│
├── utils.py                      # Utility functions
├── visualization.py              # Visualization utilities
├── main.py                       # Main execution script
│
├── Input_Data/                   # Input data directory
├── Scenario_Results/             # Scenario results
├── Optimization_Results/         # Optimization results
│
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
└── .gitignore                    # Git ignore file
```

## LastEditDate

19/July/2025