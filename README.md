# Power System Resilience Planning

A Python framework for multi-stage stochastic MILP modeling of power system resilience-oriented strategic planning under multi-dimensional uncertainties.

## Author

Maoyuan Yin\
PhD Student\
University of Manchester\
maoyuan.yin@postgrad.manchester.ac.uk

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
├── core/                           # Core modeling components
│   ├── __init__.py
│   ├── config.py                           # Stores default instances for the core classes
│   ├── investment_model_multi_stage.py     # The multi-stage stochastic MILP investment planning model
│   ├── investment_model_two_stage.py       # The old two-stage stochastic MILP investment planning model
│   ├── network.py                          # Core classes and methods for network modeling and power flow
│   ├── scenario_tree.py                    # Core classes and methods for scenario tree construction
│   └── windstorm.py                        # Core classes and methods for windstorm simulation
│
├── data_processing/                # Scripts or modules for data reading and processing, scenario generation etc.
│   ├── __init__.py
│   ├── compute_baseline_yearly_cost.py                 #
│   ├── compute_line_length.py                          #
│   ├── dfes_data_processor.py                          # Process the DFES data workbook for creating a scenario tree based on DFES pathways
│   ├── fine_tune_bus_coordinates.py                    #
│   ├── normalize_demand_profile.py                     #
│   ├── scenario_generation_for_multi_stage_model.py    # Script to generate candidate scenarios for the multi-stage model 
│   ├── scenario_generation_for_two_stage_model.py      # The old scripts to generate scenarios for the two-stage model
│   ├── scenario_tree_builder_dfes.py                   # Helper functions for builder a scenario tree based on DFES pathways
│   ├── select_and_combine_ws_scenarios.py              #
│   └── set_dfes_data.py                                #
│
├── factories/                      # Factory pattern implementations (quick instance construction for core classes)
│   ├── __init__.py
│   ├── network_factory.py          # Network creation factory
│   ├── scenario_tree_factory.py    # Scenario Tree creation factory
│   └── windstorm_factory.py        # Windstorm creation factory
│
├── tests/                          # Test scripts (for debugging purposes; ignore this if you are a user)
│   ├── __init__.py
│   ├── temp.py
│   ├── test.py
│   ├── test_network_modelling.py
│   ├── test_scenario_tree.py
│   └── verify_OPF.py
│
├── utils.py                    # Utility functions (currently not in use)
├── visualization.py            # Visualization utilities
├── main.py                     # Main execution script
│
├── Images_and_Plots/           # Images and plots output by visualization functions
│
├── Input_Data/                             # Input data directory
│   ├── Demand_Profile/                     # All demand profiles
│   ├── DFES_Projections/                   # DFES projections including the modified workbook for the distribution network model
│   ├── GB_Network_29bus/                   # The 29-bus simplified GB network model
│   ├── GB_Network_full/                    # The 401-bus full GB network model (imported from pyWELLNESS)
│   └── Manchester_Distribution_Network/    # The manchester distribution network (currently only the 'Kearsley' GSP group)
│
├── Optimization_Results/         # Results output by the investment planning optimisation model
│
├── Output_Results/               # full .lp models and logs of the optimisation results (not useful less debugging)
│
├── Scenario_Database/            # Scenario results
│   ├── Scenarios_for_Scenario_Tree/        # Candidate scenarios for the scenario tree operational block
│   └── Scenarios_for_Two_Stage_Model       # (Old) Scenarios as the input to the old two-stage MILP model (works with 'investment_model_two_stage.py')
│
├── Scenario_Trees/               # Scenario trees stored in .json files
│
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
└── .gitignore                    # Git ignore file
```

## LastEditDate

24/July/2025