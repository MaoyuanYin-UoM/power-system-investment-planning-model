#!/usr/bin/env python
# coding: utf-8

# # Multi-energy generation systems - General formulation examples

# ***&copy; 2023 Martínez Ceseña<sup>1</sup> and Mancarella<sup>2</sup> — <sup>1,2</sup>University of Manchester, UK, and <sup>2</sup>University of Melbourne, Australia***

# This notebook provides examples of different MES configurations using the tools developed in the previous notebook: **Multi-energy generation systems - General formulation**.
# 
# For more information about general MES modelling frameworks and other examples, please see:
# 
# - E. A. Martínez Ceseña, T. Capuder and P. Mancarella, “[Flexible distributed multi-energy generation system expansion planning under uncertainty](https://ieeexplore.ieee.org/document/7064771),” IEEE Transactions on Smart Grid, vol. 7, no. 1, pp. 348 –357, 2016.
# 
# 
# - T. Capuder, P. Mancarella, "[Techno-economic and environmental modelling and optimization of flexible distributed multi-generation options](https://www.sciencedirect.com/science/article/pii/S0360544214005283)," Energy, vol. 71, pp. 516-533, 2014.

# In order to import models from another jupyter notebook, you will need the `nbimporter` library. The library can be installed with the following command.

# No we can import `nbimporter` and `pyomo`, as well as models from previous jupyter notebooks. However, unlike typical python packages, the title of some jupyter notebooks have spaces which prevents the traditional use of the `import` command. Accordingly, we will use an alternative form of the `import` command to import the model from the notebook and then export the relevant models to build MES models and visualize them (i.e., plot time series, and sankey diagrams).

# In[1]:


import nbimporter
import pyomo.environ as pyo

mes = __import__('Multi-energy generation systems - General formulation')
build_MES=mes.build_MES
plot_MES=mes.plot_MES
sankey_MES=mes.sankey_MES


# ## List of contents

# - [MES with gas boiler](#MES-with-gas-boiler)
# - [Flexible MES with boler and EHP](#Flexible-MES-with-boler-and-EHP)
# - [MES with renewables and storage](#MES-with-renewables-and-storage)
# - [Electrified MES with thermal storage](#Electrified-MES-with-thermal-storage)
# - [MES with heating and cooling](#MES-with-heating-and-cooling)
# - [MES with heating, cooling and shedding](#MES-with-heating,-cooling-and-shedding)
# - [MES with heating, cooling and spilling](#MES-with-heating,-cooling-and-spilling)
# - [MES fully electrified](#MES-fully-electrified)
# - [Create your own example](#Create-your-own-example)

# ## MES with gas boiler

# ![Building_Boiler.png](Figures/Building_Boiler.png)

# In[2]:


model = pyo.ConcreteModel()
model.Set_Vectors = ['Electricity', 'Heat', 'Gas']

model.Name = 'MES1'
model.MES1_Set_DER = ['Boiler']
model.MES1_Boiler = {
    'Capacity': 5,
    'Vector': 'Heat',
    'Efficiency': [0.85],
    'Input': ['Gas'],
    'Output': ['Heat']
}

model.MES1_Electricity_Demand = [1, 2, 1]  # (kW)
model.MES1_Heat_Demand = [2, 4, 1]  # (kW)
model.MES1_Gas_Demand = [0, 0, 0]  # (kW)

model.MES1_Electricity_Import_Price = [0.15, 0.20, 0.25]  # (£/kWh)
model.MES1_Gas_Import_Price = [0.05, 0.05, 0.05]  # (£/kWh)
model.dt = 1

# Build and solve model
build_MES(model)

# Visualize
Fig_Names = ['MES1_Electricity_Import', 'MES1_Boiler_Heat_Output',
             'MES1_Gas_Import']
plot_MES(model, Fig_Names)
sankey_MES(model, 'MES with gas boiler')
print('Costs: %.2f'%model.Objective_Function.expr(), '(£)')


# [Back to top](#List-of-contents)

# ## Flexible MES with boler and EHP

# ![Building_Boiler_EHP.png](Figures/Building_Boiler_EHP.png)

# In[3]:


model = pyo.ConcreteModel()
model.Set_Vectors = ['Electricity', 'Heat', 'Gas']
model.dt = 1

model.Name = 'MES2'
model.MES2_Set_DER = ['Boiler', 'EHP']
model.MES2_Boiler = {
    'Capacity': 5,
    'Vector': 'Heat',
    'Efficiency': [0.85],
    'Input': ['Gas'],
    'Output': ['Heat']
}
model.MES2_EHP = {
    'Capacity': 5,
    'Vector': 'Heat',
    'Efficiency': [3],
    'Input': ['Electricity'],
    'Output': ['Heat']
}

model.MES2_Electricity_Demand = [1, 2, 1]  # (kW)
model.MES2_Heat_Demand = [2, 4, 1]  # (kW)

model.MES2_Electricity_Import_Price = [0.15, 0.20, 0.25]  # (£/kWh)
model.MES2_Gas_Import_Price = [0.05, 0.05, 0.05]  # (£/kWh)

# Build and solve model
build_MES(model)

# Visualize
Fig_Names = ['MES2_Electricity_Import', 'MES2_Boiler_Heat_Output',
             'MES2_EHP_Heat_Output', 'MES2_Gas_Import']
plot_MES(model, Fig_Names)
sankey_MES(model, 'MES with gas boiler and EHP')
print('Costs: %.2f'%model.Objective_Function.expr(), '(£)')


# [Back to top](#List-of-contents)

# ## MES with renewables and storage

# ![Building_Boiler_PV_BES.png](Figures/Building_Boiler_PV_BES.png)

# In[4]:


model = pyo.ConcreteModel()
model.Set_Vectors = ['Electricity', 'Heat', 'Gas', 'Insolation']
model.dt = 1

model.Name = 'MES3'
model.MES3_Set_DER = ['Boiler', 'PV', 'BES']
model.MES3_Boiler = {
    'Capacity': 5,
    'Vector': 'Heat',
    'Efficiency': [0.85],
    'Input': ['Gas'],
    'Output': ['Heat']
}
model.MES3_PV = {
    'Capacity': 4,
    'Vector': 'Electricity',
    'Efficiency': [1],
    'Input': ['Insolation'],
    'Output': ['Electricity']
}
model.MES3_BES = {
    'Capacity': 2,
    'Vector': 'Electricity'
}

model.MES3_Electricity_Demand = [1, 2, 1]  # (kW)
model.MES3_Heat_Demand = [2, 4, 1]  # (kW)

model.MES3_Electricity_Import_Price = [0.15, 0.20, 0.25]  # (£/kWh)
model.MES3_Electricity_Export_Price = [0.01, 0.01, 0.01]  # (£/kWh)
model.MES3_Insolation_Import_Price = [0, 0, 0]  # (£/kWh)
model.MES3_Gas_Import_Price = [0.05, 0.05, 0.05]  # (£/kWh)

model.MES3_Insolation_Import_Limit = [0, 2, 0]  # (kW)

# Build and solve model
build_MES(model)

# Visualize
#Fig_Names = ['MES3_Electricity_Import', 'MES3_Electricity_Export',
#             'MES3_PV_Electricity_Output', 'MES3_BES_Electricity_Output']
#plot_MES(model, Fig_Names)
#Fig_Names = ['MES3_Gas_Import', 'MES3_Boiler_Heat_Output']
#plot_MES(model, Fig_Names)
sankey_MES(model, 'MES with PV, BES and gas boiler')
#print('Costs: %.2f'%model.Objective_Function.expr(), '(£)')


# [Back to top](#List-of-contents)

# ## Electrified MES with thermal storage

# ![Building_EHP_TES.png](Figures/Building_EHP_TES.png)

# In[5]:


model = pyo.ConcreteModel()
model.Set_Vectors = ['Electricity', 'Heat']
model.dt = 1

model.Name = 'MES4'
model.MES4_Set_DER = ['EHP', 'TES']
model.MES4_EHP = {
    'Capacity': 5,
    'Vector': 'Heat',
    'Efficiency': [3],
    'Input': ['Electricity'],
    'Output': ['Heat']
}
model.MES4_TES = {
    'Capacity': 2,
    'Vector': 'Heat'
}

model.MES4_Electricity_Demand = [1, 2, 1]  # (kW)
model.MES4_Heat_Demand = [2, 4, 1]  # (kW)

model.MES4_Electricity_Import_Price = [0.15, 0.20, 0.25]  # (£/kWh)

# Build and solve model
build_MES(model)

# Visualize
Fig_Names = ['MES4_Electricity_Import', 'MES4_EHP_Electricity_Input']
plot_MES(model, Fig_Names)
Fig_Names = ['MES4_EHP_Heat_Output', 'MES4_TES_Heat_Output']
plot_MES(model, Fig_Names)
sankey_MES(model, 'MES with EHP and TES')
print('Costs: %.2f'%model.Objective_Function.expr(), '(£)')


# [Back to top](#List-of-contents)

# ## MES with heating and cooling

# In[6]:


model = pyo.ConcreteModel()
model.Set_Vectors = ['Electricity', 'Heat', 'Cooling', 'Gas']
model.dt = 1

model.Name = 'MES5'
model.MES5_Set_DER = ['CHP', 'Chiller']
model.MES5_CHP = {
    'Capacity': 5,
    'Vector': 'Heat',
    'Efficiency': [0.35, 0.45],
    'Input': ['Gas', 'Gas'],
    'Output': ['Electricity', 'Heat']
}
model.MES5_Chiller = {
    'Capacity': 5,
    'Vector': 'Cooling',
    'Efficiency': [1],
    'Input': ['Heat'],
    'Output': ['Cooling']
}

model.MES5_Electricity_Demand = [1, 2, 1]  # (kW)
model.MES5_Heat_Demand = [2, 2, 1]  # (kW)
model.MES5_Cooling_Demand = [1, 3, 0]  # (kW)

model.MES5_Electricity_Import_Price = [0.15, 0.20, 0.25]  # (£/kWh)
model.MES5_Electricity_Export_Price = [0.01, 0.01, 0.01]  # (£/kWh)
model.MES5_Gas_Import_Price = [0.05, 0.05, 0.05]  # (£/kWh)

# Build and solve model
result=build_MES(model)

# Visualize
Fig_Names = ['MES5_Electricity_Import', 'MES5_Electricity_Export', 'MES5_CHP_Electricity_Output']
plot_MES(model, Fig_Names)
Fig_Names = ['MES5_CHP_Heat_Output', 'MES5_Heat_Demand', 'MES5_Chiller_Cooling_Output']
plot_MES(model, Fig_Names)
Fig_Names = ['MES5_CHP_Gas_Input']
plot_MES(model, Fig_Names)
sankey_MES(model, 'MES with cogeneration')
print('Costs: %.2f'%model.Objective_Function.expr(), '(£)')


# [Back to top](#List-of-contents)

# ## MES with heating, cooling and shedding

# In[7]:


model = pyo.ConcreteModel()
model.Set_Vectors = ['Electricity', 'Heat', 'Cooling', 'Gas', 'Shedding']
model.dt = 1

model.Name = 'MES5'
model.MES5_Set_DER = ['CHP', 'Chiller', 'Shedding_Electricity', 'Shedding_Heat', 'Shedding_Cooling']
model.MES5_CHP = {
    'Capacity': 5,
    'Vector': 'Heat',
    'Efficiency': [0.35, 0.45],
    'Input': ['Gas', 'Gas'],
    'Output': ['Electricity', 'Heat']
}
model.MES5_Chiller = {
    'Capacity': 5,
    'Vector': 'Cooling',
    'Efficiency': [1],
    'Input': ['Heat'],
    'Output': ['Cooling']
}
model.MES5_Shedding_Electricity = {
    'Capacity': 100,
    'Vector': 'Shedding',
    'Efficiency': [1],
    'Input': ['Shedding'],
    'Output': ['Electricity']
}
model.MES5_Shedding_Heat = {
    'Capacity': 100,
    'Vector': 'Shedding',
    'Efficiency': [1],
    'Input': ['Shedding'],
    'Output': ['Heat']
}
model.MES5_Shedding_Cooling = {
    'Capacity': 100,
    'Vector': 'Shedding',
    'Efficiency': [1],
    'Input': ['Shedding'],
    'Output': ['Cooling']
}

model.MES5_Electricity_Demand = [1, 2, 1]  # (kW)
model.MES5_Heat_Demand = [2, 2, 1]  # (kW)
model.MES5_Cooling_Demand = [1, 3, 0]  # (kW)

#model.MES5_Electricity_Import_Price = [0.15, 0.20, 0.25]  # (£/kWh)
#model.MES5_Electricity_Export_Price = [0.01, 0.01, 0.01]  # (£/kWh)
model.MES5_Gas_Import_Price = [0.05, 0.05, 0.05]  # (£/kWh)
model.MES5_Shedding_Import_Price = [1000, 1000, 1000]  # (£/kWh)

# Build and solve model
result=build_MES(model)

# Visualize
sankey_MES(model, 'Sankey diagram')
print('Costs: %.2f'%model.Objective_Function.expr(), '(£)')


# [Back to top](#List-of-contents)

# ## MES with heating, cooling and spilling

# In[8]:


model = pyo.ConcreteModel()
model.Set_Vectors = ['Electricity', 'Heat', 'Cooling', 'Gas', 'Shedding', 'Spilling']
model.dt = 1

model.Name = 'MES5'
model.MES5_Set_DER = ['CHP', 'Chiller', 'Shedding_Electricity', 'Shedding_Heat', 'Shedding_Cooling', 'Spill_CHP']
model.MES5_CHP = {
    'Capacity': 5,
    'Vector': 'Heat',
    'Efficiency': [0.35, 0.45],
    'Input': ['Gas', 'Gas'],
    'Output': ['Electricity', 'Heat']
}
model.MES5_Chiller = {
    'Capacity': 5,
    'Vector': 'Cooling',
    'Efficiency': [1],
    'Input': ['Heat'],
    'Output': ['Cooling']
}
model.MES5_Shedding_Electricity = {
    'Capacity': 100,
    'Vector': 'Shedding',
    'Efficiency': [1],
    'Input': ['Shedding'],
    'Output': ['Electricity']
}
model.MES5_Shedding_Heat = {
    'Capacity': 100,
    'Vector': 'Shedding',
    'Efficiency': [1],
    'Input': ['Shedding'],
    'Output': ['Heat']
}
model.MES5_Shedding_Cooling = {
    'Capacity': 100,
    'Vector': 'Shedding',
    'Efficiency': [1],
    'Input': ['Shedding'],
    'Output': ['Cooling']
}
model.MES5_Spill_CHP = {
    'Capacity': 100,
    'Vector': 'Electricity',
    'Efficiency': [1],
    'Input': ['Electricity'],
    'Output': ['Spilling']
}

model.MES5_Electricity_Demand = [1, 2, 1]  # (kW)
model.MES5_Heat_Demand = [2, 2, 1]  # (kW)
model.MES5_Cooling_Demand = [1, 3, 0]  # (kW)

#model.MES5_Electricity_Import_Price = [0.15, 0.20, 0.25]  # (£/kWh)
#model.MES5_Electricity_Export_Price = [0.01, 0.01, 0.01]  # (£/kWh)
model.MES5_Gas_Import_Price = [0.05, 0.05, 0.05]  # (£/kWh)
model.MES5_Shedding_Import_Price = [1000, 1000, 1000]  # (£/kWh)
model.MES5_Spilling_Export_Price = [-1000, -1000, -1000]  # (£/kWh)

# Build and solve model
result=build_MES(model)

# Visualize
sankey_MES(model, 'Sankey diagram')
print('Costs: %.2f'%model.Objective_Function.expr(), '(£)')


# [Back to top](#List-of-contents)

# ## MES fully electrified

# In[9]:


model = pyo.ConcreteModel()
model.Set_Vectors = ['Electricity', 'Insolation']
model.dt = 1

model.Name = 'MES3'
model.MES3_Set_DER = ['PV', 'BES']
model.MES3_PV = {
    'Capacity': 4,
    'Vector': 'Electricity',
    'Efficiency': [1],
    'Input': ['Insolation'],
    'Output': ['Electricity']
}
model.MES3_BES = {
    'Capacity': 2,
    'Vector': 'Electricity'
}

model.MES3_Electricity_Demand = [1, 2, 1]  # (kW)
model.MES3_Electricity_Import_Price = [0.15, 0.20, 0.25]  # (£/kWh)
model.MES3_Electricity_Export_Price = [0.01, 0.01, 0.01]  # (£/kWh)
model.MES3_Insolation_Import_Price = [0, 0, 0]  # (£/kWh)

model.MES3_Insolation_Import_Limit = [0, 2, 0]  # (kW)

# Build and solve model
build_MES(model)

# Visualize
#Fig_Names = ['MES3_Electricity_Import', 'MES3_Electricity_Export',
#             'MES3_PV_Electricity_Output', 'MES3_BES_Electricity_Output']
#plot_MES(model, Fig_Names)
#Fig_Names = ['MES3_Gas_Import', 'MES3_Boiler_Heat_Output']
#plot_MES(model, Fig_Names)
sankey_MES(model, 'MES with PV and BES')
#print('Costs: %.2f'%model.Objective_Function.expr(), '(£)')


# [Back to top](#List-of-contents)

# ## Create your own example

# In[10]:


model = pyo.ConcreteModel()
model.Set_Vectors = ['Electricity', 'Insolation']
model.dt = 1

model.Name = 'MES3'
model.MES3_Set_DER = ['PV', 'BES']
model.MES3_PV = {
    'Capacity': 4,
    'Vector': 'Electricity',
    'Efficiency': [1],
    'Input': ['Insolation'],
    'Output': ['Electricity']
}
model.MES3_BES = {
    'Capacity': 2,
    'Vector': 'Electricity'
}

model.MES3_Electricity_Demand = [1, 2, 1]  # (kW)
model.MES3_Electricity_Import_Price = [0.15, 0.20, 0.25]  # (£/kWh)
model.MES3_Electricity_Export_Price = [0.01, 0.01, 0.01]  # (£/kWh)
model.MES3_Insolation_Import_Price = [0, 0, 0]  # (£/kWh)

model.MES3_Insolation_Import_Limit = [0, 2, 0]  # (kW)

# Build and solve model
build_MES(model)

# Visualize
#Fig_Names = ['MES3_Electricity_Import', 'MES3_Electricity_Export',
#             'MES3_PV_Electricity_Output', 'MES3_BES_Electricity_Output']
#plot_MES(model, Fig_Names)
#Fig_Names = ['MES3_Gas_Import', 'MES3_Boiler_Heat_Output']
#plot_MES(model, Fig_Names)
sankey_MES(model, 'MES with PV and BES')
#print('Costs: %.2f'%model.Objective_Function.expr(), '(£)')


# [Back to top](#List-of-contents)
