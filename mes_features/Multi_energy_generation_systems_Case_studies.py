#!/usr/bin/env python
# coding: utf-8

# # Multi-energy generation systems - Case studies

# ***&copy; 2023 Martínez Ceseña<sup>1</sup> and Mancarella<sup>2</sup> — <sup>1,2</sup>University of Manchester, UK, and <sup>2</sup>University of Melbourne, Australia***

# This is one of the documents in a series of jupyter notebooks which presents a general formulation to optimise the operation of centralised multi-energy systems (MES). The optimisation models provide time series (e.g., series of hourly periods) with the optimal set points of distributed energy resources (DER) and imports and exports from different networks required to meet energy demand (e.g., electricity, heat and gas). The specific models presented here are based on the following literature:
# 
# 1. E. A. Martínez Ceseña, T. Capuder and P. Mancarella, “[Flexible distributed multi-energy generation system expansion planning under uncertainty](https://ieeexplore.ieee.org/document/7064771),” IEEE Transactions on Smart Grid, Vol. 7, pp. 348 –357, 2016.
# 
# 1. T. Capuder, P. Mancarella, "[Techno-economic and environmental modelling and optimization of flexible distributed multi-generation options](https://www.sciencedirect.com/science/article/pii/S0360544214005283)," Energy, Vol. 71, pp. 516-533, 2014.
# 
# 1. E. A. Martínez Ceseña, E. Loukarakis, N. Good and P. Mancarella, "[Integrated Electricity-Heat-Gas Systems: Techno-Economic Modeling, Optimization, and Application to Multienergy Districts](https://ieeexplore.ieee.org/document/9108286)," in Proceedings of the IEEE, Vol. 108, pp. 1392 –1410, 2020.
# 
# 1. G. Chicco, S. Riaz, A. Mazza and P. Mancarella, "[Flexibility From Distributed Multienergy Systems](https://ieeexplore.ieee.org/document/9082595)," in Proceedings of the IEEE, Vol. 108, pp. 1496-1517, 2020.
# 
# 1. E. Corsetti, S. Riaz, M. Riello, P. Mancarella, “[Modelling and deploying multi-energy flexibility: The energy lattice framework](https://www.sciencedirect.com/science/article/pii/S2666792421000238)”, Advances in Applied Energy, Vol. 2, 2021.
# 
# This particular notebook provides several examples based on the MES formulations presented in previous notebooks, namely:
# - *Multi-energy generation systems - Example with gas boiler*
# - *Multi-energy generation systems - Example with cogeneration*

# ## List of contents

# - [MES with gas boiler](#MES-with-gas-boiler)
#   - [Baseline example](#Baseline-example)
# - [MES with cogeneration](#MES-with-cogeneration)
#   - [Replicating baseline example](#Replicating-baseline-example)
#   - [MES with gas boiler and EHP](#MES-with-gas-boiler-and-EHP)
#   - [MES with boiler, EHP and CHP](#MES-with-boiler,-EHP-and-CHP)
#   - [Flexible MES under variable conditions](#Flexible-MES-under-variable-conditions)

# [Back to top](#Multi-energy-generation-systems---Case-studies)

# ## Before we begin

# Before we begin, be aware that, to benefit the most from this notebook, you will need a basic understanding of: 
# - [Linear programming](https://realpython.com/linear-programming-python/) (LP) models, which are the types of models presented below.
# - [Python](https://www.python.org/), which is the language used in this notebook.
# - The [pyomo](https://pyomo.readthedocs.io/en/stable/index.html) library, which is the optimisation software used to solve the examples in this notebook.

# You will also need to check the previous jupyter notebooks titled *Multi-energy generation systems - Example with gas boiler* and *Multi-energy generation systems - Example with cogeneration* for detailed explanations about the models presented here.
# 
# The MES models required for this notebook are loaded using the `nbimporter` library:

# In[1]:


import nbimporter


# Now that we have installed the notebook importer, we can import commands from the first notebook:

# In[2]:


# Loading methods from Multi-energy generation systems - Example with gas boiler
mes_Boiler = __import__('Multi-energy generation systems - Example with gas boiler')
build_MES_Boiler_Model = mes_Boiler.build_MES_Boiler_Model
plot_MES_boiler = mes_Boiler.plot_MES_boiler
sankey_MES_Boiler = mes_Boiler.sankey_MES_Boiler


# We can also import commands from the second notebook.

# In[3]:


# Loading methods from Multi-energy generation systems - Example with cogeneration
mes_CoG = __import__('Multi-energy generation systems - Example with cogeneration')
build_MES_Cogeneration_Model = mes_CoG.build_MES_Cogeneration_Model
plot_MES = mes_CoG.plot_MES
sankey_MES_Cogeneration = mes_CoG.sankey_MES_Cogeneration
flexibility_MES_Cogeneration = mes_CoG.flexibility_MES_Cogeneration


# It is also needed to import `pyomo` which will be used in all the examples below.

# In[4]:


import pyomo.environ as pyo


# Now we have everything ready to do use the case studies below. **Note that the case studies will not work until you run all the commands above**.

# [Back to top](#Multi-energy-generation-systems---Case-studies)

# ## MES with gas boiler

# Let us begin with the MES model developed in the *Multi-energy generation systems - Example with gas boiler* notebook.

# ![Building_Boiler.png](Figures/Building_Boiler.png)
# <center><b>Figure 1. </b>Flow diagram of MES with gas boiler.</center>

# ### Baseline example

# The MES can be modelled using the set of python methods imported from the first notebook. This MES is inflexible, as the system does not have any alternative options to meet the demands.
# - **Change the energy prices. Does the operation of the MES change?**

# In[5]:


# Create pyomo object
model = pyo.ConcreteModel()
model.Set_Periods = range(3)

# Add parameters
model.dt = 1
model.Electricity_Demand = [1, 2, 1]  # [kW]
model.Heat_Demand = [2, 4, 1]  # [kW]

model.Electricity_Import_Price = [0.15, 0.20, 0.25]  # [£/kW]
model.Gas_Import_Price = [0.06, 0.06, 0.06]  # [£/kWh]

model.Boiler_Heat_Capacity = 5  # [kW]
model.Boiler_Heat_Efficiency = 0.85  # [pu]

# Solve optimisation problem
results = build_MES_Boiler_Model(model)

# Visualize results
plot_MES_boiler(model)
sankey_MES_Boiler(model)
print('Costs: %.2f'%model.Objective_Function.expr(), '[£]')


# [Back to top](#Multi-energy-generation-systems---Case-studies)

# ## MES with cogeneration

# This time we will model a more complex MES, which is shown in **Figure 2**:

# ![MES_cogeneration.png](Figures/MES_cogeneration.png)
# <center><b>Figure 2. </b>Flow diagram of a MES with cogeneration.</center>

# ### Replicating baseline example

# To remove some DER set their capacities to zero. If we set the capacities of the CHP and EHP units to zero, the results would be the same as those from the previous case where the MES only had a gas boiler.

# ![MES_examples_No_CHP_EHP.png](Figures/MES_examples_No_CHP_EHP.png)
# <center><b>Figure 3. </b>Flow diagram of a MES with cogeneration - Setting CHP and EHP capacity to zero.</center>

# As mentioned above, this system is not flexible to change its energy use. Thus, the energy flows will remain the same even if the energy prices change. 
# - **You can change the prices and update the results**.

# In[6]:


# Setting pyomo model and number of periods to optimise
model = pyo.ConcreteModel()
model.Set_Periods = range(3)
model.dt = 1

# Demands
model.Electricity_Demand = [1, 2, 1]  # [kW]
model.Heat_Demand = [2, 4, 1]  # [kW]

# DER
model.Boiler_Heat_Capacity = 5  # [kW]
model.Boiler_Heat_Efficiency = 0.85  # [pu]

model.EHP_Heat_Capacity = 0  # Setting capacity to zero [kW]
model.EHP_Heat_Efficiency = 3  # [pu]

model.CHP_Heat_Capacity = 0  # Setting capacity to zero [kW]
model.CHP_Electricity_Efficiency = 0.35  # [pu]
model.CHP_Heat_Efficiency = 0.45  # [pu]

# Prices and penalties
model.Electricity_Import_Price = [0.15, 0.20, 0.25]  # [£/kWh]
model.Electricity_Export_Price = [0.01, 0.01, 0.01]  # [£/kWh]
model.Gas_Import_Price = [0.06, 0.06, 0.06]  # [£/kWh]

model.Heat_Spill_Penalty = [0, 0, 0]  # [£/kWh]
model.Heat_Shedding_Penalty = [1000, 1000, 1000]  # [£/kWh]

# Building and solving model
results = build_MES_Cogeneration_Model(model)

# Visualizing model
Fig_Names = ['Electricity_Import', 'Boiler_Heat_Output', 'Boiler_Gas_Input']
plot_MES(model, Fig_Names, 'Electrical/Thermal power [kW]')

sankey_MES_Cogeneration(model)

flexibility_MES_Cogeneration(model)
print('Costs: %.2f'%model.Objective_Function.expr(), '[£]')


# ### MES with gas boiler and EHP

# Using the same model, no enable the EHP unit again by increasing its capacity.

# ![MES_examples_No_CHP.png](Figures/MES_examples_No_CHP.png)
# <center><b>Figure 4. </b>Flow diagram of a MES with cogeneration - Setting CHP capacity to zero.</center>

# The EHP provides a highly efficient option to meet heating demand, but it consumes electricity which can be expensive. As shown in the example below, The EHP is an attractive option to meet heat demand as long as the electricity import prices are not too high. 
# - **How would the results change with different electricity import prices?**

# In[7]:


# Setting pyomo model and number of periods to optimise
model = pyo.ConcreteModel()
model.Set_Periods = range(3)
model.dt = 1

# Demands
model.Electricity_Demand = [1, 2, 1]  # [kW]
model.Heat_Demand = [2, 4, 1]  # [kW]

# DER
model.Boiler_Heat_Capacity = 5  # [kW]
model.Boiler_Heat_Efficiency = 0.85  # [pu]

model.EHP_Heat_Capacity = 3  # Adding EHP capacity [kW]
model.EHP_Heat_Efficiency = 3  # [pu]

model.CHP_Heat_Capacity = 0  # Setting capacity to zero [kW]
model.CHP_Electricity_Efficiency = 0.35  # [pu]
model.CHP_Heat_Efficiency = 0.45  # [pu]

# Prices and penalties
model.Electricity_Import_Price = [0.15, 0.20, 0.25]  # [£/kWh]
model.Electricity_Export_Price = [0.01, 0.01, 0.01]  # [£/kWh]
model.Gas_Import_Price = [0.06, 0.06, 0.06]  # [£/kWh]

model.Heat_Spill_Penalty = [0, 0, 0]  # [£/kWh]
model.Heat_Shedding_Penalty = [1000, 1000, 1000]  # [£/kWh]

# Building and solving model
results = build_MES_Cogeneration_Model(model)

# Visualizing model
Fig_Names_E = ['Electricity_Import', 'EHP_Electricity_Input']
plot_MES(model, Fig_Names_E, 'Electrical power [kW]')

Fig_Names_H = ['Boiler_Heat_Output', 'EHP_Heat_Output']
plot_MES(model, Fig_Names_H, 'Thermal power [kW]')

sankey_MES_Cogeneration(model)

flexibility_MES_Cogeneration(model)
print('Costs: %.2f'%model.Objective_Function.expr(), '[£]')


# It is also possible to analyse the flexibility of individual devices such as the EHP

# In[8]:


Boiler=False
EHP=True
CHP=False
flexibility_MES_Cogeneration(model, Boiler, EHP, CHP)


# ### MES with boiler, EHP and CHP

# Let us now enable all available DER.

# ![MES_cogeneration.png](Figures/MES_cogeneration.png)
# <center><b>Figure 5. </b>Flow diagram of a MES with cogeneration - Enabling all DER.</center>

# Now that the system includes gas boilers, EHP and CHP we can replicate the example that is presented in the *Multi-energy generation systems - Example with cogeneration* notebook.
# 
# In this particular example, the CHP unit is used to keep the MES from importing costly electricity from the grid. If we check the results in the third period,  where electricity prices are high and heat demand is low (compared to electricity demand), it can be seen that it is more convenient to self-supply with the CHP and spill heat than to import electricity. 
# - **What would happen if the electricity import prices were lower?**
# - **How would the MES operate with higher electricity demand?**

# In[9]:


# Setting pyomo model and number of periods to optimise
model = pyo.ConcreteModel()
model.Set_Periods = range(3)
model.dt = 1

# Demands
model.Electricity_Demand = [1, 2, 1]  # [kW]
model.Heat_Demand = [2, 4, 1]  # [kW]

# DER
model.Boiler_Heat_Capacity = 5  # [kW]
model.Boiler_Heat_Efficiency = 0.85  # [pu]

model.EHP_Heat_Capacity = 5  # [kW]
model.EHP_Heat_Efficiency = 3  # [pu]

model.CHP_Heat_Capacity = 5  # [kW]
model.CHP_Electricity_Efficiency = 0.35  # [pu]
model.CHP_Heat_Efficiency = 0.45  # [pu]

# Prices and penalties
model.Electricity_Import_Price = [0.15, 0.20, 0.25]  # [£/kWh]
model.Electricity_Export_Price = [0.01, 0.01, 0.01]  # [£/kWh]
model.Gas_Import_Price = [0.06, 0.06, 0.06]  # [£/kWh]

model.Heat_Spill_Penalty = [0, 0, 0]  # [£/kWh]
model.Heat_Shedding_Penalty = [1000, 1000, 1000]  # [£/kWh]

# Building and solving model
results = build_MES_Cogeneration_Model(model)

# Visualizing model
Fig_Names_E = ['Electricity_Demand', 'EHP_Electricity_Input', 'Electricity_Export',
               'Electricity_Import', 'CHP_Electricity_Output']
plot_MES(model, Fig_Names_E, 'Electrical power [kW]')
Fig_Names_H = ['Heat_Demand', 'Heat_Spill', 'EHP_Heat_Output',
               'CCHP_Heat_Output', 'Boiler_Heat_Output', 'Heat_Shedding']
plot_MES(model, Fig_Names_H, 'Thermal power [kW]')

Fig_Names_G = ['CHP_Gas_Input', 'Boiler_Gas_Input', 'Gas_Import']
plot_MES(model, Fig_Names_G, 'Thermal (gas) power [kW]')

sankey_MES_Cogeneration(model)

flexibility_MES_Cogeneration(model)
print('Costs: %.2f'%model.Objective_Function.expr(), '[£]')


# As before, it is possible to analyse the flexibilty of selected DER

# In[10]:


Boiler=False
EHP=False
CHP=True
flexibility_MES_Cogeneration(model, Boiler, EHP, CHP)


# ### Flexible MES under variable conditions

# In practice, the conditions of the system are constantly changing due to natural demand variations, intermittent output of low marginal cost renewables and many other factors. The flexibility of MES allows them to use the most attractive combinations of energy vectors at any given period.
# 
# To analyse these different conditions, let us take the full model presented below, and expose it to different conditions.

# ![MES_cogeneration.png](Figures/MES_cogeneration.png)
# <center><b>Figure 6. </b>Flow diagram of a MES with cogeneration</center>

# 
# - **Analyse the system below; can you identify the conditions that motivate different MES operation?** 
# - **Think about the conditions that motivate the use of specific combinations of DER, spilling, shedding, etc.**

# In[11]:


# Setting pyomo model and number of periods to optimise
model = pyo.ConcreteModel()
model.Set_Periods = range(5)
model.dt = 1

# Demands
model.Electricity_Demand = [1, 1, 2, 2, 1]  # [kW]
model.Heat_Demand = [2, 4, 8, 10, 1]  # [kW]

# DER
model.Boiler_Heat_Capacity = 1  # [kW]
model.Boiler_Heat_Efficiency = 0.85  # [pu]

model.EHP_Heat_Capacity = 3  # [kW]
model.EHP_Heat_Efficiency = 3  # [pu]

model.CHP_Heat_Capacity = 4  # [kW]
model.CHP_Electricity_Efficiency = 0.35  # [pu]
model.CHP_Heat_Efficiency = 0.45  # [pu]

# Prices and penalties
model.Electricity_Import_Price = [0.10, 0.15, 0.20, 0.30, 0.20]  # [£/kWh]
model.Electricity_Export_Price = [0.01, 0.01, 0.01, 0.01, 0.01]  # [£/kWh]
model.Gas_Import_Price = [0.06, 0.06, 0.06, 0.06, 0.06]  # [£/kWh]

model.Heat_Spill_Penalty = [0, 0, 0, 0, 0]  # [£/kWh]
model.Heat_Shedding_Penalty = [1000, 1000, 1000, 1000, 1000]  # [£/kWh]

# Building and solving model
results = build_MES_Cogeneration_Model(model)

# Visualizing model
Fig_Names_E = ['Electricity_Demand', 'EHP_Electricity_Input', 'Electricity_Export',
               'Electricity_Import', 'CHP_Electricity_Output']
plot_MES(model, Fig_Names_E, 'Electrical power [kW]')
Fig_Names_H = ['Heat_Demand', 'Heat_Spill', 'EHP_Heat_Output',
               'CCHP_Heat_Output', 'Boiler_Heat_Output', 'Heat_Shedding']
plot_MES(model, Fig_Names_H, 'Thermal power [kW]')

Fig_Names_G = ['CHP_Gas_Input', 'Boiler_Gas_Input', 'Gas_Import']
plot_MES(model, Fig_Names_G, 'Thermal (gas) power [kW]')

sankey_MES_Cogeneration(model)

flexibility_MES_Cogeneration(model)
print('Costs: %.2f'%model.Objective_Function.expr(), '[£]')


# [Back to top](#Multi-energy-generation-systems---Case-studies)
