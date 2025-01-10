#!/usr/bin/env python
# coding: utf-8

# # Integrated networks modelling - Electricity examples

# ***&copy; 2023 Martínez Ceseña<sup>1</sup> and Mancarella<sup>2</sup> — <sup>1,2</sup>University of Manchester, UK, and <sup>2</sup>University of Melbourne, Australia***

# This is one of the documents in a series of jupyter notebooks which presents a Newton's based formulation to simulate integrated electricity, heating and gas networks under steady-state conditions. This particular notebook presents several case studies for the application of the power system model. More information about the models and some of their applications can also be found in the following literature:
# 
# 1. X. Liu and P. Mancarella, "[Modelling, assessment and Sankey diagrams of integrated electricity-heat-gas networks in multi-vector district energy systems](https://www.sciencedirect.com/science/article/pii/S0306261915010259)" in Applied Energy, Vol. 167, pp. 136 - 352, 2016.
# 
# 1. E. A. Martínez Ceseña, E. Loukarakis, N. Good and P. Mancarella, "[Integrated Electricity-Heat-Gas Systems: Techno-Economic Modeling, Optimization, and Application to Multienergy Districts](https://ieeexplore.ieee.org/document/9108286)," in Proceedings of the IEEE, Vol. 108, pp. 1392 –1410, 2020.
# 
# 1. E. A. Martínez Ceseña, T. Capuder and P. Mancarella, “[Flexible distributed multi-energy generation system expansion planning under uncertainty](https://ieeexplore.ieee.org/document/7064771),” IEEE Transactions on Smart Grid, Vol. 7, pp. 348 –357, 2016.
# 
# 1. G. Chicco, S. Riaz, A. Mazza and P. Mancarella, "[Flexibility From Distributed Multienergy Systems](https://ieeexplore.ieee.org/document/9082595)," in Proceedings of the IEEE, Vol. 108, pp. 1496-1517, 2020.
# 
# 1. E. Corsetti, S. Riaz, M. Riello, P. Mancarella, “[Modelling and deploying multi-energy flexibility: The energy lattice framework](https://www.sciencedirect.com/science/article/pii/S2666792421000238)”, Advances in Applied Energy, Vol. 2, 2021.
# 
# Other notebooks that cover the formulation of the integrated network model include:
# - *Integrated networks modelling - Heat examples*
# - *Integrated networks modelling - Gas examples*
# - *Integrated networks modelling - Examples*

# ## List of contents

# - [Same configuration but different values](#Same-configuration-but-different-values)
# - [Different system configuration](#Different-system-configuration)
# - [Interactive system](#Interactive-system)

# [Back to top](#Integrated-networks-modelling---Electricity-examples)

# ## Before we begin

# Before we begin, be aware that, to benefit the most from this notebook, you will need a basic understanding of: 
# - [Newton's method](https://www.sciencedirect.com/topics/mathematics/newtons-method), which is the method used in this notebook.
# - [Python](https://www.python.org/), which is the language used in this notebook.
# 
# It is also strongly suggested to review the following notebook:
# - Integrated networks modelling - Electricity

# The notebook requires some python functionalities, which should be imported as follows:

# In[1]:


import numpy as np
import ipywidgets as widgets
from ipywidgets import interact


# The notebook also requires a power network simulator developed in another notebook:

# In[2]:


import nbimporter
Power_Network = __import__('Integrated networks modelling - Electricity')
Elec_Model = Power_Network.Elec_Model


# [Back to top](#Integrated-networks-modelling---Electricity-examples)

# Now that we have loaded all the required tools, let us simulate a few power systems under different conditions.

# ## Same configuration but different values

# We can begin with applications of the model to the same system after changing some parameters. The default system settings lead to low voltage problems. 
# 
# ***How can you solve the issue?***

# ![Power_Network_3Bus.png](Figures/Power_Network_3Bus.png)

# In[3]:


Elec_Network = {}
Elec_Network['Connectivity'] = np.array([[1, 2], [1, 3], [2, 3]])
Elec_Network['R'] = [0.02, 0.05, 0.08]
Elec_Network['X'] = [0.2, 0.5, 0.8]

Elec_Network['Demand_Active'] = [20, 35, 30]
Elec_Network['Demand_Reactive'] = [0, 7, 15]
Elec_Network['Generation_Active'] = [0, 0, 0]
Elec_Network['Generation_Reactive'] = [0, 0, 0]

Elec_Network['Slack_Bus'] = 3 

# Simulate network using Newton's method
model = Elec_Model(Elec_Network)
model.run()
model.display()


# [Back to top](#Integrated-networks-modelling---Electricity-examples)

# ## Different system configuration

# It is possible to create other system configurations as shown below. There are once again voltage problems. 
# 
# ***Assuming you can constrol `Generator 1` and `Geberator 5`, how can you solve the problem?***

# ![Power_Network_5Bus.png](Figures/Power_Network_5Bus.png)

# In[4]:


Elec_Network = {}
Elec_Network['Connectivity'] = np.array([[1, 2], [2, 3], [3, 4], [3, 5]])
Elec_Network['R'] = [0.5, 0.5, 0.5, 0.2]
Elec_Network['X'] = [1.0, 1.0, 1.0, 0.4]

Elec_Network['Demand_Active'] = [0, 1, 0, 5, 0]
Elec_Network['Demand_Reactive'] = [0, 0.1, 0, 0.5, 0]
Elec_Network['Generation_Active'] = [0, 0, 0, 0, 0]
Elec_Network['Generation_Reactive'] = [0, 0, 0, 0, 0]

Elec_Network['Slack_Voltage'] = 1.0

# Simulate network using Newton's method
model = Elec_Model(Elec_Network)
model.run()
model.display()


# [Back to top](#Integrated-networks-modelling---Electricity-examples)

# ## Interactive system

# Lets use the same system, but this time you can only control the output of `Geberator 5`.
# 
# *** how can you solve the problem?***
# 
# > Use the sliders

# ![Power_Network_5Bus.png](Figures/Power_Network_5Bus.png)

# In[5]:


@interact
def Pow_Sys(P5 = widgets.FloatSlider(min=0,max=1,step=0.01,value=0.0), 
            Q5 = widgets.FloatSlider(min=0,max=1,step=0.01,value=0.0)):
    Elec_Network = {}
    Elec_Network['Connectivity'] = np.array([[1, 2], [2, 3], [3, 4], [3, 5]])
    Elec_Network['R'] = [0.5, 0.5, 0.5, 0.2]
    Elec_Network['X'] = [1.0, 1.0, 1.0, 0.4]

    Elec_Network['Demand_Active'] = [0, 1, 0, 5, 0]
    Elec_Network['Demand_Reactive'] = [0, 0.1, 0, 0.5, 0]
    Elec_Network['Generation_Active'] = [0, 0, 0, 0, P5]
    Elec_Network['Generation_Reactive'] = [0, 0, 0, 0, Q5]

    Elec_Network['Slack_Voltage'] = 1.05

    # Simulate network using Newton's method
    model = Elec_Model(Elec_Network)
    model.run()
    model.display()


# [Back to top](#Integrated-networks-modelling---Electricity-examples)
