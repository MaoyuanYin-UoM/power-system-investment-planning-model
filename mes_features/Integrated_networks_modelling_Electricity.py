#!/usr/bin/env python
# coding: utf-8

# # Integrated networks modelling - Electricity

# ***&copy; 2023 Martínez Ceseña<sup>1</sup> and Mancarella<sup>2</sup> — <sup>1,2</sup>University of Manchester, UK, and <sup>2</sup>University of Melbourne, Australia***

# This is one of the documents in a series of jupyter notebooks which presents a Newton's based formulation to simulate integrated electricity, heating and gas networks under steady-state conditions. This particular notebook presents the formulation for the power system model. More information about the models and some of their applications can also be found in the following literature:
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

# ## List of contents

# - [Power network](#Power-network)
# - [Newton's method applied to power systems](#Newton's-method-applied-to-power-systems)
# - [Coding Newton's method](#Coding-Newton's-method)
#   - [Jacobian matrix](#Jacobian-matrix)
#   - [Vector of differences](#Vector-of-differences)
#   - [Correction factors](#Correction-factors)
#   - [Solving the model](#Solving-the-model)
# - [Development as python tool](#Development-as-python-tool)

# [Back to top](#Integrated-networks-modelling---Electricity)

# ## Before we begin

# Before we begin, be aware that, to benefit the most from this notebook, you will need a basic understanding of: 
# - [Newton's method](https://www.sciencedirect.com/topics/mathematics/newtons-method), which is the method used in this notebook.
# - [Python](https://www.python.org/), which is the language used in this notebook.

# The notebook also requires some python functionalities, which should be imported as follows:

# In[1]:


import numpy as np
import scipy.sparse as sp
import cmath, math


# [Back to top](#Integrated-networks-modelling---Electricity)

# ## Power network

# Let us begin by defining the information that is needed to represent a power system. For that purpose, let us begin with an illustrative 3-bus system presented in Figure 1.

# ![Power_Network_3Bus.png](Figures/Power_Network_3Bus.png)
# <center><b>Figure 1. </b>Example 3-bus power network.</center>

# As shown, the network includes power lines, generation and demand. This information will be modelled in per unit value, so we will need to define an MVA base.

# In[2]:


Elec_Network = {}  # Creating python object
Elec_Network['Base'] = 100  # Adding base


# - To represent the network, we can describe the connectivity of each line and the impedances ($Z = R + jX$) across every line:

# In[3]:


Elec_Network['Connectivity'] = np.array([[1, 2], [1, 3], [2, 3]])
Elec_Network['R'] = [0.05, 0.05, 0.05]  # [pu]
Elec_Network['X'] = [0.5, 0.5, 0.5]  # [pu]


# - The generators are presented in terms of their active and reactive power injections

# In[4]:


Elec_Network['Generation_Active'] = [50.8622, 0, 10.173]
Elec_Network['Generation_Reactive'] = [27.363, 0, 10.2181]


# - The demands are also expressed in terms of their active and reactive components:

# In[5]:


Elec_Network['Demand_Active'] = [0.1451, 30, 30.136]
Elec_Network['Demand_Reactive'] = [0, 15, 15]


# The aim of the power flow calculation is to determine the outputs of all generators with variable outputs (i.e., slack generators and generators connected to PV buses) which can sustain the power injections introduced by the demands and fixed generators declared above. For the sake of simplicity, let us just declare a slack bus in a selected location:

# In[6]:


Elec_Network['Slack_Bus'] = 1
Elec_Network['Slack_Voltage'] = 1.0


# Now we have all the information required to populate the steady-state equations of the power system, which are presented below:

# $$
# P_k = \sum {V_k V_i (G_{ki} cos \theta_{ki} + B_{ki} sin \theta_{ki})}
# $$
# 
# $$
# Q_k = \sum {V_k V_i (G_{ki} sin \theta_{ki} + B_{ki} cos \theta_{ki})}
# $$

# where:
# - $P_k$ is the net active power injection at a node, e.g., for node 1, it would be the active generation minus the active demand.
# - $Q_k$ is the net reactive power injection at a node, e.g., for node 1, it would be the reactive generation minus the reactive demand.
# - V and $\theta$ are the polar components of complex voltages at each node.
# - G and B are respectively the admittance and susceptance components of the $Y_{bus}$ matrix.

# The $Y_{bus}$ matrix will have to be calculated to populate the power flow equations. The matrix can be generated with basic circuit analysis, or by following the shortcut method below:
# - Convert all line impedances to admittances

# In[7]:


def get_Admittance(Elec_Network):
    Elec_Network['G'] = []
    Elec_Network['B'] = []
    for i in range(len(Elec_Network['R'])):
        val = (Elec_Network['R'][i]**2+Elec_Network['X'][i]**2)**0.5
        ang = Elec_Network['R'][i]/val
        Elec_Network['G'].append(Elec_Network['R'][i]/val/val)
        Elec_Network['B'].append(-Elec_Network['X'][i]/val/val)
get_Admittance(Elec_Network)


# - The diagonal elements of the $Y_{bus}$ can be calculated as the summation of admittances connected to the node
# - The off-diagonal elements (representing connections between nodes) can be taken as the negative of the admittance connecting the relevant nodes.

# In[8]:


def get_Ybus(Elec_Network):
    '''Method to calculate Ybus'''
    Elec_Network['Buses'] = len(Elec_Network['Demand_Active'])
    Elec_Network['Lines'] = len(Elec_Network['G'])
    
    Ybus = np.zeros((Elec_Network['Buses'], Elec_Network['Buses']), dtype=complex)
    for b in np.arange(Elec_Network['Lines']):
        x = Elec_Network['Connectivity'][b][0]-1
        y = Elec_Network['Connectivity'][b][1]-1

        # Off-diagonal elements
        Ybus[x, y] -= (Elec_Network['G'][b] + Elec_Network['B'][b] * 1j)
        Ybus[y, x] -= (Elec_Network['G'][b] + Elec_Network['B'][b] * 1j)

        # Diagonal elements
        Ybus[x, x] += (Elec_Network['G'][b] + Elec_Network['B'][b] * 1j)
        Ybus[y, y] += (Elec_Network['G'][b] + Elec_Network['B'][b] * 1j)
    
    Elec_Network['Ybus'] = Ybus
get_Ybus(Elec_Network)
print(Elec_Network['Ybus'])


# Now that we have all the data, to simulate the power system, we need to identify the values of the unknown variables (most V and $\theta$ values). There are a wide range of models available to solve these equations; we will use one of the most established models, namely Newton's method.

# [Back to top](#Integrated-networks-modelling---Electricity)

# ## Newton's method applied to power systems

# At a high level, Newton's method is an approach that guesses an initial value iteratively approximates unknown variables in a function so that the function becomes close to zero.
# 
# For that purpose, an initial value is assigned to the unknown variable ($x_k$), and evaluated by the equation ($f('X_k)$) and its derivative ($f(X_k)$). This information and the equation below are used to estimate an improved value of the unknown variable ($x_{k+1}$) which should take the equation closer to zero.

# $$
# x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}
# $$

# This process is repeated until the function is closed to zero within an acceptable error margin.

# Based on the above, to apply Newton's method, need to:
# - Identify the unknown variables in the system. These would be most V and $\theta$ values excluding the slack bus (and voltage magnitudes for PV buses). In our example, the unknown variables would be $V_2$, $V_3$, $\theta_2$ and $\theta_3$

# - Now that we identified the unknown variables, initial guesses for their values can be assigned. Considering the characteristics of the power system, an acceptable guess is to set all voltage magnitues to one and all angles to zero. These are known as flat-start assumptions.

# - Next, we need to set the power flow equations to zero, which can be achieved by substracting the power injections as shown below. This will produce $\Delta P_k$ and $\Delta Q_k$ values which show how far the functions are from zero based on our guesses for the unknown variables.

# $$
# \Delta P_k = \sum {V_k V_i (G_{ki} cos \theta_{ki} + B_{ki} sin \theta_{ki})} - P_k
# $$
# 
# $$
# \Delta Q_k = \sum {V_k V_i (G_{ki} sin \theta_{ki} + B_{ki} cos \theta_{ki})} - Q_k
# $$

# - Newtons correction factor has to be developed in matrix form, as we have several unknown variables:

# 
# $$
# x_{k+1} = x_k - J^{-1}(x_k)f(x_k)
# $$
# 
# $$
# x_{k} = \begin{bmatrix} \theta_k\\V_k\end{bmatrix}
# $$
# 
# $$
# J = \begin{bmatrix} \frac{\partial \Delta P_k}{\partial\Delta \theta_k} & \frac{\partial \Delta P_k}{\partial\Delta V_k} \\ \frac{\partial \Delta Q_k}{\partial\Delta \theta_k} & \frac{\partial \Delta Q_k}{\partial\Delta V_k}\end{bmatrix}
# $$
# 
# $$
# f(x_k) = \begin{bmatrix} \Delta P_k\\ \Delta V_k\end{bmatrix}
# $$

# [Back to top](#Integrated-networks-modelling---Electricity)

# ## Coding Newton's method

# This section will briefly provide the code required to develop the Jacobian matrix ($J$), vector of differences ($f(x_k)$) and iterative approach required to apply Newton's method to solve the power system problem.
# 
# For convenience, several parameters are assigned shorter names (e.g., mV instead of Elec_Network['Voltage_Magnitude']) and are stored in polar form.

# $$
# x_{k} = \begin{bmatrix} \theta_k\\V_k\end{bmatrix}
# $$

# In[9]:


def get_Polar(Elec_Network):
    ''' Get data in polar form '''
    
    # Number of buses
    N = Elec_Network['Buses']
    
    # Slack
    slack = Elec_Network['Slack_Bus']-1
    
    # Power injections
    P = np.ones(Elec_Network['Buses'])
    Q = np.ones(Elec_Network['Buses'])
    for x in range(Elec_Network['Buses']):
        P[x] = (Elec_Network['Generation_Active'][x] -
                Elec_Network['Demand_Active'][x])/Elec_Network['Base']
        Q[x] = (Elec_Network['Generation_Reactive'][x] -
                Elec_Network['Demand_Reactive'][x])/Elec_Network['Base']

    # Y bus
    aY = np.ones((Elec_Network['Buses'], Elec_Network['Buses']))
    mY = np.ones((Elec_Network['Buses'], Elec_Network['Buses']))
    for x in range(Elec_Network['Buses']):
        for y in range(Elec_Network['Buses']):
            aY[x][y] = cmath.phase(Elec_Network['Ybus'][x][y])
            mY[x][y] = abs(Elec_Network['Ybus'][x][y])

    # Voltages - Assuming flat start
    Elec_Network['Voltage_Magnitude'] = np.ones((Elec_Network['Buses']))
    Elec_Network['Voltage_Angle'] = np.zeros((Elec_Network['Buses']))

    mV = Elec_Network['Voltage_Magnitude']
    aV = Elec_Network['Voltage_Angle']
    mV[slack] = Elec_Network['Slack_Voltage']
    
    return (P, Q, aY, mY, mV, aV, N, slack)


# ### Jacobian matrix

# The jacobian matrix is developped by differentiating the $\Delta P_k$ and $\Delta Q_k$ equations with respect to the unknown variables ($V$ and $\thetas$). 
# 
# We will not cover the maths required for this, and instead provide the code directly. That said, there is abundant literature about the topic, such as [this book](https://www.readinglists.manchester.ac.uk/leganto/readinglist/citation/327930681560001631?institute=44MAN_INST&auth=CAS).

# $$
# J = \begin{bmatrix} \frac{\partial \Delta P_k}{\partial\Delta \theta_k} & \frac{\partial \Delta P_k}{\partial\Delta V_k} \\ \frac{\partial \Delta Q_k}{\partial\Delta \theta_k} & \frac{\partial \Delta Q_k}{\partial\Delta V_k}\end{bmatrix}
# $$

# In[10]:


def get_Jacobian(Elec_Network, aY, mY, mV, aV, N, slack):
    ''' Build Jacobian matrix'''
    J = np.zeros((2*(N-1), 2*(N-1)))
    
    if slack == 0:
        x = 1
    else:
        x = 0
    xj = 0
    while x < N:
        if slack == 0:
            y = 1
        else:
            y = 0
        yj = 0
        while y < N:
            if x==y:
                J[xj][xj+N-1] = 2*mV[x]*mY[x][x]*np.cos(aY[x][x])
                J[xj+N-1][xj+N-1]=-2*mV[x]*mY[x][x]*np.sin(aY[x][x])
                for z in range(N):
                   if z != x:
                       J[xj][xj] += mV[x]*mV[z]*mY[x][z]*np.sin(aY[x][z]-aV[x]+aV[z])
                       J[xj][xj+N-1] += mV[z]*mY[x][z]*np.cos(aY[x][z]-aV[x]+aV[z])
                       J[xj+N-1][xj] += mV[x]*mV[z]*mY[x][z]*np.cos(aY[x][z]-aV[x]+aV[z])
                       J[xj+N-1][xj+N-1] += -mV[z]*mY[x][z]*np.sin(aY[x][z]-aV[x]+aV[z])
            else:
                if aY[x][y] != 0:
                   J[xj][yj] = -1*mV[x]*mV[y]*mY[x][y]*np.sin(aY[x][y]-aV[x]+aV[y])
                   J[xj][yj+N-1] = mV[x]*mY[x][y]*np.cos(aY[x][y]-aV[x]+aV[y])
                   J[xj+N-1][yj] = -1*mV[y]*J[x-1][y+N-2]
                   J[xj+N-1][yj+N-1] = -1*mV[x]*mY[x][y]*np.sin(aY[x][y]-aV[x]+aV[y])
            y += 1
            if y == slack:
                y += 1
            yj += 1
        x += 1
        if x == slack:
            x += 1
        xj += 1

    return J


# [Back to top](#Integrated-networks-modelling---Electricity)

# ### Vector of differences

# The power differences at each node ($f(x_k)$) can be calculated by sunvtracting the power injections to the relevant power flow equations as discussed in the previous section.

# $$
# f(x_k) = \begin{bmatrix} \Delta P_k\\ \Delta Q_k\end{bmatrix}
# $$

# In[11]:


def get_Power_Differences(P, Q, aY, mY, mV, aV, N, slack):
    ''' # Get vector of differences '''
    df = np.zeros(2*(N-1))
    if slack == 0:
        x = 1
    else:
        x = 0
    xj = 0
    while x < N:
        df[xj] = P[x]-mV[x]**2*mY[x][x]*np.cos(aY[x][x])
        df[xj+N-1] = Q[x]+mV[x]**2*mY[x][x]*np.sin(aY[x][x])
        y = 0
        while y < N:
            if x != y:
                df[xj] += -mV[x]*mV[y]*mY[x][y]*np.cos(aY[x][y]-aV[x]+aV[y])
                df[xj+N-1] += mV[x]*mV[y]*mY[x][y]*np.sin(aY[x][y]-aV[x]+aV[y])
            y += 1
        x += 1
        if x == slack:
            x += 1
        xj += 1
    
    return df


# [Back to top](#Integrated-networks-modelling---Electricity)

# ### Correction factors

# We can now calculate the correction factors and use them to update the unknown variables ($V$ and $\theta$).

# $$
# x_{k+1} = x_k - J^{-1}(x_k)f(x_k)
# $$

# In[12]:


def update_Voltages(mV, aV, J, df, N, slack):
    ''' Update volteges with correction factors '''

    # Get correction factors and update unknown variables
    dx = np.linalg.inv(J).dot(df)
    
    # Update unknown variables
    x = 0
    for Node in np.arange(N):
        if Node != slack:
            aV[Node] += dx[x]
            mV[Node] += dx[x + N - 1]
            x += 1
    return dx


# [Back to top](#Integrated-networks-modelling---Electricity)

# ### Solving the model

# We can now bring together all the methods we have created above and code Newton's method. The processes that are only needed once (e.g., calculation of $Y_{bus}$ are placed first. The processes that should be applied iteratively (e.g., calculation of Jacobian matrix) are placed within a loop which is repeated until the expected error falls below a threshold.

# In[13]:


def Newton_Elec(Elec_Network):
    # Build Y bus
    get_Admittance(Elec_Network)
    get_Ybus(Elec_Network)
    
    # Get parameters - shorter names   
    (P, Q, aY, mY, mV, aV, N, slack)=get_Polar(Elec_Network)    
    
    dx = np.inf  # Current error
    Max_it = 20  # Maximum number of iterations
    Elec_Network['Iteration'] = 0  # Current iteration
    Elec_Network['Succes'] = True  # Flag for convergence
    while np.max(np.abs(dx)) > 1e-6:
        Elec_Network['Iteration'] += 1
        
        # Get Jacobian matrix
        J = get_Jacobian(Elec_Network, aY, mY, mV, aV, N, slack)
        
        # Get vector of differences
        df = get_Power_Differences(P, Q, aY, mY, mV, aV, N, slack)
        
        # Get correction factors and update unknown variables
        dx = update_Voltages(mV, aV, J, df, N, slack)

        if Elec_Network['Iteration'] >= Max_it:
            Elec_Network['Succes'] = False
            dx = 0
            print('The model did not converge after %d iterations'%Max_it)    
           
Newton_Elec(Elec_Network)
print('Voltage magnitude:', Elec_Network['Voltage_Magnitude'], '[pu]')
print('Voltage Angle: ', Elec_Network['Voltage_Angle'], '[rad]')
print('Iterations: ', Elec_Network['Iteration'])


# [Back to top](#Integrated-networks-modelling---Electricity)

# ## Development as python tool

# The code avobe provides the means to solve the power flow equations using Newton's method, but is functionalities as a tool are limited. For starters, the model only provides the voltages for every bus across the network, whereas there are other parameters that may be of interest for us, such as the currents, power losses, etc.
# 
# Accordingly, it is convenient to add the calculations of the different parameters to our model.

# In[14]:


def get_Parameters(Elec_Network):
    ''' Calculate additional parameters '''
    Elec_Network['Current'] = []
    Elec_Network['Sending_Power'] = []
    Elec_Network['Receiving_Power'] = []
    Elec_Network['Loss'] = []
    for b in range(Elec_Network['Lines']):
        s = Elec_Network['Connectivity'][b][0]-1  # Supply side
        r = Elec_Network['Connectivity'][b][1]-1  # Sending side
        Y = complex(Elec_Network['G'][b], Elec_Network['B'][b])
        Vs = Elec_Network['Voltage_Magnitude'][s] * \
            complex(cmath.cos(Elec_Network['Voltage_Angle'][s]),
                    cmath.sin(Elec_Network['Voltage_Angle'][s]))
        Vr = Elec_Network['Voltage_Magnitude'][r] * \
            complex(cmath.cos(Elec_Network['Voltage_Angle'][r]),
                    cmath.sin(Elec_Network['Voltage_Angle'][r]))
        I = Y * (Vs - Vr)
        Ss = Vs*I.conjugate()*Elec_Network['Base']
        Sr = -Vr*I.conjugate()*Elec_Network['Base']

        Elec_Network['Current'].append(I)
        Elec_Network['Sending_Power'].append(Ss)
        Elec_Network['Receiving_Power'].append(Sr)
        Elec_Network['Loss'].append(Ss+Sr)
get_Parameters(Elec_Network)


# We can take this further by adding functionalities to the model to display all findings.

# In[15]:


def Visualize_Elec(Elec_Network, flg=[True, True, True]):
    if flg[0] and Elec_Network['Succes']:
        print('VOLTAGES  [pu] [deg]:')
        for n in range(Elec_Network['Buses']):
            V = Elec_Network['Voltage_Magnitude'][n] * \
                complex(cmath.cos(Elec_Network['Voltage_Angle'][n]),
                        cmath.sin(Elec_Network['Voltage_Angle'][n]))
            print('%2.0f) %8.4f +j %8.4f (%8.4f ∠ %8.4f)'
                  %(n+1, V.real, V.imag, abs(V), cmath.phase(V)*180/math.pi))
    if flg[1] and Elec_Network['Succes']:
        print('CURRENTS [pu] [deg]:')
        for b in range(Elec_Network['Lines']):
            s = Elec_Network['Connectivity'][b][0]
            r = Elec_Network['Connectivity'][b][1]
            I = Elec_Network['Current'][b]
            print('%2.0f-%2.0f) %8.4f +j %8.4f (%8.4f ∠ %8.4f)'
                  %(s, r, I.real, I.imag, abs(I), cmath.phase(I)*180/math.pi))
    if flg[2] and Elec_Network['Succes']:
        print('POWER  [MVA]:')
        print('      From:                To:                   Loss:')
        for b in range(Elec_Network['Lines']):
            s = Elec_Network['Connectivity'][b][0]
            r = Elec_Network['Connectivity'][b][1]
            Ss = Elec_Network['Sending_Power'][b]
            Sr = Elec_Network['Receiving_Power'][b]
            print('%2.0f-%2.0f) %8.4f +j %8.4f %8.4f +j %8.4f (%8.4f +j %8.4f)'
                  %(s, r, Ss.real, Ss.imag, Sr.real, Sr.imag, Ss.real+Sr.real,Ss.imag+Sr.imag))

Visualize_Elec(Elec_Network)


# The above is useful, but we would need to use several methods to access these functionalities. 
# 
# >It would be more convenient to have all these functionalities within the same model. 
# 
# This can be achieved by puting the models within a python class as follows.

# In[16]:


class Elec_Model:
    import numpy as np
    import scipy.sparse as sp
    import cmath, math

    def __init__(self, obj=None):
        '''Default values '''
        self.parameters = {}
        self.parameters['Base'] = 100
        self.parameters['Slack_Voltage'] = 1.0
        self.parameters['Slack_Bus'] = 1
        
        if obj is not None:
            for pars in obj.keys():
                self.parameters[pars] = obj[pars]

    def run(self):
        Newton_Elec(self.parameters)
        get_Parameters(self.parameters)

    def display(self):
        Visualize_Elec(self.parameters)


# This object, can take our Elec_Model as inputs, and automatically declares some default input data. For example, the MVA base and slack voltage magnitude are not included, so the model automatically adds them. It can also be seen that when a parameter is declared it overrides its default value, e.g., the slack bus is node 3 instead of 1.

# In[17]:


# Defining input data
Elec_Network = {}
Elec_Network['Connectivity'] = np.array([[1, 2], [1, 3], [2, 3]])
Elec_Network['R'] = [0.05, 0.05, 0.05]  # [pu]
Elec_Network['X'] = [0.5, 0.5, 0.5]  # [pu]

Elec_Network['Generation_Active'] = [50.8622, 0, 10.173]
Elec_Network['Generation_Reactive'] = [27.363, 0, 10.2181]

Elec_Network['Demand_Active'] = [0.1451, 30, 30.136]
Elec_Network['Demand_Reactive'] = [0, 15, 15]

Elec_Network['Slack_Bus'] = 3 

# Initialising model
model = Elec_Model(Elec_Network)

# Displaying data taken my the model
print(model.parameters)


# The model can solve the power system model with its `run` method, and show the results with its `display` method

# In[18]:


model.run()
model.display()


# [Back to top](#Integrated-networks-modelling---Electricity)

# In[ ]:




