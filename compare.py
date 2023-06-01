import numpy as np
from simulation import sim
import matplotlib.pyplot as plt

# Plot and show data
def display(data):
    plt.axis = 'equal'
    plt.title = 'Yeet'
    plt.xlabel = '$x$ / m'
    plt.ylabel = '$y$ / m'
    plt.grid(True)
    for tpl in data:
        s, v, a, t, s_list, v_list, a_list = tpl
        plt.plot(s_list[:,0], s_list[:,1])
    plt.show()

# mass, radius, drag, air density, gravity
params=[0.1, 0.05, 0.47, 1.293, np.array([0., -9.81])]
dt = 0.001 # Time step in simulations

data = []                       # List for data storage
angles = np.linspace(0, 90, 19) # List of angles
magnitude = 200                 # Start velocity

# Simulate and store data from all angles
for angle in angles:
    data.append(sim(angle, magnitude, params, dt))

# Show data
display(data)
