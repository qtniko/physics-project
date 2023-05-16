import numpy as np
from simulation import sim
import matplotlib.pyplot as plt
import sys

# Linear interpolation
def lerp(a: float, b: float, t: float):
    return (1 - t) * a + b * t

# Parameters
target = 333                                            # Target x value
magnitude = 200                                         # Start velocity
params=[32, 0.1, 0.47, 1.293, np.array([0., -9.81])]    # Mass, radius, drag, air density, graivty
tolerance = 0.0001                                      # Distance tolerance
dt = 0.01                                               # Timestep used in simulations

# Print params
print(f'target distance: {target}\ntolerance: {tolerance}\nstart velocity: {magnitude}\n')
# print(f'params:\n\t\tmass: {params[0]}\n\t\tradius: {params[1]}\n\t\tdrag: {params[2]}\n\t\tair density: {params[3]}\n\t\tgravity: {params[4]}\n')

# Finding greatest distance
angles = np.linspace(0, 90, 181)[::-1]  # Array of angles
idx = 0                                 # Angle array index
stop = False                            # Force stop
x = -1                                  # Fake first position
dists = []                              # List to store angles and their launch distance
while not stop:                             # Start loop
    angle = angles[idx]                     # Pick angle
    res = sim(angle, magnitude, params, dt) # Simulate
    s, v, a, t, sl, vl, al = res            # Unpack result

    P1 = sl[-2]             # Second final position
    P2 = s                  # Final position
    y1 = P1[1]              # y component of second final position
    y2 = abs(P2[1])         # y component of final component
    
    t = y1 / (y2 + y1)      # Finding appropriate interpolation value (for finding y=0)
    P = lerp(sl[-2], s, t)  # Interpoalting from P1 to P2 (to find y=0)

    x1 = P[0]                       # x component of interpolated point (distance traveled at y=0)
    if x > x1: break                # Stop if distance is shorter than previous distance
    else: print(f'{angle}: {x1}')   # Otherwise print the angle
    x = x1                          # Update to latest distance

    dists.append((angle, x)) # Append the distance

    if x > target: break # If we pass the target, stop

    idx += 1 # Increment index (go to next angle)
    if idx == len(angles): break # If at final index, stop

print()

# Check if target is within reach
if target > dists[-1][1]:
    print(f'Cannot launch {target} meters! (max: {round(dists[-1][1])})')
    sys.exit()

A1 = dists[-2] # Angle closest to target (left)
A2 = dists[-1] # Angle closest to target (right)

print(f'{A2[0]} < [target angle] < {A1[0]}\n')

current = A2[1] # Set current distance to A2's distance

# Run until we find an angle that hits within set tolerance
# The second turm is purely cosmetic and makes the result look nicer (also takes more iterations)
while abs(target - current) > tolerance or current < target:

    # Calculate new guesstimate angle
    angle = (A1[0] + A2[0]) * 0.5

    # Simulate and unpack result
    res = sim(angle, magnitude, params, dt)
    s, v, a, t, s_list, v_list, a_list = res
    
    # Interpolate to find x at y=0
    P1 = s_list[-2]
    P2 = s
    y1 = P1[1]
    y2 = abs(P2[1])
    #
    t = y1 / (y2 + y1)
    P = lerp(s_list[-2], s, t)
    #
    x = P[0]

    # Update current distance
    current = x

    # Print current result
    print(f'{angle} | {magnitude} | {current}')

    # If we are to the left of the target, approach from the left
    if current < target:
        A1 = (angle, current)
    # If we are to the right of the target, approach from the right
    elif current > target:
        A2 = (angle, current)

# Print final result
print(f'\nTARGET HIT: {angle} | {magnitude} | {current}')

# Simulate with corrent angle, unpack result
res = sim(angle, magnitude, params, dt)
s, v, a, t, s_list, v_list, a_list = res

# Plot position graph with vertical targetline
plt.axis = 'equal'
plt.title = 'Yeet'
plt.xlabel = '$x$ / m'
plt.ylabel = '$y$ / m'
plt.grid(True)
plt.axvline(target, color='red')
plt.plot(s_list[:,0], s_list[:,1])
plt.show()