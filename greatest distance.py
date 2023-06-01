import numpy as np
from simulation import sim

# Linear interpolation
def lerp(a: float, b: float, t: float):
    return (1 - t) * a + b * t


angles = np.linspace(0, 90, 181)[::-1]  # Array of angles
print(angles)
magnitude = 200                         # Magnitude (v0)

# mass, radius, drag, air density, gravity
params=[32, 0.1, 0.47, 1.293, np.array([0., -9.81])]

dt = 0.01       # Timestep in simulations
idx = 0         # Angle index
stop = False    # Force stop
x = -1          # Fake first position
dists = []      # List to store launch distances

# Looping through angles until furthest distance is found
while not stop and idx < len(angles):
    angle = angles[idx]                     # Picking angle
    res = sim(angle, magnitude, params, dt) # Simulate with imported function
    s, v, a, t, sl, vl, al = res            # Unpack result tuple

    # Interpolate to find y=0
    P1 = sl[-2]             # Second final position (above y=0)
    P2 = s                  # Final position (below y=0)
    y1 = P1[1]              # y component of second final position
    y2 = abs(P2[1])         # y component of final component
    #
    t = y1 / (y2 + y1)      # Finding appropriate interpolation value (for finding y=0)
    P = lerp(sl[-2], s, t)  # Interpoalting from P1 to P2 (to find y=0)

    x1 = P[0]                       # x component of interpolated point (distance traveled at y=0)
    if x > x1: stop = True          # Stop if distance is shorter than previous distance
    else: print(f'{angle}: {x1}')   # Otherwise print the angle
    x = x1                          # Update to latest distance

    dists.append(x) # Append the distance
    idx += 1        # Increment index (go to next angle)

dists.pop() # Remove last found distance
