import numpy as np
import matplotlib.pyplot as plt

# Function calculating the air resistance force vector
def air_resistance(density, v, A, drag):
    v_norm = (v.dot(v))**0.5                                    # Magnitude of velocity vector
    v_unit = v / v_norm                                         # Unit velocity vector
    air_resistance = 0.5 * density * v_norm*v_norm * A * drag   # Air resistance force
    return -air_resistance * v_unit                             # Return: air resistance force vector

# Function calculating the acceleration vector
def Acceleration(g, rho, v, A, Cd, m):
    # ΣF = G + A = m * a
    # a = (G + A) / m = g + A/m
    return g + air_resistance(rho, v, A, Cd)/m

def sim(angle, magnitude, params, dt=0.01, showplot=False):
    # Parameters
    # m - mass of projectile
    # r - radius of projectile
    # Cd - drag coefficient of projectile
    # rho - air density
    # g - gravitational acceleration vector
    # A - equatorial cross section of projectile
    m, r, Cd, rho, g = params
    A = np.pi * r*r
    
    # Rotate start velocity vector (complex numbers go brr)
    P = complex(magnitude, 0)                           # Scaled
    Q = P * np.exp(np.radians(angle)*complex(0, 1))     # Rotated
    v = np.array([np.real(Q), np.imag(Q)], dtype=float) # Vectorized

    # Initiation
    s = np.array([0, 0], dtype=float) # Position
    a = np.array([0, 0], dtype=float) # Acceleration
    #
    s_list = [s.copy()] # Position over time
    v_list = [v.copy()] # Velocity over time
    a_list = [a.copy()] # Acceleration over time
    #
    t = 0

    # Simulate throw
    while s[1] > 0 or t == 0:
        # Update values
        a = Acceleration(g, rho, v, A, Cd, m)   # Acceleration
        v += a * dt                             # Velocity
        s += v * dt                             # Position
        t += dt                                 # Time
        
        # Store updated values
        s_list = np.concatenate([s_list, [s.copy()]])
        v_list = np.concatenate([v_list, [v.copy()]])
        a_list = np.concatenate([a_list, [a.copy()]])

    # Plot position graph (optional)
    if showplot:
        plt.axis('equal')
        plt.title('Yeet')
        plt.xlabel('$x$ / m')
        plt.ylabel('$y$ / m')
        plt.plot(s_list[:,0], s_list[:,1], color='blue')
        plt.grid(True)
        plt.show()

    # Return values
    # (final position, final velocity, final acceleration, time elapsed, all positions, all velocities, all accelerations)
    # NOTE: The final position will not be at y=0
    #       One option to find roughly y=0 would be to interpolate appropriately between the two last positions..
    return (s, v, a, t, s_list, v_list, a_list)



# -TEST- #
if __name__=='__main__':
    res = sim(angle=45, magnitude=200, params=[0.1, 0.05, 0.47, 1.293, np.array([0., -9.81])], dt=0.001, showplot=True)