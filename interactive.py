import pygame
import numpy as np
import sys

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

def sim(angle, magnitude, params, dt=0.01):
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

    # Return values
    # (final position, final velocity, final acceleration, time elapsed, all positions, all velocities, all accelerations)
    # NOTE: The final position will not be at y=0
    #       One option to find roughly y=0 would be to interpolate appropriately between the two last positions..
    return (s, v, a, t, s_list, v_list, a_list)

# Linear interpolation
def lerp(a: float, b: float, t: float):
    return (1 - t) * a + b * t

def best_angle(magnitude, params, dt=0.01, printstuff=False, anglesims=19, max_dist=None):
    angles = np.linspace(0, 90, anglesims)[::-1]  # Array of angles
    if printstuff: print()

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
        elif printstuff: print(f'{angle}: {x1}')   # Otherwise print the angle
        x = x1                          # Update to latest distance

        dists.append(x) # Append the distance
        idx += 1        # Increment index (go to next angle)

        # Draw
        screen.fill('#000000')
        if showMax and max_dist: pygame.draw.line(screen, '#dd00ff', (max_dist/4, 0), (max_dist/4, HEIGHT))
        pygame.draw.line(screen, '#55cc55', (0, HEIGHT-200), (WIDTH*2, HEIGHT-200))
        pygame.draw.circle(screen, '#55ffff', (target/4, HEIGHT-200), 5)
        for didx, p2 in enumerate(sl):
            if didx == 0: continue
            p2 = p2.copy()/4
            p2 = np.array((p2[0]*2, HEIGHT-200)) - p2
            p1 = sl[didx-1].copy()/4
            p1 = np.array((p1[0]*2, HEIGHT-200)) - p1
            pygame.draw.line(screen, '#dd00ff', p1, p2, 1)
        pygame.display.flip()

    dists.pop() # Remove last found distance
    return (dists[-1], dists)

###################
def calculate(magnitude, params, dt, target, tolerance, printstuff=False, anglesims=19):
    # Print params
    print(f'\nTarget distance: {target}\nTolerance: {tolerance}\nStart velocity: {magnitude}')
    if printstuff: print()
    # print(f'params:\n\t\tmass: {params[0]}\n\t\tradius: {params[1]}\n\t\tdrag: {params[2]}\n\t\tair density: {params[3]}\n\t\tgravity: {params[4]}\n')

    # Finding greatest distance
    angles = np.linspace(0, 90, anglesims)[::-1]  # Array of angles (181)
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
        # else: print(f'{angle}: {x1}')   # Otherwise print the angle
        x = x1                          # Update to latest distance

        dists.append((angle, x)) # Append the distance

        # Print current result
        if printstuff: print(f'{angle} | {magnitude} | {x}')

        # Draw
        screen.fill('#000000')
        if showMax: pygame.draw.line(screen, '#dd00ff', (max_dist/4, 0), (max_dist/4, HEIGHT))
        pygame.draw.line(screen, '#55cc55', (0, HEIGHT-200), (WIDTH*2, HEIGHT-200))
        pygame.draw.circle(screen, '#55ffff', (target/4, HEIGHT-200), 5)
        for didx, p2 in enumerate(sl):
            if didx == 0: continue
            p2 = p2.copy()/4
            p2 = np.array((p2[0]*2, HEIGHT-200)) - p2
            p1 = sl[didx-1].copy()/4
            p1 = np.array((p1[0]*2, HEIGHT-200)) - p1
            pygame.draw.line(screen, '#ffff00', p1, p2, 1)
        pygame.display.flip()

        if x > target: break # If we pass the target, stop

        idx += 1 # Increment index (go to next angle)
        if idx == len(angles): break # If at final index, stop

    # Check if target is within reach
    if target > dists[-1][1]:
        print(f'\nCannot launch {target} meters! (max: {round(dists[-1][1])})')
        return None, None
    
    if printstuff: print()

    A1 = dists[-2] # Angle closest to target (left)
    A2 = dists[-1] # Angle closest to target (right)

    # print(f'{A2[0]} < [target angle] < {A1[0]}\n')

    current = A2[1] # Set current distance to A2's distance

    return A1, A2

    # Run until we find an angle that hits within set tolerance
    # The second turm is purely cosmetic and makes the result look nicer (also takes more iterations)
    # while abs(target - current) > tolerance or current < target:
def findnext(target, A1, A2, tolerance, printstuff=False):
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
    if printstuff: print(f'{angle} | {magnitude} | {current}')

    # If we are to the left of the target, approach from the left
    if current < target:
        A1 = (angle, current)
    # If we are to the right of the target, approach from the right
    elif current > target:
        A2 = (angle, current)
    
    return angle, current, A1, A2, (abs(target - current) < tolerance), s_list

    # Print final result
    # print(f'\nTARGET HIT: {angle} | {magnitude} | {current}')

    # Simulate with corrent angle, unpack result
    res = sim(angle, magnitude, params, dt)
    s, v, a, t, s_list, v_list, a_list = res
###################

# Parameters
target = 1500                                           # Target x value
magnitude = 200                                         # Start velocity
params=[32, 0.1, 0.47, 1.293, np.array([0., -9.81])]    # Mass, radius, drag, air density, graivty
tolerance = 0.0001                                      # Distance tolerance (0.0001)
dt = 0.01                                               # Time step
angle = 45
editTraj = True
trajClr = '#ff5555'
anglesims = 19

A1 = None
A2 = None

####################

HEIGHT, WIDTH = SIZE = (700, 600)
selected = 'magnitude'

pygame.init()
screen = pygame.display.set_mode(SIZE)
clock = pygame.time.Clock()

res = sim(angle, magnitude, params, dt)
s_, v_, a_, t_, sl_, vl_, al_ = res
# y1 = sl_[-2][1]
# y2 = s_
# t = y1 / (y2 + y1)
# P = lerp(sl_[-2], s_, t)
# target = round(P[0])

printstuff = False
showMax = True
max_dist, bl_ = best_angle(magnitude, params, dt, printstuff, anglesims)

print(f'\nTarget > {target}\nAngle > {angle}\nMagnitude > {magnitude}')

while 1:
    # A/D >> change target pos
    # UP/DOWN >> change angle
    # LEFT/RIGHT >> change magnitude
    update = False
    keys = pygame.key.get_pressed()
    if editTraj and keys[pygame.K_a]: target = max(target - 20, 10)
    if editTraj and keys[pygame.K_d]: target = min(target + 20, np.floor(max_dist))
    if editTraj and keys[pygame.K_s]: target = max(target - 10, 10)
    if editTraj and keys[pygame.K_w]: target = min(target + 10, np.floor(max_dist))
    if editTraj and keys[pygame.K_q]: target = max(target - 1, 10)
    if editTraj and keys[pygame.K_e]: target = min(target + 1, np.floor(max_dist))
    if editTraj and keys[pygame.K_LEFT]: magnitude = max(magnitude - 1, 1); update = True
    if editTraj and keys[pygame.K_RIGHT]: magnitude += 1;                   update = True
    if editTraj and keys[pygame.K_DOWN]: angle = max(angle - 1, 0);         update = True
    if editTraj and keys[pygame.K_UP]: angle = min(angle + 1, 90);          update = True

    if update:
        res = sim(angle, magnitude, params, dt)
        s_, v_, a_, t_, sl_, vl_, al_ = res
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_v and editTraj:
                print(f'\nTarget > {target}\nAngle > {angle}\nMagnitude > {magnitude}')
            if event.key == pygame.K_p:
                printstuff = not printstuff
                print(f'\nPrint stuff: {printstuff}')
            if event.key == pygame.K_c:
                editTraj = False
                trajClr = '#55ffff'
                A1, A2 = calculate(magnitude, params, dt, target, tolerance, printstuff, anglesims)
                if not (A1 and A2):
                    editTraj = True
                    trajClr = '#ff5555'
            if event.key == pygame.K_m:
                max_dist, bl_ = best_angle(magnitude, params, dt, printstuff, anglesims, max_dist)
                if target > max_dist: target = np.floor(max_dist)
                showMax = True
            if event.key == pygame.K_n:
                if max_dist: showMax = not showMax
    
    if not editTraj:
        a, cur, a1, a2, found, sl_ = findnext(target, A1, A2, tolerance, printstuff)
        A1, A2 = a1, a2
        if found:
            A1 = A2 = None
            editTraj = True
            trajClr = '#ff5555'
            print(f'\nHit: {cur}\nAngle: {a}\nMagnitude: {magnitude}')
        angle = a
    
    screen.fill('#000000')

    if showMax: pygame.draw.line(screen, '#dd00ff', (max_dist/4, 0), (max_dist/4, HEIGHT))
    pygame.draw.line(screen, '#55cc55', (0, HEIGHT-200), (WIDTH*2, HEIGHT-200))

    pygame.draw.circle(screen, '#55ffff', (target/4, HEIGHT-200), 5)

    for idx, p2 in enumerate(sl_):
        if idx == 0: continue
        p2 = p2.copy()/4
        p2 = np.array((p2[0]*2, HEIGHT-200)) - p2
        p1 = sl_[idx-1].copy()/4
        p1 = np.array((p1[0]*2, HEIGHT-200)) - p1
        pygame.draw.line(screen, trajClr, p1, p2, 1)
    
    pygame.display.flip()
    clock.tick(60)
