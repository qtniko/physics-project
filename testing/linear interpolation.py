"""
Using linear interpolation to find y=0
"""

import numpy as np

def lerp(a: float, b: float, t: float):
    return (1 - t) * a + b * t

P1 = np.array([1, 2], dtype=float)
P2 = np.array([4, -5], dtype=float)
y1 = P1[1]
y2 = abs(P2[1])

t = y1 / (y2 + y1)
P = lerp(P1, P2, t)

print(tuple(P))