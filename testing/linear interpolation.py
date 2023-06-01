"""
Using linear interpolation to find y=0
"""

import numpy as np

def lerp(a, b, t):
    return (1 - t) * a + b * t

A = np.array([1, 2], dtype=float)
B = np.array([4, -5], dtype=float)

t = A[1] / abs(B[1] - A[1])
P = lerp(A, B, t)

print(tuple(P))
