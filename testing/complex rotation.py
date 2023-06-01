"""
Easy rotation using complex numbers
"""

import numpy as np

P = complex(1, 2) # Point(1, 2)

Q = P * np.exp(complex(0, np.pi)) # Rotate P π radians ccw about the origin

print(f'{P = }')
print(f'{Q = }')

# >>> P = (1+2j)
# >>> Q = (-1-2j)
