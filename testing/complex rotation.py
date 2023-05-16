"""
Easy rotation using complex numbers
"""

import numpy as np

P = complex(1, 2) # Point(1, 2)

Q = P * np.exp(np.pi*complex(0, 1)) # Rotate P π radians ccw about the origin

Q = complex(round(np.real(Q), 10), round(np.imag(Q), 10)) # Rounding Q cuz oogli boogli

print(f'{P = }')
print(f'{Q = }')

# >>> P = (1+2j)
# >>> Q = (-1-2j)
