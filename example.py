import numpy as np

import numpy as np

# Original arrays
a = np.zeros((5, 5))
b = np.ones((2, 2))

# Creating a buffer for array b
buffered_b = np.zeros_like(a)
buffered_b[:a.shape[0]-b.shape[1]-1, a.shape[0]-b.shape[1]:] = b

print(buffered_b)
