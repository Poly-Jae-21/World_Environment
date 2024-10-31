import numpy as np

MAP = np.ones(shape=(1 + 1, 1 + 1, 2))

New_MAP = np.ones(shape=(2,2))
New_MAP = New_MAP + 1

MAP[:,:,0] = New_MAP
print(MAP)

