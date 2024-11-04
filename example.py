import numpy as np

class K_():
    def __init__(self):
        self.a = np.array([[1, 2, 3],
                 [3,4,5],
                 [4,5,6]])
        self.iter = 0
    def ss(self):
        if self.iter == 0:
            c = np.array([[7, 55, 6]])
            self.a = np.append(self.a, c, axis=0)
            print(self.a)

        k = self.a[-1]
        return k

a = np.array([[1, 2]])

b = np.array([[0]])

c= np.hstack((a, b))
print(c)
