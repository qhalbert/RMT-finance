import numpy as np


# matrix = np.random.normal(size=(2000, 2000))
# symmetric_matrix = (matrix + matrix.T) / 2
N = 1000
M = 2000
matrix = np.random.normal(size=(N, M))
W = (1/M) * (matrix.dot(np.transpose(matrix)))





# eigenvalues = []
# for i in range(100):
#     eigenvalues.append(np.linalg.eigvals(symmetric_matrix))

eigenvalues = np.linalg.eigvals(W)


print(max(eigenvalues), min(eigenvalues))

import matplotlib.pyplot as plt

plt.hist(eigenvalues, bins=200, range=(-0.1, 3), density=True)
plt.show()


