import numpy as np

J = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
D = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
D1 = np.array([
    [1],
    [4],
    [7]
])
D2 = np.array([
    [2],
    [5],
    [8]
])
D3 = np.array([
    [3],
    [6],
    [9]
])

print(np.dot(J, D))
print(np.dot(J, D1))
print(np.dot(J, D2))
print(np.dot(J, D3))
print(-J)
