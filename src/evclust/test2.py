import numpy as np
from evclust.fcm.fcm_keller2000 import fcm
from ucimlrepo import fetch_ucirepo

# fetch dataset - IRIS
# iris = fetch_ucirepo(id=53)
#
# # data (as pandas dataframes)
# X = iris.data.features
# y = iris.data.targets
#
# c = 3
# model = fcm(X, c, verbose=True)
#
# print(f"Fuzzy partition: {model['fuzzy_part']}")

arr = np.array([
    [4.8, 5.0, 3.0, 2.0],
    [2.0, 3.0, 4.0, 5.0],
    [5.0, 5.0, 2.0, 3.0],
    [1.0, 5.0, 3.0, 1.0],
    [1.0, 4.9, 5.0, 1.0],
])
beta = 0.26
n = arr.shape[0]
distance = np.zeros((n, n))
for i in range(arr.shape[0]):
    for j in range(arr.shape[0]):
        distance[i, j] = np.linalg.norm(arr[i] - arr[j])

print(distance)

similarity_sum = 0
iter = 0
for i in range(arr.shape[0]):
    for j in range(i):
        print(f"[{i}, {j}]")
        similarity_sum += 1 / (1 + beta * distance[i, j])

similarity_sum *= 2 / (n * (n - 1))

print("similarity_sum", similarity_sum)
