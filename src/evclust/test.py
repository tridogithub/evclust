import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import pandas as pd

# Generate dataset with 2 crescent-shaped clusters
X, y = make_moons(n_samples=500, noise=0.1, random_state=43)


# Function to rotate points
def rotate(X, angle):
    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return X.dot(rotation_matrix)


# Rotate one of the crescents by 180 degrees to make them face each other
X[y == 1] = rotate(X[y == 1], 90)
X[y == 0] = rotate(X[y == 0], 90)

n = X[y == 1].shape[0]
shift = np.tile([1.85, 1], (n, 1))
X[y == 1] = X[y == 1] + shift

X[X[:, 1] > 0.5, 1] = X[X[:, 1] > 0.5, 1] - 0.25
X[X[:, 1] < -0.5, 1] = X[X[:, 1] < -0.5, 1] + 0.25

# size = X[(np.array(X[:, 1] > 0.4) & np.array(X[:, 0] > 1.1))].shape[0]
# tmp1 = np.tile([-0.15, 0.4], (size, 1))
# X[(np.array(X[:, 1] > 0.4) & np.array(X[:, 0] > 1.1))] = X[(np.array(X[:, 1] > 0.4) & np.array(X[:, 0] > 1.1))] - tmp1
#
# size = X[(np.array(X[:, 1] < -0.4) & np.array(X[:, 0] > 1.1))].shape[0]
# tmp1 = np.tile([-0.15, -0.4], (size, 1))
# X[(np.array(X[:, 1] < -0.4) & np.array(X[:, 0] > 1.1))] = X[(np.array(X[:, 1] < -0.4) & np.array(X[:, 0] > 1.1))] - tmp1
#
# size = X[(np.array(X[:, 1] > 0.3) & np.array(X[:, 0] < 1.2))].shape[0]
# tmp1 = np.tile([0.2, 0.4], (size, 1))
# X[(np.array(X[:, 1] > 0.3) & np.array(X[:, 0] < 1.2))] = X[(np.array(X[:, 1] > 0.3) & np.array(X[:, 0] < 1.2))] - tmp1
#
# size = X[(np.array(X[:, 1] < -0.3) & np.array(X[:, 0] < 1.2))].shape[0]
# tmp1 = np.tile([0.2, -0.4], (size, 1))
# X[(np.array(X[:, 1] < -0.3) & np.array(X[:, 0] < 1.2))] = X[(np.array(X[:, 1] < -0.3) & np.array(X[:, 0] < 1.2))] - tmp1
#
# shift = np.tile([-0.3, 0], (n, 1))
# X[y == 1] = X[y == 1] + shift
#
df = pd.DataFrame(data=X, columns=['Feature 1', 'Feature 2'])
df['Label'] = y
df.to_csv(
    'D:\\vtdo\projects\evclust\src\evclust\datasets\crescent2D.csv',
    index=False, header=False)

# Plot the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Crescent-shaped clusters facing each other')
plt.show()
