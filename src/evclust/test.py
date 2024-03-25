import numpy
import numpy as np
import matplotlib.pyplot as plt
from evclust.ecm import ecm
from evclust.fwecm import fwecm
from evclust.utils import ev_summary, ev_plot, ev_pcaplot
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# Non-specificity values
def calculate_non_specificity(cluster_model):
    m = cluster_model['mass']
    F = cluster_model['F']
    c = F.shape[1]
    card = np.sum(F[1:F.shape[0], :], axis=1)

    log_card = np.log2(card)
    tmp = np.tile(log_card.transpose(), (m.shape[0], 1))
    m_log_card = m[:, :-1] * tmp

    mvide = m[:, -1][:, np.newaxis]
    tmp2 = mvide * np.log2(c)
    tmp3 = np.tile(tmp2, (1, m.shape[1] - 1))

    non_specificity = m_log_card + tmp3
    object_non_specificity = np.sum(non_specificity, axis=1)

    print(f"Maximum Non-specificity value: {max(object_non_specificity)}")
    print(f"Minimum Non-specificity value: {min(object_non_specificity)}")
    print(f"Average Non-specificity value: {np.mean(object_non_specificity)}")


# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
# scaler = MinMaxScaler(feature_range=(0, 1))
X = iris.data.features
y = iris.data.targets

labels_encoder = LabelEncoder()
numeric_labels = labels_encoder.fit_transform(y['class'])
df = pd.concat([X, y], axis=1)

# Feature weighted ECM clustering
W = np.array([[0.06947213, 0.21751942, 0.0691499, 0.64385855],
              [0.06810348, 0.26238236, 0.06911913, 0.60039504],
              [0.25188023, 0.1001694, 0.47634386, 0.17160651]])
# W = None
c = 3
model = fwecm(x=X, c=c, W=W, beta=2, alpha=1, delta=100, epsi=1e-5, ntrials=10)
# model = ecm(x=X, c=c, beta=2, alpha=1, delta=100, epsi=1e-5, ntrials=1)
print(f"Jbest: {model['crit']}")
print(f"Centers: \n {model['g']}")

true_labels = numeric_labels
Y_betP = model['betp']
predicted_labels = np.argmax(Y_betP, axis=1)

# Compute the Adjusted Rand Index (ARI)
ari = adjusted_rand_score(true_labels, predicted_labels)
print("----------Feature weighted ECM----------")
print(f"True labels: {numeric_labels}")
print(f"Predicted labels: {predicted_labels}")
print(f"Adjusted Rand Index (ARI): {ari}")

# # Traditional ECM
# model = ecm(x=X, c=3, beta=2, alpha=0.1, delta=19, ntrials=1)
# # Compute the Adjusted Rand Index (ARI)
# true_labels = numeric_labels
# Y_betP = model['betp']
# predicted_labels = np.argmax(Y_betP, axis=1)
#
# ari = adjusted_rand_score(true_labels, predicted_labels)
# print("----------Traditional ECM----------")
# print(f"True labels: {numeric_labels}")
# print(f"Predicted labels: {predicted_labels}")
# print(f"Adjusted Rand Index (ARI): {ari}")
