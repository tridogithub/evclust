import numpy
import numpy as np
import matplotlib.pyplot as plt
from evclust.ecm import ecm
from evclust.fwecm import fwecm
from evclust.wecm_new import wecm
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

### The initial things produce 0.7445 of ARI value
# W = np.array([
# [0.09743954 0.3559397  0.05490756 0.49171319]
#  [0.08977217 0.11029495 0.1650923  0.63484057]
#  [0.24106103 0.13353718 0.0570726  0.56832919]
# ])
#
# g0 = np.array([
# [6.69837057 3.07635634 5.87417793 2.21114012]
#  [4.95830926 3.27085993 1.47396326 0.23649202]
#  [6.07134779 2.63496741 4.81821913 1.46395735]
# ])

W = None
g0 =None
c = 3
# model = fwecm(x=X, c=c, W=W, beta=2, alpha=1, delta=100, epsi=1e-5, ntrials=10)
model = wecm(x=X, c=c, g0=g0, W=W, beta=2, alpha=1, delta=100, epsi=1e-2, ntrials=10)
# model = ecm(x=X, c=c, g0=g0, beta=2, alpha=1, delta=100, epsi=1e-3, ntrials=1)
print(f"Jbest: {model['crit']}")
print(f"Centers: \n {model['g']}")
print(f"Final weights: {model['W']}")

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
