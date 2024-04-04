import numpy
import numpy as np
import matplotlib.pyplot as plt
from evclust.ecm import ecm
from evclust.fwecm import fwecm
from evclust.wecm_new import wecm
from evclust.utils import ev_summary, ev_plot, ev_pcaplot, calculate_non_specificity, ev_plot_PCA
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from evclust.datasets import load_letters, load_seeds

# fetch dataset
# df = fetch_ucirepo(id=53)
# df = pd.read_csv("./datasets/2c2dDataset.csv", header=None)
# df = load_letters()
# df = load_seeds()
# df = df.drop(columns=['C'])
#
# # data (as pandas dataframes)
# y = pd.DataFrame(df.iloc[:, -1])
# X = df.iloc[:, :-1]

iris = fetch_ucirepo(id=52)

# data (as pandas dataframes)
X = iris.data.features
X = X.fillna(0)
X = X.astype(float)
y = iris.data.targets

labels_encoder = LabelEncoder()
numeric_labels = labels_encoder.fit_transform(y['Class'])

# Feature weighted ECM clustering

best_model = None
best_ari = 0
for i in range(30):
    W = None
    g0 = None
    c = 2
    # model = fwecm(x=X, c=c, W=W, beta=2, alpha=1, delta=100, epsi=1e-5, ntrials=10)
    model = wecm(x=X, c=c, g0=g0, W=W, beta=2, alpha=1, delta=100, epsi=1e-3, ntrials=10)
    # model = ecm(x=X, c=c, g0=g0, beta=2, alpha=1, delta=100, epsi=1e-3, ntrials=1)

    true_labels = numeric_labels
    Y_betP = model['betp']
    predicted_labels = np.argmax(Y_betP, axis=1)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    print(f"{i + 1}-ARIIIIII: {ari}")
    if ari > best_ari:
        best_ari, best_model = ari, model
model = best_model
print(f"Jbest: {model['crit']}")
print(f"Centers: \n {model['g']}")
print(f"Final weights: \n{model['W']}")

true_labels = numeric_labels
Y_betP = model['betp']
predicted_labels = np.argmax(Y_betP, axis=1)

# Compute the Adjusted Rand Index (ARI)
ari = adjusted_rand_score(true_labels, predicted_labels)
print("----------Feature weighted ECM----------")
print(f"True labels: {numeric_labels}")
print(f"Predicted labels: {predicted_labels}")
print(f"Adjusted Rand Index (ARI): {ari}")
calculate_non_specificity(model)

# ev_plot_PCA(model, X=X)
# ev_pcaplot(data=X, x=model, normalize=False, cex=30)
