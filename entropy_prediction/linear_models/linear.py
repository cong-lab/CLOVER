#################################################
###### assemble linear models
#################################################

import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt
import random
import pickle

# store features in X_full & labels in Y_full
X_full = np.loadtxt("TATS_v2_D14_pDOX_fuzzy-3_X.txt")
Y_full = np.loadtxt("TATS_v2_Rep1_D14_pDOX_fuzzy-3_labels.txt")


# random separate into 80% training set and 20% testing set
n, p = X_full.shape
tempidx = [i for i in range(n)]
random.shuffle(tempidx)
trainidx = tempidx[: -(int)(n / 5)]
testidx = tempidx[-(int)(n / 5) :]
X = X_full[trainidx,]
Y = Y_full[trainidx,]
X_test = X_full[testidx,]
Y_test = Y_full[testidx,]


# tag features in keys
keys = []

nctd = ["A", "T", "C", "G"]
for i in range(54):
    for j in nctd:
        keys.append(j + (str)(i + 3))

dnctd = [
    "AA",
    "AT",
    "AG",
    "AC",
    "TA",
    "TT",
    "TC",
    "TG",
    "GA",
    "GT",
    "GG",
    "GC",
    "CA",
    "CT",
    "CG",
    "CC",
]
for i in range(53):
    for j in dnctd:
        keys.append(j + (str)(i + 3))

for i in [k + 4 for k in range(52)]:
    for j in [k + i + 1 for k in range(56 - i)]:
        for t in [1, 2, 3, 4, 5, 6]:
            if j - i == t or (j - i > 6 and t == 6):
                keys.append(
                    "mh" + "(" + (str)(i) + "," + (str)(j) + ";" + (str)(t) + ")"
                )
                keys.append(
                    "GC" + "(" + (str)(i) + "," + (str)(j) + ";" + (str)(t) + ")"
                )
                keys.append(
                    "mh" + "(" + (str)(-i) + "," + (str)(-j) + ";" + (str)(t) + ")"
                )
                keys.append(
                    "GC" + "(" + (str)(-i) + "," + (str)(-j) + ";" + (str)(t) + ")"
                )

keys.append("4bp")
keys.append("g1")
keys.append("g2")


# store the correlation between features and label
with open("TATS_v2_Rep1_D14_pDOX_fuzzy-3_cor1.txt", "r") as file:
    content = file.readlines()[0].strip().split(" ")
abs_corr = {}
corr = {}
out = []  # constant features
for i, key in enumerate(keys):
    try:
        abs_corr[key] = abs((float)(content[i]))
        corr[key] = (float)(content[i])
    except:
        out.append(key)


# regression based on top 1000 correlated features
print("Regression on top 1000 correlated features...")
featureKey = sorted(abs_corr, key=abs_corr.__getitem__)[-1000:]
idx = [keys.index(k) for k in featureKey]
scalerTopCorrX = preprocessing.StandardScaler().fit(X[:, idx])  # standardization
X_1 = scalerTopCorrX.transform(X[:, idx])
X_0 = scalerTopCorrX.transform(X_test[:, idx])

ridgeRegrTopCorr = RidgeCV(
    alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10], fit_intercept=True, cv=5
).fit(
    X_1, Y
)  # ridge
print(
    "Ridge with 5-fold CV, training: R = %.2f, testing: R^2 = %.2f"
    % (np.sqrt(ridgeRegrTopCorr.score(X_1, Y)), ridgeRegrTopCorr.score(X_0, Y_test))
)

lassoRegrTopCorr = LassoCV(
    eps=0.001, n_alphas=100, fit_intercept=True, cv=5, max_iter=1000, tol=0.31
).fit(
    X_1, Y
)  # lasso
print(
    "Lasso with 5-fold CV, training: R = %.2f, testing: R^2 = %.2f"
    % (np.sqrt(lassoRegrTopCorr.score(X_1, Y)), lassoRegrTopCorr.score(X_0, Y_test))
)


# regression based on first 1000 principal components
print("Regression on first 1000 principal components...")
pca = PCA(n_components=1000)
pca.fit(X)
X_2 = pca.transform(X)
scalerPCAX = preprocessing.StandardScaler().fit(X_2)  # standardization
X_2 = scalerPCAX.transform(X_2)
X_0 = scalerPCAX.transform(pca.transform(X_test))

ridgeRegrPCA = RidgeCV(
    alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10], fit_intercept=True, cv=5
).fit(
    X_2, Y
)  # ridge
print(
    "Ridge with 5-fold CV, training: R = %.2f, testing: R^2 = %.2f"
    % (np.sqrt(ridgeRegrPCA.score(X_2, Y)), ridgeRegrPCA.score(X_0, Y_test))
)

lassoRegrPCA = LassoCV(eps=0.001, n_alphas=100, fit_intercept=True, cv=5).fit(
    X_2, Y
)  # lasso
print(
    "Lasso with 5-fold CV, training: R = %.2f, testing: R^2 = %.2f"
    % (np.sqrt(lassoRegrPCA.score(X_2, Y)), lassoRegrPCA.score(X_0, Y_test))
)


# regression based on first 500 principal components
print("Regression on first 500 principal components...")
pca_less = PCA(n_components=500)
pca_less.fit(X)
X_3 = pca_less.transform(X)
scalerPCAlessX = preprocessing.StandardScaler().fit(X_3)  # standardization
X_3 = scalerPCAlessX.transform(X_3)
X_0 = scalerPCAlessX.transform(pca_less.transform(X_test))

ridgeRegrPCAless = RidgeCV(
    alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10], fit_intercept=True, cv=5
).fit(
    X_3, Y
)  # ridge
print(
    "Ridge with 5-fold CV, training: R = %.2f, testing: R^2 = %.2f"
    % (np.sqrt(ridgeRegrPCAless.score(X_3, Y)), ridgeRegrPCAless.score(X_0, Y_test))
)

lassoRegrPCAless = LassoCV(eps=0.001, n_alphas=100, fit_intercept=True, cv=5).fit(
    X_3, Y
)  # lasso
print(
    "Lasso with 5-fold CV, training: R = %.2f, testing: R^2 = %.2f"
    % (np.sqrt(lassoRegrPCAless.score(X_3, Y)), lassoRegrPCAless.score(X_0, Y_test))
)


# regression based on the full data
print("Regression on all features...")
scalerX = preprocessing.StandardScaler().fit(X)  # standardization
X_4 = scalerX.transform(X)
X_0 = scalerX.transform(X_test)

lassoRegr = LassoCV(
    eps=0.001, n_alphas=100, fit_intercept=True, cv=5, max_iter=1000, tol=0.097
).fit(
    X_4, Y
)  # tol = 0.065 if 3-fold
print(
    "Lasso with 5-fold CV, training: R = %.2f, testing: R^2 = %.2f"
    % (np.sqrt(lassoRegr.score(X_4, Y)), lassoRegr.score(X_0, Y_test))
)


# create a class to store the results
class mylinear(preprocessing.StandardScaler, PCA, LassoCV, RidgeCV):
    pass

    def __init__(
        self,
        feature_tag,
        abs_corr,
        corr,
        out,
        idx,
        scalerTopCorrX,
        ridgeRegrTopCorr,
        lassoRegrTopCorr,
        pca,
        scalerPCAX,
        ridgeRegrPCA,
        lassoRegrPCA,
        pca_less,
        scalerPCAlessX,
        ridgeRegrPCAless,
        lassoRegrPCAless,
        scalerX,
        lassoRegr,
    ):
        self.feature_tag = feature_tag
        self.abs_corr = abs_corr
        self.corr = corr
        self.out = out
        self.idx = idx
        self.scalerTopCorrX = scalerTopCorrX
        self.ridgeRegrTopCorr = ridgeRegrTopCorr
        self.lassoRegrTopCorr = lassoRegrTopCorr
        self.pca = pca
        self.scalerPCAX = scalerPCAX
        self.ridgeRegrPCA = ridgeRegrPCA
        self.lassoRegrPCA = lassoRegrPCA
        self.pca_less = pca_less
        self.scalerPCAlessX = scalerPCAlessX
        self.ridgeRegrPCAless = ridgeRegrPCAless
        self.lassoRegrPCAless = lassoRegrPCAless
        self.scalerX = scalerX
        self.lassoRegr = lassoRegr


# dump in pickle
linear = mylinear(
    keys,
    abs_corr,
    corr,
    out,
    idx,
    scalerTopCorrX,
    ridgeRegrTopCorr,
    lassoRegrTopCorr,
    pca,
    scalerPCAX,
    ridgeRegrPCA,
    lassoRegrPCA,
    pca_less,
    scalerPCAlessX,
    ridgeRegrPCAless,
    lassoRegrPCAless,
    scalerX,
    lassoRegr,
)
pickle.dump(linear, open("linear.p", "wb"))
