#################################################
###### predict the entropy of designs
#################################################

import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
import random

# use lassoPCAless

X_full = np.loadtxt('TATS_v2_D14_pDOX_fuzzy-3_X.txt')
Y_full = np.loadtxt('TATS_v2_D14_pDOX_fuzzy-3_labels.txt')

n,p = X_full.shape
tempidx = [i for i in range(n)]
random.shuffle(tempidx)
trainidx = tempidx[:-(int)(n/5)]
testidx = tempidx[-(int)(n/5):]
X = X_full[trainidx,]
Y = Y_full[trainidx,]
X_test = X_full[testidx,]
Y_test = Y_full[testidx,]

pca_less =  PCA(n_components = 500)
pca_less.fit(X)
X_3 = pca_less.transform(X)
scalerPCAlessX = preprocessing.StandardScaler().fit(X_3) #standardization
X_3 = scalerPCAlessX.transform(X_3)
X_0 = scalerPCAlessX.transform(pca_less.transform(X_test))

Regr = RidgeCV(alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10], fit_intercept = True, cv = 5).fit(X_3, Y)
print('Ridge with 5-fold CV, training: R = %.2f, testing: R^2 = %.2f' %( np.sqrt(Regr.score(X_3, Y)),Regr.score(X_0, Y_test)))

predict = open("***.txt","w")
with open("****.txt","r") as file:
    content = file.readlines()
    content = [x.strip() for x in content]
    i = 0
    while i < len(content):
        design = content[i][1:]
        i += 1
        x = np.array([(float)(s) for s in content[i].split(' ')]).reshape(1,-1)
        y = Regr.predict(scalerPCAlessX.transform(pca_less.transform(x)))
        predict.write(design + '\t' + (str)(y[0]) + '\n') 
        i += 1
predict.close()
