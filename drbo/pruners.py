from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from causallearn.utils.cit import KCI

# from gCastle (Zhang et al., 2021): https://github.com/huawei-noah/trustworthyAI/blob/f2eaa6c0e1c176b31bb48049df38d95f65bf210d/gcastle/castle/algorithms/gradient/corl/torch/utils/graph_analysis.py#L77
def prune_linear(X, dag):
    d = len(dag)
    reg = LinearRegression()
    W = []

    for i in range(d):
        col = dag[:, i] == 1
        if np.sum(col) <= 0.1:
            W.append(np.zeros(d))
            continue

        X_train = X[:, col]

        y = X[:, i]
        reg.fit(X_train, y)
        reg_coeff = reg.coef_

        cj = 0
        new_reg_coeff = np.zeros(d, )
        for ci in range(d):
            if col[ci]:
                new_reg_coeff[ci] = reg_coeff[cj]
                cj += 1

        W.append(new_reg_coeff)

    W = np.asarray(W)
    A = np.float32(np.abs(W) > .3).T
    return A

def prune_cit(X, dag):
    ret = dag.copy()
    n, d = X.shape
    cit = KCI(data=X)
    for i in range(d):
        parents, = np.nonzero(dag[:, i])
        for j in parents:
            if cit(i, j, [k for k in parents if k != j]) >= 0.001:
                ret[j, i] = 0

    return ret