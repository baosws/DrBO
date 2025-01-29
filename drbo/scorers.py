import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import euclidean_distances
from sklearn.pipeline import FunctionTransformer, make_pipeline

def median_width(data):
    K = euclidean_distances(data)
    K = K[K > 0]
    return np.median(K)

def BIC(X, noise_var, med_bw=False, reg=None, **reg_params):
    '''Bayesian Information Criterion (BIC) for scoring DAGs

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_nodes)
        Observational data that you want to learn.
    noise_var : str
        'ev' for equal noise variance and 'nv' for non-equal noise variance.
    med_bw : bool, default=False
        Whether to divide data by the median distance before applying GP regression. Used in RL-BIC (Zhu et al., 2020), CORL (Wang et al., 2021), & RCL-OG (Yang et al., 2023).
    reg : str
        Regression method. 'linear' for linear regression and 'gp' for Gaussian process regression.
    reg_params : dict, optional
        Additional params to be passed to the regressor, e.g., regularization `alpha`.
    '''
    reg_params.setdefault('alpha', 1.e-8)
    _scale = lambda data: (median_width(data) if med_bw else 1)
    if reg == 'linear':
        XtX = X.T @ X
    n, d = X.shape
    dag_pen = np.log(n) / n / d

    def local_mse(node, parents):
        x = X[:, parents]
        y = X[:, node]
        if parents:
            if reg == 'linear':
                xtx = XtX[np.ix_(parents, parents)]
                xty = XtX[parents, node]
                theta = np.linalg.solve(xtx, xty)
                y_pred = x.dot(theta)
            elif reg == 'gp':
                model = make_pipeline(FunctionTransformer(lambda x: x / _scale(x)), GaussianProcessRegressor(kernel=RBF(length_scale_bounds=(1, 1e5)), **reg_params, random_state=0))
                model.fit(x, y)
                y_pred = model.predict(x)
        else:
            y_pred = np.mean(y)
        mse = np.mean(np.square(y - y_pred))

        mse = mse + ('gp' in reg) / len(y) # from gCastle (Zhang et al., 2021): https://github.com/huawei-noah/trustworthyAI/blob/c90310b2e36b7ee0e4434162426427aa78f78dd5/gcastle/castle/algorithms/gradient/rl/torch/rewards/Reward_BIC.py#L135

        return np.log(mse)

    def aggregate(dag, ln_mse):
        edges = np.sum(dag)
        if noise_var == 'ev':
            mse = np.mean(np.exp(np.asarray(list(ln_mse.values()))))
            total_score = -(edges * dag_pen + np.log(mse))
        elif noise_var == 'nv':
            total_score = -(edges * dag_pen + np.mean(np.asarray(list(ln_mse.values()))))

        scores = [0] * d
        for (node, _), val in ln_mse.items():
            scores[node] = val
        return scores + [total_score]
    
    def aggregate_batch(dags, ln_mse):
        edges = np.sum(dags, axis=(-1, -2))[..., None] # Bx1
        if noise_var == 'ev':
            est_scores = -(edges * dag_pen + np.log(np.mean(np.exp(ln_mse), axis=-1))) # Bxs
        elif noise_var == 'nv':
            est_scores = -(edges * dag_pen + np.mean(ln_mse, axis=-1))
        
        return est_scores
    
    return local_mse, aggregate, aggregate_batch

class Scorer:
    def __init__(self, score_method, data, score_params):
        score_method = eval(score_method) if isinstance(score_method, str) else score_method
        self.local_score, self.aggregate, self.aggregate_batch = score_method(X=data, **score_params)
        self.cache = dict()
        self.data = data
        self.samples, self.nodes = data.shape

    def __call__(self, dag):
        scores = {}
        for node in range(self.nodes):
            parents = tuple(np.nonzero(dag[:, node])[0])
            key = (node, parents)
            if (val := self.cache.get(key, None)) is None:
                self.cache[key] = val = self.local_score(node, parents)
            scores[key] = val

        score = self.aggregate(dag, scores)
        return score
    
    def batch_eval(self, dags):
        ret = np.asarray([self(dag) for dag in dags])
        return ret