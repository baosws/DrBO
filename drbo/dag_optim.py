
import numpy as np
from torch.quasirandom import SobolEngine
from drbo.nets import Dropout_Local_BIC

class BaseDagOptim:
    def __init__(self, dag_space, X, GT, logger, pruner, scorer):
        self.dag_space = dag_space
        self.X = X
        self.GT = GT
        self.logger = logger
        self.pruner = (lambda X, dag: dag) if pruner is None else pruner
        self.scorer = scorer
        self.nodes = self.dag_space.nodes
        self.best_dag = self.best_score = None
        self.cnt = 0
        self.n_steps = 0
        self.unique_dags = set()

    def add_data(self, zs, dags, scores, **kwargs):
        if self.best_score is None or self.best_score < scores[:, -1].max():
            self.best_score = scores[:, -1].max()
            best_idx = np.argmax(scores[:, -1]).item()
            best_dag, best_score = dags[best_idx], scores[best_idx, -1]
            self.best_dag = best_dag
            self.best_idx = best_idx + self.cnt
            if self.logger.verbose:
                from castle.metrics import MetricsDAG
                pruned = self.pruner(self.X, best_dag)
                self.best_metrics = MetricsDAG._count_accuracy(pruned, self.GT)

        self.cnt += len(zs)
        self.n_steps += 1
        if self.best_score is not None:
            if self.logger.verbose:
                self.logger(f'Best={self.best_score:.6f}@{self.best_idx}')
                metrics = self.best_metrics
                self.logger(f'shd={int(metrics["shd"])}')
                self.logger.add(BIC=self.best_score, **metrics)
        self.update(zs, dags, scores, **kwargs)


# from TurBO (Eriksson et al., 2019): https://github.com/uber-research/TuRBO/blob/de0db39f481d9505bb3610b7b7aa0ebf7702e4a5/turbo/utils.py#L15
def to_unit_cube(x, lb, ub):
    xx = (x - lb) / (ub - lb)
    return xx

# from TurBO (Eriksson et al., 2019): https://github.com/uber-research/TuRBO/blob/de0db39f481d9505bb3610b7b7aa0ebf7702e4a5/turbo/utils.py#L22
def from_unit_cube(x, lb, ub):
    xx = x * (ub - lb) + lb
    return xx

# from TurBO (Eriksson et al., 2019): https://github.com/uber-research/TuRBO/blob/de0db39f481d9505bb3610b7b7aa0ebf7702e4a5/turbo/utils.py#L29
def latin_hypercube(n_pts, dim):
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X

class DagOptimBO(BaseDagOptim):
    def __init__(self, dag_space, X, GT, lr, n_cands, n_grads, n_replay, dropout, hidden_size, device, max_size, logger, pruner, scorer):
        super().__init__(dag_space, X, GT, logger, pruner, scorer)
        for k, v in locals().items():
            if k not in ['self', 'unused', '__class__'] and k not in self.__dict__:
                setattr(self, k, v)

        self.bic_model = Dropout_Local_BIC(
            dag_space=dag_space,
            scorer=scorer,
            max_size=max_size,
            lr=lr,
            n_replay=n_replay,
            dropout=dropout,
            hidden_size=hidden_size,
            GT=GT,
            logger=logger,
            n_grads=n_grads,
            device=device
        )
        self.zs = self.bics = None
        self.length = 1
        self.succcount = self.failcount = 0
        self.suctol, self.failtol = 3, 5
        self.min_length, self.max_length = .01, 2

    def get_center(self):
        return self.zs[self.best_idx]

    def update(self, zs, dags, scores):
        self.bic_model.train(zs, dags, scores)
        self.zs = zs if self.zs is None else np.concatenate((self.zs, zs))

    def add_data(self, zs, dags, scores, **kwargs):
        # based on TurBO (Eriksson et al., 2019): https://github.com/uber-research/TuRBO/blob/de0db39f481d9505bb3610b7b7aa0ebf7702e4a5/turbo/turbo_1.py#L137
        if self.best_score is not None:
            if self.best_score < scores[:, -1].max():
                self.succcount += 1
                self.failcount = 0
            else:
                self.succcount = 0
                self.failcount += 1
        
        if self.succcount >= self.suctol:
            self.length = min(2 * self.length, self.max_length)
            self.succcount = 0
        elif self.failcount >= self.failtol:
            self.length = self.length / 2
            self.failcount = 0
        
        if self.length < self.min_length:
            self.length = self.max_length

        return super().add_data(zs, dags, scores, **kwargs)

    def suggest(self, batch_size, **unused):
        if self.best_dag is None:
            zs = latin_hypercube(batch_size, self.dag_space.dim)
            zs = from_unit_cube(zs, -1, 1)
            dag_cands = self.dag_space.vec2dag(zs)
            return zs, dag_cands, np.zeros(batch_size)
        else:
            # based on TurBO (Eriksson et al., 2019): https://github.com/uber-research/TuRBO/blob/de0db39f481d9505bb3610b7b7aa0ebf7702e4a5/turbo/turbo_1.py#L181
            x_center_unit = to_unit_cube(self.get_center(), -1, 1)  # dim
            weights = 1
            length = self.length
            lb = np.clip(x_center_unit - weights * length / 2.0, 0.0, 1.0)
            ub = np.clip(x_center_unit + weights * length / 2.0, 0.0, 1.0)

            seed = np.random.randint(int(1e6))
            sobol = SobolEngine(self.dag_space.dim, scramble=True, seed=seed)
            pert = sobol.draw(self.n_cands).numpy()
            pert = lb + (ub - lb) * pert

            prob_perturb = min(20.0 / self.dag_space.dim, 1.0)
            mask = np.random.rand(self.n_cands, self.dag_space.dim) <= prob_perturb
            ind = np.where(np.sum(mask, axis=1) == 0)[0]
            mask[ind, np.random.randint(0, self.dag_space.dim - 1, size=len(ind))] = 1

            X_cand_unit = x_center_unit.copy() * np.ones((self.n_cands, self.dag_space.dim))  # n_cand x dim
            X_cand_unit[mask] = pert[mask]
            zs = from_unit_cube(X_cand_unit, -1, 1)

            dag_cands = self.dag_space.vec2dag(zs)

            est_scores = self.bic_model.sample(zs=zs, dags=dag_cands, num_samples=1)
            assert est_scores.shape == (self.n_cands, 1), f'{est_scores.shape=}'

            est_scores = est_scores.max(axis=-1)
            topk = np.argpartition(-est_scores, kth=batch_size)[:batch_size]

            return zs[topk], dag_cands[topk], est_scores[topk]