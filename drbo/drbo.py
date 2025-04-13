from tqdm import trange
from sklearn.metrics import mean_squared_error, r2_score
import random
import numpy as np
import torch
from drbo.dag_optim import DagOptimBO
from drbo.dag_spaces import DAGLowRank
from drbo.scorers import Scorer
from drbo.logger import Logger
from sklearn.preprocessing import StandardScaler
import torch

def DrBO(
        X: np.ndarray,
        score_method: str,
        score_params: dict,
        max_evals: int,
        batch_size: int = 64,
        normalize: bool = False,
        dag_rank: int = 8,
        pruner=None,
        n_cands: int = 100000,
        n_grads: int = 10,
        lr: float = 0.1,
        hidden_size: int = 64,
        dropout: float = 0.1,
        n_replay: int = 1024,
        device: str | torch.device = 'cpu', 
        verbose: bool = False,
        random_state: int | None = 0,
        GT: np.ndarray | None = None,
        **kwargs
) -> dict:
    '''DAG recovery via Bayesian Optimization.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_nodes)
        Observational data that you want to learn.
    score_method : str or callable
        DAG scoring method. If str, "BIC" and "BGe" are implemented. If callable, define it the same way as `scorers.BIC`.
    score_params : dict
        Scoring parameters passed to the initialization of the scorer. For example, BIC's parameters include regression method, noise variant, etc. (see `scorers.BIC`).
    max_evals : int
        Maximum number of DAG score evaluations.
    batch_size : int, default=64
        Number of DAGs evaluated at each BO iteration. The final iteration may evaluate fewer DAGs if `max_evals` is not divisible by `batch_size`.
    normalize : bool, default=False
        Whether `X` is normalized to zero mean and unit variance (per dimension) in the beginning.
    dag_rank : int, default=8
        Rank of the DAG representation.
    pruner : callable, default=None
        Pruning strategy. Should take as input `X` and the ground dag `GT` (ndarray) and return the pruned DAG. `None` means no pruning.
    n_cands : int, default=100000
        Number of preliminary candidates generated per BO iteration.
    n_grads : int, default=10
        Number of gradient steps to train the neural nets per BO iteration.
    lr : float, default=0.1
        Learning rate for Adam optimizer to update the neural nets per BO iteration.
    hidden_size : int, default=64
        Number of hidden units in each neural net.
    dropout : float, default=0.1
        Dropout rate for the neural nets.
    n_replay : int, default=1024
        Number of past evaluations to train in each BO iteration.
    device : str or torch.device, default='cpu'
        Device to use for training neural nets.
    verbose : bool, default=False
        Whether to print progress bar or not.
    random_state : None or int
        Random seed.
    
    Returns
    --------
    pred : dict
        "raw": the raw (unpruned) DAG found by DrBO.
        "history": learning history, including timestamp and performance metrics at each step.
    '''
    np.random.seed(random_state)
    random.seed(random_state)
    torch.random.fork_rng()
    torch.random.manual_seed(random_state)
    torch.backends.cuda.matmul.allow_tf32 = True

    if normalize:
        X = StandardScaler().fit_transform(X)

    n, d = X.shape
    scorer = Scorer(score_method=score_method, data=X, score_params=score_params)
    dag_space = DAGLowRank(nodes=d, rank=dag_rank)

    gt_score = None
    if GT is not None:
        gt_score = scorer(GT)[-1]

    logger = Logger(verbose=verbose)
    max_size = (max_evals + batch_size - 1) // batch_size * batch_size
    model = DagOptimBO(
        X=X,
        GT=GT,
        dag_space=dag_space,
        max_size=max_size,
        lr=lr,
        n_grads=n_grads,
        n_cands=n_cands,
        dropout=dropout,
        hidden_size=hidden_size,
        n_replay=n_replay,
        device=device,
        logger=logger,
        pruner=pruner,
        scorer=scorer
    )
    with trange(max_evals, disable=not verbose) as pbar:
        pbar.refresh()
        n_evals = 0

        while n_evals < max_evals:
            logger.reset()
            next_zs, next_dags, est_scores = model.suggest(batch_size)
            true_scores = scorer.batch_eval(next_dags)
            rmse = mean_squared_error(true_scores[:, -1], est_scores) ** .5
            r2 = r2_score(true_scores[:, -1], est_scores)
            logger(f'{rmse=:.3f}', f'{r2=:.3f}')

            logger.add(test_rmse=rmse, test_r2=r2)

            if gt_score is not None:
                logger(f'GT={gt_score:.6f}')
            n_evals += batch_size
            logger.set_step(n_evals)
            model.add_data(next_zs, next_dags, true_scores)

            pbar.set_postfix_str(logger.log_str())
            pbar.update(batch_size)
            pbar.refresh()

    return dict(raw=model.best_dag, history=logger.track_logs)