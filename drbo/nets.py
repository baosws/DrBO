import gc
import numpy as np
from torch import nn
import torch

def SingleNodeMLP(nodes, hidden_sizes, batch_norm, dropout, activation, dtype, **unused):
    last_dim = nodes
    act = {
        'relu': lambda: nn.ReLU(inplace=True),
        'tanh': lambda: nn.Tanh(),
    }[activation]
    layers = []
    for h in hidden_sizes:
        layer = [nn.Linear(last_dim, h, dtype=dtype), nn.Dropout(p=dropout, inplace=True), act()]
        if batch_norm:
            layer.append(nn.BatchNorm1d(h))
        layers.extend(layer)
        last_dim = h
    layers.append(nn.Linear(last_dim, 1, dtype=dtype))
    return nn.Sequential(*layers)

class MultiSingleNodeMLP(nn.Module):
    def __init__(self, nodes, hidden_sizes, batch_norm, dropout, activation, dtype, **unused):
        super().__init__()
        self.mlps = nn.ModuleList((SingleNodeMLP(nodes=nodes, hidden_sizes=hidden_sizes, batch_norm=batch_norm, dropout=dropout, activation=activation, dtype=dtype, **unused) for i in range(nodes)))
    
    def forward(self, X):
        xs = torch.unbind(X, dim=-1)
        xs = [mlp(x) for mlp, x in zip(self.mlps, xs)]
        return torch.concat(xs, dim=-1)

class Dropout_Local_BIC:
    def __init__(self, max_size, dag_space, hidden_size, dropout, n_replay, GT, logger, scorer, device, n_grads, lr, sampling_chunksize=None, **unused):
        for k, v in locals().items():
            if k not in ['self', 'unused', '__class__'] and k not in self.__dict__:
                setattr(self, k, v)
        self.logs = []
        self.nodes = self.dag_space.nodes
        self.dtype = torch.float32
        self.dags = torch.empty((max_size, dag_space.nodes, dag_space.nodes), dtype=torch.uint8, pin_memory=True)
        self.local_scores = torch.empty((max_size, dag_space.nodes), dtype=self.dtype, pin_memory=True)
        self.idx = 0
        self.model = MultiSingleNodeMLP(nodes=self.nodes, dtype=self.dtype, hidden_sizes=[hidden_size], batch_norm=True, dropout=dropout, activation='relu').to(device=self.device, dtype=self.dtype)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def sample(self, zs, dags, num_samples, **kwargs):
        est_scores = []
        chunksize = len(dags) if self.sampling_chunksize is None else self.sampling_chunksize
        for i in range(0, len(dags), chunksize):
            dags_torch = torch.as_tensor(dags[i: i + chunksize], dtype=self.dtype, device=self.device)
            with torch.no_grad():
                est_scores_ = np.stack([self.model(dags_torch).cpu().numpy() for i in range(num_samples)], axis=1)
            del dags_torch
            est_scores.append(est_scores_)
            torch.cuda.empty_cache()
            gc.collect()

        est_scores = np.concatenate(est_scores)

        est_local_scores = np.clip(est_scores, -12, 6)
        est_scores = self.scorer.aggregate_batch(dags, est_local_scores)

        return est_scores
    
    def prepare_data(self, dags, scores, new_dags, new_scores):
        old_size = self.idx - len(new_dags)
        if old_size == 0:
            return new_dags.to(self.device), new_scores.to(self.device)
        buffer = torch.multinomial(torch.as_tensor([1 / old_size] * old_size), num_samples=min(old_size, self.n_replay))
        buffer = torch.concat((buffer, torch.arange(old_size, self.idx)))
        X, y = dags[buffer].to(self.device, dtype=self.dtype), scores[buffer].to(self.device)
        return X, y
    
    def train(self, zs, dags, scores, **unused):
        scores = scores[:, :self.nodes]
        dags, scores = map(lambda x: torch.as_tensor(x, dtype=self.dtype), (dags, scores))
        self.dags[self.idx: self.idx + len(dags)] = dags
        self.local_scores[self.idx: self.idx + len(dags)] = scores
        self.idx += len(dags)
            
        for step in range(self.n_grads):
            X, y = self.prepare_data(self.dags, self.local_scores, dags, scores)
            pred = self.model(X)
            loss = torch.nn.functional.mse_loss(pred, y)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .5)
            self.opt.step()
            loss = loss.item()
            del X, y
            torch.cuda.empty_cache()
            gc.collect()