from functools import partial
import numpy as np
import jax
from opt_einsum import contract

class DAGLowRank:
    def __init__(self, nodes, rank, **unused):
        self.nodes = nodes
        self.rank = rank
        self.dim = self.nodes * (1 + rank)
    
    @partial(jax.jit, static_argnums=(0,))
    def __vec2dag(self, z):
        c = z[..., :self.nodes]
        s = z[..., self.nodes:].reshape(*z.shape[-2::-1], self.nodes, self.rank)
        D = contract('...ik,...jk->...ij', s, s)
        C = 1 * (c[..., None] > c[..., None, :])
        E = D > 0
        A = C * E
        return A

    def vec2dag(self, z):
        with jax.default_device(jax.devices('cpu')[0]):
            A = self.__vec2dag(z)
        return np.asarray(A)