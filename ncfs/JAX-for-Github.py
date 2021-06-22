# JAX for Github

# Nicer Distances

import jax.numpy as jnp
from jax import grad, jit, vmap

x1 = jnp.array([[3., 7., 2., 1.], [9., 2., 1., 3.], [5., 6., 2., 6.]])
y1 = jnp.array([[6., 14., 4., 2.], [18., 4., 2., 6.], [10., 12., 4., 12.]])
w1 = jnp.ones(4)

#Manhattan
def manhattan(x: jnp.array, y: jnp.array, w: jnp.array) -> float:
    return jnp.sum(w*jnp.abs(x-y))

print(grad(manhattan, argnums=2)(x1, y1, w1))

from jax import jacfwd, jacrev
f = lambda w: manhattan(x1, y1, w1)
J = jacfwd(f)(w1)
Jr = jacrev(f)(w1)
print(J)
print(Jr)

#SqEuclidean
def sqeuclidean(x: jnp.array, y: jnp.array, w: jnp.array) -> float:
    return jnp.sum(w * (x-y) **2)
print(sqeuclidean(x1, y1, w1))

#Euclidean
def euclidean(x: jnp.array, y: jnp.array, w: jnp.array) -> float:
    return jnp.sqrt(sqeuclidean(x, y, w))
print(euclidean(x1, y1, w1))

print(grad(euclidean, argnums=2)(x1, y1, w1))
# %%
### Distance Matrix Calculation

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from jax import ops
from typing import Callable
import functools

x3 = jnp.array([[3, 7, 2, 1], [9, 2, 1, 3], [5, 6, 2, 6]])
y2 = jnp.array([[6, 14, 4, 2], [18, 4, 2, 6], [10, 12, 4, 12]])

w2 = jnp.ones(4)
w2 = w2.reshape(4,)

def manhattan(x: jnp.array, y: jnp.array, w: jnp.array) -> float:
    return jnp.sum(w**2 * jnp.abs(x - y))

man_comp = jit(manhattan)

@functools.partial(jax.jit, static_argnums=(0))
def distmat(func: Callable, X: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(lambda x1: jax.vmap(lambda x2: func(x1, x2, w))(X))(X)

def sqeuclidean(x: jnp.array, y: jnp.array, w: jnp.array) -> float:
    return (w**2) * jnp.sum((x-y) **2)

def euclidean(x: jnp.array, y: jnp.array, w: jnp.array) -> float:
   return jnp.sqrt(jnp.sum((w**2)*(x-y)**2))


def p_ref_vec_func(x: jnp.array) -> float:
    return (x * 1.0) / jnp.sum(x)

def p_ref_vec(x: jnp.ndarray) -> jnp.ndarray:
    m = []
    m = jnp.where(jnp.all((x == 0), axis = 1))
    m = jnp.asarray(m)
    x = jax.ops.index_update(x, jax.ops.index[m, :], jnp.exp(-20))
    return jax.vmap(lambda x1: p_ref_vec_func(x1))(x)

def exponential_transform(D: np.array, sigma: float) -> np.array:
    return jnp.exp(-1 * D/sigma)

D = jnp.zeros(1)
diag_ind = 0

## Probability Matrix
def probability_mat(x, w, D, dist_metric, transform, sigma):
    D = distmat(dist_metric, x, w) # does D need to be in inputs?
    p_ref_prev = transform(D, sigma) ## correct syntax?
    diag_ind = jnp.diag_indices(D.shape[0])
    diag_ind = jnp.asarray(diag_ind)
    p_ref_prev = jax.ops.index_update(p_ref_prev, jax.ops.index[diag_ind[0, :], diag_ind[1, :]], 0)
    p_ref = p_ref_vec(p_ref_prev)
    return print(p_ref)



# %%
import numpy as np
import jax.numpy as jnp


z = jnp.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])

diag_ind = jnp.diag_indices(z.shape[0])
diag_ind = jnp.asarray(diag_ind)
#z[(diag_ind[0, :]), (diag_ind[1, :])] = 0
#z[diag_ind[0, :], diag_ind[1, :]] = 0.0
z = jax.ops.index_update(z, jax.ops.index[diag_ind[0, :], diag_ind[1, :]], jnp.exp(-20))
#print(diag_ind[0, :])
print(z)
# %%
