# Nicer Distances

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

D = jnp.zeros(1)
diag_ind = 0
sample_weights = jnp.array([1, 1, 1])
coef = jnp.array([[1., 1., 1., 1.]])

p_ref = jnp.zeros(1)
reg = 1

from ncfs import NCFS
y = np.array([0, 0, 1, 1]) # class vector
class_matrix = NCFS.calculate_class_matrix(y)
class_matrix = jnp.asarray(class_matrix)

@jax.jit
def manhattan(x: jnp.array, y: jnp.array, w: jnp.array) -> float:
    return jnp.sum(w**2 * jnp.abs(x - y))
#man_comp = jit(manhattan)

def dist_grad(func: Callable, x: jnp.array, w: jnp.array) -> float:
    return jax.vmap(lambda x1: jax.vmap(lambda x2: func(x1, x2, w))(x))(x)

manhattan_grad = grad(manhattan, argnums=2)(x3, x3, coef)

@functools.partial(jax.jit, static_argnums=(0))
def distmat(func: Callable, X: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(lambda x1: jax.vmap(lambda x2: func(x1, x2, w))(X))(X)

@jax.jit
def sqeuclidean(x: jnp.array, y: jnp.array, w: jnp.array) -> float:
    return (w**2) * jnp.sum((x-y) **2)
#sqeuclidean = jit(sqeuclidean)

@jax.jit
def euclidean(x: jnp.array, y: jnp.array, w: jnp.array) -> float:
   return jnp.sqrt(jnp.sum((w**2)*(x-y)**2))
#euclidean = jit(euclidean)

@jax.jit
def p_ref_vec_func(x: jnp.array) -> float:
    return (x * 1.0) / jnp.sum(x)
#p_ref_vec_func = jit(p_ref_vec_func)

#@functools.partial(jax.jit, static_argnums=(0))
def p_ref_vec(x: jnp.ndarray) -> jnp.ndarray:
    m = []
    m = jnp.where(jnp.all((x == 0), axis = 1))
    m = jnp.asarray(m)
    x = jax.ops.index_update(x, jax.ops.index[m, :], 1)
    return jax.vmap(lambda x1: p_ref_vec_func(x1))(x)
p_ref_vec = functools.partial(p_ref_vec)

@jax.jit
def exponential_transform(D: jnp.array, sigma: float) -> jnp.array:
    return jnp.exp(-1 * D/sigma)
#exponential_transform = jit(exponential_transform)


## Probability Matrix

#@functools.partial(jax.jit, static_argnums=(0))
def probability_mat(x, w, D, dist_metric, transform, sigma):
    D = distmat(dist_metric, x, w) # does D need to be in inputs?
    p_ref_prev = transform(D, sigma) ## correct syntax?
    diag_ind = jnp.diag_indices(D.shape[0])
    diag_ind = jnp.asarray(diag_ind)
    p_ref_prev = jax.ops.index_update(p_ref_prev, jax.ops.index[diag_ind[0, :], diag_ind[1, :]], 0)
    p_ref = p_ref_vec(p_ref_prev)
    #return print(p_ref)
    return p_ref
probability_mat = functools.partial(probability_mat)

## Objective Function

def objective(p_ref, class_matrix, sample_weights, coef, reg):
    #score = jnp.sum(jnp.sum((p_ref * class_matrix), axis = 1) * sample_weights)
    #reg_term = jnp.sum(coef ** 2)
    # p_correct = jnp.sum(p_ref * class_matrix, axis =1)


    return jnp.sum(jnp.sum((p_ref * class_matrix), axis = 1) * sample_weights) - (reg * jnp.sum(coef ** 2))


def score(x, class_matrix, sample_weights, coefs, distance_matrix, distance, transform, sigma, reg):
    p_ref = probability_mat(x, coefs, distance_matrix, distance, transform, sigma)
    return objective(p_ref, class_matrix, sample_weights, coefs, reg)

objective_grad = grad(objective, argnums=3)
# %%

import numpy as np
import jax.numpy as jnp
from jax import grad, vmap, jit


def objective(p_ref, class_matrix, sample_weights, coef, reg):
    #score = jnp.sum(jnp.sum((p_ref * class_matrix), axis = 1) * sample_weights)
    #reg_term = jnp.sum(coef ** 2)
    # p_correct = jnp.sum(p_ref * class_matrix, axis =1)


    return jnp.sum(jnp.sum((p_ref * class_matrix), axis = 1) * sample_weights) - (reg * jnp.sum(coef ** 2))


def score(x, class_matrix, sample_weights, coefs, distance_matrix, distance, transform, sigma, reg):
    p_ref = probability_mat(x, coefs, distance_matrix, distance, transform, sigma)
    return objective(p_ref, class_matrix, sample_weights, coefs, reg)

objective_grad = grad(objective, argnums=3)

def partial_fit(x, class_matrix, distance_metric, sample_weights, coef, transform, sigma):
    D = distmat(distance_metric, x, coef)
    p_ref = probability_mat(x, coef, D, distance_metric, transform, sigma)
    


# %%

def partial_fit(X, class_matrix, sample_weights, coefs,
                distance, distance_grad, transform,
                sigma, reg, alpha,
                distance_matrix, partials_vec,
                score):
    # Matrix P, where P_ij is the probability of j referencing i
    p_reference = probability_matrix(X, coefs, distance_matrix, distance,
                                     transform, sigma)
    # a sample length vector -- probability of correctly assigning sample i
    p_correct = (p_reference * class_matrix).sum(axis=1)
    for l in numba.prange(X.shape[1]):
        # (sample x sample) matrix to store gradients where element
        # (i, j) is the gradient for w_l between samples i and 
        partials_vec[l] = feature_gradient(X, class_matrix, coefs, sample_weights,
                                           p_reference, p_correct, l,
                                           distance_grad,
                                           sigma, reg)
    new_score = objective(p_reference, class_matrix, sample_weights, coefs, reg)
    loss = new_score - score
    ncfs_update_weights(coefs, partials_vec, alpha, loss)
    return new_score




@numba.njit(parallel=True, fastmath=True)
def objective(p_reference, class_matrix, sample_weights, coef, reg):
    score = 0
    for i in numba.prange(p_reference.shape[0]):
        row_score = 0
        for j in numba.prange(p_reference.shape[1]):
            row_score += p_reference[i, j] * class_matrix[i, j]
        row_score *= sample_weights[i]
        score += row_score
    reg_term = 0
    for i in numba.prange(coef.size):
        reg_term += coef[i] ** 2
    return score - reg * reg_term

# %%


def variance(x: jnp.array, w: jnp.array) -> float:
    mean = jnp.sum((x * w), axis = 0)
    sum_of_weights = jnp.sum(w)
    sum_of_sq_weights = jnp.sum(w**2)

    mean = mean / sum_of_weights
    result = jnp.sum((w * (x - mean))**2)
    return result / (sum_of_weights - sum_of_sq_weights / sum_of_weights)

def phi_s(x: jnp.array, y: jnp.array, w: jnp.array) -> float:
    if jnp.all(x==y):
        return 0.0
    nume = variance(x - y, w)
    denom = variance(x+y, w)
    if denom == 0:
        return np.inf
    return nume/denom



# %%
