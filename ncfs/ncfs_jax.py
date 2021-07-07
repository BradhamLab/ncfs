# JAX for Github

#Things to add in: error messages, default values, and JIT FOR OBJ_


import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap, ops
from typing import Callable
import functools

import ncfs
from ncfs import NCFS

D = jnp.zeros(1)
diag_ind = 0
reg = 1
scorevar = 0


""" DISTANCE METRICS AND KERNEL TRANSFORMS """
def manhattan(x: jnp.array, y: jnp.array, w: jnp.array) -> float:
    return jnp.sum(w**2 * jnp.abs(x - y))
    
def sqeuclidean(x: jnp.array, y: jnp.array, w: jnp.array) -> float:
    return jnp.sum((w**2)*(x-y)**2)

def euclidean(x: jnp.array, y: jnp.array, w: jnp.array) -> float:
    return jnp.sqrt(jnp.sum((w**2)*(x-y)**2))

def variance(x: jnp.array, w: jnp.array) -> float:
    mean = jnp.sum(x*w)
    sum_of_weights = jnp.sum(w)
    sum_of_sq_weights = jnp.sum(w**2)
    mean = mean / sum_of_weights
    result = jnp.sum((w * (x - mean)**2))
    return result / (sum_of_weights - sum_of_sq_weights / sum_of_weights)

def phi_s(x: jnp.array, y: jnp.array, w: jnp.array) -> jnp.array:
    return zoe_try.variance(x - y, w) / zoe_try.variance(x+y, w)

def distmat(func: Callable, X: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(lambda x1: jax.vmap(lambda x2: func(x1, x2, w))(X))(X)

def exponential_transform(D: jnp.array, sigma_var: float) -> jnp.array:
    return jnp.exp(-1 * D/sigma_var)


metrics = {'manhattan': (manhattan),
           'cityblock': (manhattan),
           'taxicab': (manhattan),
           'l1': (manhattan),
           'l2': (euclidean),
           'euclidean': (euclidean),
           'sqeuclidean': (sqeuclidean)}


""" PROBABILITY MATRIX CALCULATION AND ASSOCIATED FXNS """
def p_ref_vec_func(x: jnp.array) -> float:
    return (x * 1.0) / jnp.sum(x)

def p_ref_vec(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.round(x + jnp.exp(-20), 8)
    diag_ind = jnp.diag_indices(x.shape[0])
    x = jax.ops.index_update(x, jax.ops.index[diag_ind[0], diag_ind[1]], 0)
    return jax.vmap(p_ref_vec_func, 0, 0)(x)

def probability_mat(x, coef, dist_metric, transform, sigma):
    D = distmat(dist_metric, x, coef) # 27us
    p_ref_prev = transform(D, sigma) # 38us
    return p_ref_vec(p_ref_prev)





class zoe_try:
    
    def __init__(self, alpha=0.1, sigma=1, reg=1, eta=0.001,
                 metric= 'cityblock', kernel='exponential',
                 max_iter=1000):

        self.alpha = alpha
        self.sigma = sigma
        self.reg = reg
        self.eta = eta 
        self.metric = metric
        self.kernel = exponential_transform
        self.max_iter = max_iter
        self.count = max_iter
        self.score_ = 0
        self.score_final = 0
        self.distance_fxn = metrics[self.metric]

    def check_and_set_params(self, x, y):
        self.n_features = x.shape[1]
        self.x = x

        self.sample_weights = NCFS.calculate_sample_weights(y)
        self.sample_weights = jnp.asarray(self.sample_weights)

        if len(y.shape) == 2:
            self.class_matrix = y
        elif len(y.shape) == 1:
            self.class_matrix = NCFS.calculate_class_matrix(y)
            self.class_matrix = jnp.asarray(self.class_matrix)

        self.coef = jnp.ones_like(x[0])
    
    def get_initial_coefficients(self, x):
        return jnp.ones_like(x.shape[1])

 
    """ PROBABILITY MATRIX CALCULATION AND ASSOCIATED FXNS """


    """ SCORE AND GRADIENT CALCULATIONS """
    def get_score(self, current_coef):
        # THE OLD OBJ SCORE
        current_coef = self.coef
        p_ref = probability_mat(self.x, current_coef, self.distance_fxn, self.kernel, self.sigma)
        return jnp.sum(jnp.sum((p_ref * self.class_matrix), axis = 1) * self.sample_weights) - (reg * jnp.sum(current_coef ** 2))
    
    new_obj_grad = grad(get_score, argnums = 1)
    # BUT COCO?
    
    def update_coef(self, feature_grad, loss):
        self.coef += feature_grad * self.alpha
        if loss > 0:
            self.alpha *= 1.01
        else:
            self.alpha *= 0.4

    
    """ MASTER FXNS FOR NCFS RUN"""
    def partial_fit(self, x, y, current_coef):
        x, y = self.check_and_set_params(x, y)
        self.score_ = self.partial_fit_noc(current_coef)
        return self
    
    def partial_fit_noc(self, current_coef) -> float:
        new_score = self.get_score(current_coef)
        loss = new_score - self.score_
        feature_grad = self.new_obj_grad(current_coef)
        self.update_coef(feature_grad, loss)
        return new_score

    def fit(self, x, y, current_coef):
        self.check_and_set_params(x, y)
        initial_score = self.get_score(current_coef)
        end_score = self.da_while_loop_aka_fitnoc(initial_score)
        return end_score

    def for_da_while_loop(self, old_score):
        current_coef = self.coef
        new_score = self.get_score(self, current_coef)
        loss = new_score - old_score
        self.score_ = new_score
        self.count = self.count - 1
        feature_grad = self.new_obj_grad(current_coef)
        self.update_coef(feature_grad, loss)
        return self.score_

    def da_while_loop_aka_fitnoc(self, old_score):
        loss = jnp.inf
        self.score_final = jax.lax.while_loop(((jnp.abs(loss) > self.eta) and (self.count > 0)), self.for_da_while_loop, old_score)
        return self.score_final
        
        

#metrics = {'manhattan': (manhattan),
           #'cityblock': (manhattan),
           #'taxicab': (manhattan),
           #'l1': (manhattan),
           #'l2': (euclidean),
           #'euclidean': (euclidean),
           #'sqeuclidean': (sqeuclidean)}



def mymain(x: jnp.array, y: jnp.array, func: Callable) -> float:
    if len(y.shape) == 2:
        class_matrix = y
    elif len(y.shape) == 1:
        class_matrix = NCFS.calculate_class_matrix(y)
        class_matrix = jnp.asarray(class_matrix)

    coef = jnp.ones(x.shape[1],)
    # need to make an allowance for warm starts

    distance_metric = func

    sample_weights = NCFS.calculate_sample_weights(y)
    sample_weights = jnp.asarray(sample_weights)

    #fit(x, class_matrix, sample_weights, coef, distance_metric, exponential_transform)


def main_notoy(x, y):
    f_select_z = zoe_try(alpha=0.01, sigma=1, reg=1, eta=0.001,
                    metric= 'manhattan',
                    kernel= 'exponential_transform')
    coco_main = f_select_z.get_initial_coefficients(x)
    from timeit import default_timer as timer
    times = jnp.zeros(10)
    for i in range(times.size):            
        start = timer()
        f_select_z.fit(x, y, coco_main)
        end = timer()
        jax.ops.index_update(times, jax.ops.index[i], (end - start))
        #times[i] = end - start
    print("Average execution time in seconds: {}".format(jnp.mean(times)))
    sorted_coef = jnp.argsort(-1 * f_select_z.coef)
    print(sorted_coef[:10])
    print(f_select_z.coef[0], f_select_z.coef[100], f_select_z.coef[sorted_coef[2]])
    print(f_select_z.score_(x, y))


def main_toy():
    x, y = NCFS.toy_dataset(n_features=1000, n1=150, n2=50)
    f_select_z = zoe_try(alpha=0.01, sigma=1, reg=1, eta=0.001,
                    metric='manhattan',
                    kernel='exponential')
    from timeit import default_timer as timer
    times = jnp.zeros(10)
    for i in range(times.size):            
        start = timer()
        f_select_z.fit(x, y)
        end = timer()
        jax.ops.index_update(times, jax.ops.index[i], (end - start))
        #times[i] = end - start
    print("Average execution time in seconds: {}".format(jnp.mean(times)))
    sorted_coef = jnp.argsort(-1 * f_select_z.coef)
    print(sorted_coef[:10])
    print(f_select_z.coef[0], f_select_z.coef[100], f_select_z.coef[sorted_coef[2]])
    print(f_select_z.score_(x, y))

 