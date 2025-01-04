#!/usr/bin/env python

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

class NWCompression:
  def __init__(self, embedding = None, timeseries = None):
    self.embedding = embedding
    self.timeseries = timeseries
    self.batch_predict = lambda X_in,X,Y : jax.vmap(lambda x: self.predict(x,X,Y))(X_in)

  base_configuration = {
     'lr': 1e-3,
     'xi': 0.9,
     'steps' : 1000
  }

  def fit(self, Y, configuration,seed = 0, return_intermediates = False):
    # fit functions
    @jax.jit
    def loss(E,X):
      return (( self.batch_predict(E,E,X) -  X )**2).sum()
    grad = jax.jit(jax.grad(loss))

    # initialize the "big" arrays
    self.Y = Y.copy()
    np.random.seed(seed)
    E = np.random.randn(Y.shape[0],2) *  1e-10

    # training
    lr = configuration['lr'] # learning rate
    xi = configuration['xi'] # momentum coefficient
    steps = configuration['steps'] # number of iteratinos

    E_fin = E.copy()
    mom    = E * 0
    T   = np.geomspace(1e-6,1e-9,steps)
    # iteration scheme
    bar = tqdm(range(steps))

    intermediates = []

    for i in bar:
      bar.set_description("%.15f" % loss(E,Y))
      d =  - lr * grad(E,Y) 
      mom = mom * xi  + d
      old_E = E.copy()
      E =  E + mom  + T[i] * np.random.normal(size = E.shape)

      if return_intermediates:
        intermediates.append(E)

      if np.isnan(E).any():
        break

      if jnp.linalg.norm(E - old_E)/jnp.linalg.norm(old_E) < 0.001:
        lr *= 1.001 # if the training stalls increase  the learning rate

      if loss(E,Y) < loss(E_fin,Y): # update  the return value only if improving
        E_fin = E.copy()

    # save data
    self.E = E_fin.copy()
    if return_intermediates:
      return intermediates
    else:
      return
  # decoder functions
  def decode(self, e):
    return self.final_predict(e,self.E,self.Y)

  def k(self,x,y):
    return jnp.exp( - (x - y)@(x - y) )

  # prediction after training
  def final_predict(self,x,X,Y):
    # build weights
    K = jax.vmap(lambda x_i: self.k(x,x_i))(X)

    # build linear combination of time series based on weights
    C = K[:,None] * Y

    # return the prediction
    x_hat = C.sum(axis = 0) / K.sum()
    return x_hat


  # prediction during training
  def predict(self,x,X,Y):
    # build weights
    K = jax.vmap(lambda x_i: self.k(x,x_i)   * (x != x_i).any()  )(X)

    # build linear combination of time series based on weights
    C = K[:,None] * Y

    # return the prediction
    x_hat = C.sum(axis = 0) / K.sum()
    return x_hat

