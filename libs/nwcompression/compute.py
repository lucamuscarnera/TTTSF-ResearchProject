#!/usr/bin/env python

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

class NWCompression:
  def __init__(self, embedding = None, timeseries = None):
    self.E = embedding
    self.Y = timeseries
    self.batch_predict = lambda X_in,X,Y : jax.vmap(lambda x: self.predict(x,X,Y))(X_in)

  base_configuration = {
     'lr': 1e-3,
     'xi': 0.9,
     'steps' : 1000,
     'dimension' : 2
  }

  def fit(self, Y, configuration,seed = 0, return_intermediates = False):
    # fit functions
    @jax.jit
    def loss(E,X):
      return (( self.batch_predict(E,E,X) -  X )**2).sum()
    grad = jax.jit(jax.grad(loss))


    # training
    lr = configuration['lr'] # learning rate
    xi = configuration['xi'] # momentum coefficient
    steps = configuration['steps'] # number of iteratinos
    dimension = configuration['dimension'] # embedding size

    # initialize the "big" arrays
    self.Y = Y.copy()
    np.random.seed(seed)
    E = np.random.randn(Y.shape[0],dimension) *  1e-10

    E_fin = E.copy()
    mom    = E * 0
    T   = np.geomspace(1e-6,1e-9,steps)
    # iteration scheme
    bar = tqdm(range(steps))

    intermediates = []

    for i in bar:
      bar.set_description("%.15f" % loss(E_fin,Y))
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

  def batch_decode(self,E):
    return jax.vmap(self.decode)(E)

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


  # backward construction; given a timeseries recovers the embedding

  def backward_predict(self, y, e_0):
    def loss(e,y,E,Y):
      return jnp.sum((self.predict(e, E, Y) - y)**2)
    grad = jax.grad(loss)

    bar = tqdm(range(1000))
    e   = e_0.copy()
    mom = e * 0
    lr  = 1e-3
    for i in bar:
      g       = grad(e,y,self.E,self.Y)
      new_e   = e - lr * g 
      if loss(new_e,y, self.E, self.Y) < loss(e,y, self.E, self.Y):
        e = new_e 
        lr *= 1.1
      else: 
        lr *= 0.8
      bar.set_description("%.12f" % loss(e,y, self.E, self.Y))

      if lr < 1e-15:
        print("Coverged earlier")
        break
    return e
