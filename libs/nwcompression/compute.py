#!/usr/bin/env python

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

class NWCompression:
  def __init__(self):
    self.batch_predict = lambda X_in,X,Y : jax.vmap(lambda x: self.predict(x,X,Y))(X_in)

  # fit functions
  @jax.jit
  def _loss(E,X):
    return (( self.batch_predict(E,E,X) -  X )**2).sum()

  def fit(self, Y, configuration):
    # initialize the "big" arrays
    self.X = X.copy()
    E = np.random.randn(Y.shape[0],2) *  1e-10

    # training
    lr = configuration['lr'] # learning rate
    xi = configuration['xi'] # momentum coefficient

    E_fin = E.copy()
    # iteration scheme
    for i in bar:
      bar.set_description("%.15f" % loss(E,Y))
      d =  - lr * grad(E,Y) 
      mom = mom * 0.99  + d
      old_E = E.copy()
      E =  E + mom + T[i] * np.random.normal(size = E.shape)


      if np.isnan(E).any():
        break

      if jnp.linalg.norm(E - old_E)/jnp.linalg.norm(old_E) < 0.001:
        lr *= 1.001 # if the training stalls increase  the learning rate

      if loss(E,Y) < loss(E_fin,Y):
        E_fin = E.copy()

    # save data
    self.E = E_fin.copy()
    return

  # decoder functions
  def decode(self, e):
    return self.final_predict(e,self.E,self.Y)

  def k(x,y):
    return jnp.exp( - (x - y)@(x - y) )

  # prediction after training
  def final_predict(x,X,Y):
    # build weights
    K = jax.vmap(lambda x_i: k(x,x_i))(X)

    # build linear combination of time series based on weights
    C = K[:,None] * Y

    # return the prediction
    x_hat = C.sum(axis = 0) / K.sum()
    return x_hat


  # prediction during training
  def predict(x,X,Y):
    # build weights
    K = jax.vmap(lambda x_i: k(x,x_i)   * (x != x_i).any()  )(X)

    # build linear combination of time series based on weights
    C = K[:,None] * Y

    # return the prediction
    x_hat = C.sum(axis = 0) / K.sum()
    return x_hat

