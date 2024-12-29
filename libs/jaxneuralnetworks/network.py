#/!usr/bin/env  python
import numpy as np
import jax
import jax.numpy as jnp

class network:
  def __init__(self, topology):
    self.params = self.init_params(topology)

  # Handle configurations as a dictionary, in order to have recyclable training configurations

  def train(self,X,Y, configuration):
    lr      = configuration['lr']       # learning rate
    xi      = confuguration['xi']       # momentum coefficient
    epochs  = configuration['epochs']   # number of epochs

    mom     = [ 0 * p for p in self.params ]
    for i in range(epochs):
      gradient = self.batch_grad()


  #########################################################################################

  def _predict(self, params, x):                            # prediction function, with param
    W = params[0::2]                                        # as an explicit param
    B = params[1::2]                                        # this allows for jax.grad to perform
                                                            # automatic differentiation
    layer = x.copy()
    for (w,b) in zip(W[:-1],B[:-1]):
      layer = self.activation(w @ layer + b)

    w,b   = W[-1], B[-1]
    layer = w@layer + b
    return layer

  def predict(self, x):					    # user-friendly function which
    return self._predict(self.params, x)                    # offers an interface to internal
                                                            # private prediction method

  #########################################################################################

  def _batch_predict(self, params, X):
    return jax.vmap(lambda x: self._predict(params,x))(X)


  def batch_predict(self,X):
    return self._batch_predict(self.params, X)

  #########################################################################################

  def _loss(self, params, x , y):
    y_hat = self._predict(params, x)
    return (y_hat - y) @ (y_hat - y)

  def loss(self,x,y):
    return self.loss(self.params,x,y)

  #########################################################################################

  def _batch_loss(self,params,X,Y):
    return jax.vmap(lambda x,y: self._loss(params, x,y) )(X,Y).mean()

  def batch_loss(self,X,Y):
    return _batch_loss(self.params,X,Y)

  #########################################################################################

  def activation(self, x):
    return jnp.maximum(0,x)

  def init_params(self,topology):
    params = []
    for (t,t_next) in zip(topology[:-1], topology[1:]):
      W = np.random.randn(t_next, t)
      b = np.zeros(t_next)
      params.append(W)
      params.append(b)
    return params
