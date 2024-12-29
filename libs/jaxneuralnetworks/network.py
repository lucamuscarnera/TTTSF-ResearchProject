#/!usr/bin/env  python
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

class network:
  def __init__(self, topology):
    self.topology = [1 * t for t in topology]
    self.params = self.init_params(topology)

    # save the jitted version of the gradient for training
    self._batch_grad = jax.jit(jax.grad(self._batch_loss))

  # Handle configurations as a dictionary, in order to have recyclable training configurations
  base_configuration = {
     'lr': 1e-1,
     'xi': 0.9,
     'epochs' : 100
  }
  def train(self,X,Y, configuration):
    lr      = configuration['lr']       # learning rate
    xi      = configuration['xi']       # momentum coefficient
    epochs  = configuration['epochs']   # number of epochs

    mom     = [ 0 * p for p in self.params ]
    bar     = tqdm(range(epochs))
    for i in bar:
      gradient = self._batch_grad(self.params, X, Y)
      for j in range(len(self.params)):
        mom[j] = mom[j] * xi - lr * gradient[j]
        self.params[j] += mom[j]
      bar.set_description("L = %.12f" % self.batch_loss(X,Y))

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
    return self._loss(self.params,x,y)

  #########################################################################################

  def _batch_loss(self,params,X,Y):
    return jax.vmap(lambda x,y: self._loss(params, x,y) )(X,Y).mean()

  def batch_loss(self,X,Y):
    return self._batch_loss(self.params,X,Y)

  #########################################################################################

  def activation(self, x):
    return jnp.maximum(0,x)

  def init_params(self,topology):
    params = []
    for (t,t_next) in zip(topology[:-1], topology[1:]):
      W = np.random.randn(t_next, t) * np.sqrt(1./(t * t_next))
      b = np.zeros(t_next)
      params.append(W)
      params.append(b)
    return params

  #########################################################################################

  def clone_params(self):
    return [1. * p for p in params]
