#!usr/bin/env python

# generate peaks data for a certain seed

import numpy as np
import jax
import jax.numpy as jnp

def peaks(N,seed):
  classes = [1,2,3,4]

  np.random.seed(seed)
  X_1 = np.random.choice(classes, size = N)
  X_2 = np.random.uniform(size = N)

  X   = np.c_[X_1,X_2]
  t   = np.linspace(0,1,50)
  def y(x,t):
      return jnp.exp( - 10* x[0] * ( t - x[1])**2 )
  Y   = jax.vmap(lambda x: y(x,t))(X)
  return X,Y
