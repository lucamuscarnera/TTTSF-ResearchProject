#!usr/bin/env python

import numpy as np
import jax
import jax.numpy as  jnp
import matplotlib.pyplot as plt

def sine(N, seed):
  t = np.linspace(0,2 * np.pi,100)
  X = np.random.uniform(size = N) + 1
  Y = jax.vmap(lambda x: jnp.sin(2 * np.pi / x * t) )(X)
  return X,Y


X,Y = sine(10,123)
plt.figure()
plt.plot(Y.T)
plt.show()
