#!usr/bin/env python

# Generates the sine dataset as a csv


import numpy as np
import jax
import jax.numpy as jnp
import sys

def sines(N,seed):
  np.random.seed(seed)
  X   = np.random.uniform(size = N) + 0.5
  t   = np.linspace(0,1,500)

  def y(x,t):
      return jnp.sin(2 * jnp. pi / x * t)

  Y   = jax.vmap(lambda x: y(x,t))(X)
  return X[:,None],Y  # return covariates  as a matrix for  compatibility


def main():
  csv_name = sys.argv[1]
  seed     = int(sys.argv[2])

  X,Y      = sines(500, seed)
  np.savetxt(csv_name + "_covariates.csv", X, delimiter = ',')
  np.savetxt(csv_name + "_timeseries.csv", Y, delimiter = ',')

if __name__ == '__main__':
  main()
