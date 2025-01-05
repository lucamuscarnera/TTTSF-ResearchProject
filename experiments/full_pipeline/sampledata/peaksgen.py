#!usr/bin/env python

# Generates the peaks dataset as a csv


import numpy as np
import jax
import jax.numpy as jnp
import sys

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


def main():
  csv_name = sys.argv[1]
  seed     = int(sys.argv[2])

  X,Y      = peaks(1000, seed)
  np.savetxt(csv_name + "_covariates.csv", X, delimiter = ',')
  np.savetxt(csv_name + "_timeseries.csv", Y, delimiter = ',')

if __name__ == '__main__':
  main()
