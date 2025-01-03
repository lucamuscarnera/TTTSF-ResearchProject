#!/usr/bin/env python

# custom imports
import libsfinder
from nwcompression import compute

# classic imports
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

def main():
  # generate 100 data points for the peak dataset
  N   = 100
  classes = [1,2,3,4]

  X_1 = np.random.choice(classes, size = N)
  X_2 = np.random.uniform(size = N) * 2 +  1
  X   = np.c_[X_1,X_2]
  t   = np.linspace(0,1,100)
  def y(x,t):
    return x[1] * jnp.exp( - 100 * ( t - x[0] * 0.2 )**2 )
  Y   = jax.vmap(lambda x: y(x,t))(X)

  # save plot of the data
  for i in classes:
    plt.plot(t,Y[X_1 == i][0]   , alpha = 0.5, color = 'C%d' % i)   # trick for forcing the legend to show the correct colors

  for i in classes:
    plt.plot(t,Y[X_1 == i][1:].T, alpha = 0.5, color = 'C%d' % i)

  plt.legend(["c = %d" % i for i in classes])
  plt.savefig("peaks.pdf",bbox_inches='tight')

if __name__ == '__main__':
  main()
