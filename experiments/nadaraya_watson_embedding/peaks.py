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
  N   = 1000
  classes = [1,2,3,4]
  axs = plt.figure(figsize = (20,4)).subplots(nrows = 1,ncols = len(classes) + 1)

  X_1 = np.random.choice(classes, size = N)
  X_2 = np.random.uniform(size = N)

  X   = np.c_[X_1,X_2]
  t   = np.linspace(0,1,50)
  def y(x,t):
      return jnp.exp( - 10* x[0] * ( t - x[1])**2 )
  Y   = jax.vmap(lambda x: y(x,t))(X)

  # save plot of the data
  axs[0].set_title("mixed")
  for i in classes:
    axs[0].plot(t,Y[X_1 == i][0]   , alpha = 0.5, color = 'C%d' % i)   # trick for forcing the legend to show the correct colors
  axs[0].legend(["c = %d" % i for i in classes])

  for i in classes:
    axs[0].plot(t,Y[X_1 == i][1:10].T, alpha = 0.5, color = 'C%d' % i)

  for i in classes:
    plot = i
    axs[plot].plot(t,Y[X_1 == i][1:10].T, alpha = 0.5, color = 'C%d' % i)
    axs[plot].set_title("$c = %d$" % i)

  plt.savefig("peaks.pdf",bbox_inches='tight')

if __name__ == '__main__':
  main()
