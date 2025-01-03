#!/usr/bin/env python

# custom imports
import libsfinder
from nwcompression.compute import NWCompression as nwc

# classic imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import jax
import jax.numpy as jnp

def main():
  # generate 100 data points for the peak dataset
  N   = 1000
  classes = [1,2,3,4]
  axs = plt.figure(figsize = (20,4)).subplots(nrows = 1,ncols = len(classes) + 1)

  np.random.seed(6_01_2025)
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
    axs[0].plot(t,Y[X_1 == i][0]   , alpha = 0.5, color = 'C%d' % (i-1))   # trick for forcing the legend to show the correct colors
  axs[0].legend(["c = %d" % i for i in classes])

  for i in classes:
    axs[0].plot(t,Y[X_1 == i][1:10].T, alpha = 0.5, color = 'C%d' % (i-1))

  for i in classes:
    plot = i
    axs[plot].plot(t,Y[X_1 == i][1:10].T, alpha = 0.5, color = 'C%d' % (i-1))
    axs[plot].set_title("$c = %d$" % i)

  plt.savefig("peaks.pdf",bbox_inches='tight')

  # train
  print("Compressing")
  compressor = nwc()
  compressor.fit(Y, nwc.base_configuration)
  plt.figure()
  for i in classes:
    plt.scatter(compressor.E[X_1 == i][0,0], compressor.E[X_1 == i][0,1], color = "C%d" % (i-1), label = '$c = %d$' % i)

  for i in classes:
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", [  plt.get_cmap('tab10')(i-1), "white"])
    plt.scatter(compressor.E[X_1 == i][1:,0], compressor.E[X_1 == i][1:,1],  c = X_2[X_1 == i][1:], cmap = cmap )

  plt.legend()

  print("Saving dimensionality reduction")
  plt.savefig("reduction.pdf",bbox_inches='tight')


if __name__ == '__main__':
  main()
