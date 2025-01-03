#!/usr/bin/env python

# custom imports
import libsfinder
from nwcompression.compute import NWCompression as nwc
import sys


# classic imports
import numpy as np

# graphics
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from PIL import Image

# GPU
import jax
import jax.numpy as jnp
import numpy as np

def save_gif(fname, data,c):
  fig, ax = plt.subplots()

  # set the correct xlim and ylim
  min_x = np.min([d[:,0].min() for d in data])
  min_y = np.min([d[:,1].min() for d in data])

  max_x = np.max([d[:,0].max() for d in data])
  max_y = np.max([d[:,1].max() for d in data])

  ax.set_xlim(min_x,max_x)
  ax.set_ylim(min_y,max_y)

  # associate a color to each class
  colors = ['C%d' % (i-1) for i in c]

  # Scatter plot initialization
  scatter = ax.scatter(data[0][:,0], data[0][:,1], c = colors, s=50)

  # Update function for animation
  def update(frame):
    # Extract current matrix
    data_now = data[frame]
    # Set scatter plot points
    scatter.set_offsets(np.array(data_now))
    return scatter,

  # Create animation
  ani = animation.FuncAnimation(fig, update, frames=len(data), interval=1, blit=True)

  # Save animation as GIF
  ani.save(fname, writer='pillow')

def main():
  make_gif = False
  if len(sys.argv) > 0:
    gif_name = sys.argv[1]
    make_gif = True
    print("save into %s" % gif_name)

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
  if make_gif:
    intermediates = compressor.fit(Y, nwc.base_configuration, return_intermediates = True)
  else:
    compressor.fit(Y, nwc.base_configuration, return_intermediates = False)

  plt.figure()
  for i in classes:
    plt.scatter(compressor.E[X_1 == i][0,0], compressor.E[X_1 == i][0,1], color = "C%d" % (i-1), label = '$c = %d$' % i)

  for i in classes:
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", [  plt.get_cmap('tab10')(i-1), "white"])
    plt.scatter(compressor.E[X_1 == i][1:,0], compressor.E[X_1 == i][1:,1],  c = X_2[X_1 == i][1:], cmap = cmap )

  plt.legend()

  print("Saving dimensionality reduction")
  plt.savefig("reduction.pdf",bbox_inches='tight')

  if make_gif:
  # creo la gif
    print("Saving the animation...")
    save_gif(gif_name,intermediates, X_1)

if __name__ == '__main__':
  main()
