#!/usr/bin/env python
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import knot_selection


def build_capacitor_dataset(N,seed):
  t = np.linspace(0,1,1000)
  RC    = jax.random.uniform(jax.random.PRNGKey(seed),    shape = (N,)) * 1.
  Q_0   = jax.random.uniform(jax.random.PRNGKey(seed + 1),shape = (N,)) * 1.
  X     = jax.vmap(lambda q_0, rc: q_0 * jnp.exp( - t/rc ) )(Q_0,RC)
  return X,t

def main():
  # list of the number of nodes that are going to be tested
  K   = [3,5,7,9]

  print("Generating data...")
  X,t = build_capacitor_dataset(250, 123)

  print("computing the smoothing factors")
  ax = plt.figure(figsize = (10,3)).subplots(nrows = 1,ncols = len(K))

  # set the number of knots
  for i,k in enumerate(K):
    print("Extracting the smoothing distribution [knots = %d]"  % k)

    # extract the smoothing factor and put it into a list
    smoothings = []
    for x in tqdm(X):
      _,s = knot_selection.get_knots_for_single_trajectory(t, x, k)
      smoothings.append(s)

    # computing the histogram of the distribution
    ax[i].set_title("knots = %d" % k)
    ax[i].set_xscale('log')
    ax[i].hist(smoothings, bins = np.geomspace(1e-3, 10., 100))


  plt.savefig("knots_smoothing_inconsistency.pdf",bbox_inches='tight')
  plt.show()


if __name__ == '__main__':
  main()
