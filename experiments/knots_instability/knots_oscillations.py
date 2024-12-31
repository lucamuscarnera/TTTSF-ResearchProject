#!/usr/bin/env python
from   tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from scipy.interpolate import UnivariateSpline

# compute the optimal knots configuration (and the required smoothing) that
# is maximal under a minimum relative accuracy constraint
def optimal_s_for_prediction(t,y):
      s_min = 1e-4
      s_max = 10.

      def loss(spl,t,x):
        return np.sum( (spl(t) - x)**2 ) / np.sum( x**2 )

      req_acc = 1e-2

      # binary search proceduure
      s     = (s_min + s_max)/2.
      while np.abs(s_max - s_min) > 1e-3:
          spl = UnivariateSpline(t, y, s = s)
          local_knots = spl.get_knots().copy()
          if(loss(spl,t,y) > req_acc):
              s_max = s
              s_min = s_min
          else:
              if(loss(spl,t,y) < req_acc):
                  s_max = s_max
                  s_min = s
          s = (s_min + s_max)/2.
      return list(UnivariateSpline(t, y, s = s).get_knots()), s




def main():
  # load data
  X = np.genfromtxt("../../data/ecg.csv", delimiter = ',')
  X = X[np.random.choice(len(X),500,False)]
  t = np.linspace(0,1,X.shape[1])


  # initialize the knot container
  knots = []
  howmany = []
  for i in tqdm(range(len(X))):
      knot_add,s = optimal_s_for_prediction(t,X[i])
      knots += knot_add
      howmany += [len(knot_add)]

  # visualize the distribution
  axs = plt.figure(figsize = (10,5)).subplots(nrows = 1, ncols = 2)
  axs[0].set_title("distribution of knots")
  axs[0].hist(knots, bins = 100)

  axs[1].set_title("distribution of the optimal number of knots")
  axs[1].hist(howmany, bins = 100)

  plt.savefig("knots_oscillations.pdf",bbox_inches='tight')
  plt.show()


if __name__ ==  '__main__':
  main()
