#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp


# compute the optimal knots configuration (and the required smoothing) that
# is maximal under a minimum relative accuracy constraint
def optimal_s_for_prediction(t,y):
      s_min = 1e-4
      s_max = 10.

      def loss(spl,t,x):
        return np.sum( (spl(t) - x)**2 ) / np.sum( x**2 )

      req_acc = 1e-4

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

  knots = []
  for i in range(400):
      knot_add,s = optimal_s_for_prediction(t,X[i])
      knots += knot_add[1:-1]
  plt.hist(knots, bins = 100)
  plt.show()

  return 0



def main():
  X = np.genfromtxt("./data/ecg.csv", delimiter = ',')
  np.random.seed(123)
  X = X[np.random.choice(len(X),1000,False)]
  t = np.linspace(0,1,X.shape[1])
  N = len(X)


if __name__ ==  'main':
  main()
