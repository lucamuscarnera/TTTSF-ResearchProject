#!usr/bin/env python

# Generates the peaks dataset as a csv


import numpy as np
import jax
import jax.numpy as jnp
import sys
import matplotlib.pyplot as plt


def tumor(N,seed):
  np.random.seed(seed)

  rescale = lambda x,a,b: x * (b - a) + a

  age                   = np.random.uniform(size = N)      # 20, 80
  weight                = np.random.uniform(size=  N)      # 40, 100
  initial_tumor_volume  = np.random.uniform(size = N)      # 0.1, 0.5
  dosage                = np.random.uniform(size = N)      # 0, 1

  X = np.c_[age,weight,initial_tumor_volume,dosage]

  g_0   = 1.0
  d_0   = 200.
  phi_0 = 10.

  g     = g_0 * (rescale(age,20,80) / 20.0) ** (0.5)
  d     = d_0 * dosage/rescale(weight,40,100)
  phi   = 1./(1 + np.exp(- dosage * phi_0))

  t     = np.linspace(0,1,200)
  Y     = np.array([
        (phi_i * np.exp( - d_i * t  ) + (1 - phi_i) * np.exp(g_i * t) - 1) * itv_i
    for g_i,d_i,phi_i,itv_i in zip(g,d,phi, rescale(initial_tumor_volume,0.1,0.5))
  ])

  plt.figure()
  plt.plot(t,Y[:10].T)
  plt.show()
  return  X,Y


def main():
  csv_name = sys.argv[1]
  seed     = int(sys.argv[2])

  X,Y      = tumor(2000, seed)
  np.savetxt(csv_name + "_covariates.csv", X, delimiter = ',')
  np.savetxt(csv_name + "_timeseries.csv", Y, delimiter = ',')

if __name__ == '__main__':
  main()
