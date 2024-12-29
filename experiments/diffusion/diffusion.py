#!/usr/bin/env python

# imports
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# custom imports
import libsfinder
from jaxneuralnetworks import network as jnn
from diffusionengine import homotopy, homotopy_interval

def main():

  # simulation data
  N     = 100
  steps = 10

  # define the initial data and the final data
  X_0 = np.random.uniform(size = (N, 2))
  X_f = np.random.randn(N,2)

  # initialize buffer
  B = X_0.copy()
  a = homotopy_interval(steps)

  for i in range(1, steps):
    X_next = homotopy(X_0, X_f, a[i])
    W      = np.linalg.pinv(B) @ X_next
    print(W)

    # update buffer with the prediction
    B      = B @ W

if __name__ == '__main__':
  main()
