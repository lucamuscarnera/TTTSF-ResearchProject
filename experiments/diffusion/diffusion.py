#!/usr/bin/env python

# imports

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# custom imports

import libsfinder
from jaxneuralnetworks import network as jnn


def main():
  # simulation data

  N     = 100
  steps = 10

  # define the initial data and the final data
  X_0 = np.random.uniform(N, 2)
  X_f = np.random.randn(N,2)

  # initialize buffer
  B = X_0.copy()
  for i in range(steps):
    X_next = diffusionengine.homotopy(X_0, X_f, i)
    W      = np.linalg.pinv(B) @ X_next

if __name__ == '__main__':
  main()
