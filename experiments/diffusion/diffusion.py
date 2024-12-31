#!/usr/bin/env python

# imports
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# custom imports
import libsfinder
from jaxneuralnetworks import network as jnn
from jaxneuralnetworks import resnetwork as jrnn
from diffusionengine import homotopy, homotopy_interval

def phi(X):
  return np.c_[ np.ones((len(X),1)), X]

def main():
  print("start")

  # simulation data
  N     = 1000
  steps = 30

  # define the initial data and the final data
  X_0 = np.random.randn(N, 2)
  X_f = X_0 - X_0 ** 2

  # initialize buffer
  B = X_0.copy()
  a = homotopy_interval(steps)
  plt.ion()
  graph = plt.scatter(X_0[:,0],X_0[:,1], color = 'red', alpha = 0.5)
  graph_2 = None
  plt.pause(0.1)

  for i in range(1, steps):
    print(i)
    X_next = homotopy(X_0, X_f, a[i])

    old_error = ((X_next - B)**2).mean()
    print("starting error = %.10f" % old_error)
    # compute the linear model

    W      = np.linalg.pinv(phi(B)) @ X_next
    linear_error =  jnp.mean( (X_next - (phi(B)@W))**2 )
    print("linear error = %.10f" % linear_error )

    # construct residual as the deviation of the real next step from the linear prediction
    res    = X_next - (phi(B) @ W)

    # fit residual with a neural network
    net    = jnn.network([B.shape[1], 300, 300, X_next.shape[1]])
    net.train(B, res, jnn.network.base_configuration)
    print("network error = %.10f" % ((res - net.batch_predict(B))**2).mean())

    # update buffer with the prediction
    B      = phi(B) @ W + net.batch_predict(B)
    print( "%.10f --> %.10f" % ( old_error, jnp.mean((X_next - B)**2)) )

    # update plot
    if graph_2:
       graph_2.remove()
       graph_2 = plt.scatter(X_next[:,0],X_next[:,1], color = 'black', alpha = 0.5)
    else:
       graph_2 = plt.scatter(X_next[:,0],X_next[:,1], color = 'black', alpha = 0.5)

    graph.remove()
    graph = plt.scatter(B[:,0],B[:,1], color = 'red', alpha = 0.5)
    plt.pause(0.1)
    ##########################################################################################

if __name__ == '__main__':
  main()
