#!/usr/bin/env python

# imports
import sys
import os

# add the custom module folder for inputs, with absoulte path with respect to script position
def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(get_script_path() + "/../../libs/jaxneuralnetworks")

import network

# produces a neural network whose output is the sum of
# another neural network and a linear operator

class resnetwork():
  def __init__(self, net, W):

    # polymorphism

    if type(net).__name__ == 'list':
       # costruction from weights
       topology     =   [ net[0].shape[1]  ] + [len(t) for t in net[1::2]]
       params       =   net.copy()

    else:
       # passing directly the network
       params = net.clone_params()
       topology = net.topology

    self.network        = network.network(topology)
    self.network.params =  params
    self.W              = W.copy()

  # combined prediction from the linear regressor and the
  # the neural network

  def predict(self,x):
    return self.network.predict(x) + self.W @ x

  def batch_predict(self, X):
    return jax.vmap(self.predict)(X)
