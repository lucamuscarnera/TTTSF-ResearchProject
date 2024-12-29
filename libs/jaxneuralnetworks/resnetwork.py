#!/usr/bin/env python

# produces a neural network whose output is the sum of
# another neural network and a linear operator

class resnetwork(network):
  def __init__(self, network, W):
    self.params = network.clone_params()
    self.W      = W.copy()

  # overload the user prediction function, such that the private _predict method
  # is used to predict the network contribution

  def predict(self,x):
    return _predict(self.params, x) + self.W @ x

