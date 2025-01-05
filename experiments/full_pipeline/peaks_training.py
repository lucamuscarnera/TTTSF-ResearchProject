#!/usr/bin/env python

# classic imports
import numpy as  np
import matplotlib.pyplot as  plt
import pickle
import sys

# parallel computing
import jax
import jax.numpy as jnp

# custom imports
import libsfinder
from nwcompression.compute import NWCompression as nwc
import jaxneuralnetworks
from jaxneuralnetworks import network as jnn
from jaxneuralnetworks import resnetwork as jrnn
from peaks_data import peaks
import json


def main():

  # argv partition

  covariates_csv = sys.argv[1]
  timeseries_csv = sys.argv[2]
  output_pkl     = sys.argv[3]
  config_json    = sys.argv[4]
  with open(config_json, 'r') as file:
    config_data  = json.load(file)
    compressor_config = config_data['compressor']
    network_config    = config_data['network']


  # generate  data
  X            = jnp.array(np.genfromtxt(covariates_csv,delimiter=','))
  Y            = jnp.array(np.genfromtxt(timeseries_csv,delimiter=','))
  N            = len(X)

  # train the decoder and  latent  representation
  compressor   = nwc()
  compressor.fit(Y, compressor_config)

  # train the encoder
  def phi(X):
    return np.c_[ np.ones((len(X),1)), X]

  # train the linear encoder
  net_topology = network_config['topology']
  W            = np.linalg.pinv(phi(X)) @ compressor.E
  res          = compressor.E - (phi(X) @ W)
  net          = jnn.network(net_topology)


  net.train(X, res, network_config)   # use the configuration loaded  from the json
  encoder = jrnn.resnetwork(net,W)

  E_hat  = phi(X) @ W + net.batch_predict(X)



  # train the inverse map
  W_inv      = np.linalg.pinv(phi(E_hat)) @ X
  res        = X - (phi(E_hat) @ W_inv)
  net_inv    = jnn.network(net_topology)

  net_inv.train(E_hat, res, network_config)
  encoder_inv = jrnn.resnetwork(net,W)
  X_hat  = phi(E_hat) @ W_inv + net_inv.batch_predict(E_hat)


  # show real embedding and predicted
  coloring = np.c_[E_hat[:,0],E_hat[:,1], 0 * E_hat[:,0]]
  coloring = (coloring - coloring.min(axis = 0)[None,:]) / (coloring.max(axis = 0) - coloring.min(axis=0))[None,:]
  axs = plt.figure(figsize = (10,5)).subplots(nrows = 2, ncols = 2)

  ## show  reconstructed embedding and original embedding
  axs[0][0].scatter(X_hat[:,0],X_hat[:,1], c = coloring )
  axs[0][1].scatter(X[:,0], X[:,1], c = coloring)

  ## show original data and reconstruction from embedding
  axs[1][0].scatter(E_hat[:,0],E_hat[:,1], c = coloring )
  axs[1][1].scatter(compressor.E[:,0], compressor.E[:,1], c = coloring)
  # plt.show()

  # test some predictions

  # save the pickle
  with open(output_pkl, 'wb') as file:
    model = {
       'encoder' : {
             'forward' : {
                 'W' : W.copy(),
                 'params' : net.params
              },
             'backward' : {
                 'W' : W_inv.copy(),
                 'params' : net_inv.params
              }
       },
       'decoder': {
             'embedding': compressor.E.copy(),
             'timeseries': compressor.Y.copy()
       }
    }
    pickle.dump(model, file)  # Serialize and save

if __name__=='__main__':
  main()

