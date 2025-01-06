#!usr/bin/env python

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import libsfinder
import sys
from loadmodel import loadmodel
import readline
import os

# small graphics
def logo():
    print('\t\t\t', r'########\                     ##\           ##\           ')
    print('\t\t\t', r'##  _____|                    ## |          \__|          ')
    print('\t\t\t', r'## |      ##\   ##\  ######\  ## | ######\  ##\ #######\  ')
    print('\t\t\t', r'#####\    \##\ ##  |##  __##\ ## | \____##\ ## |##  __##\ ')
    print('\t\t\t', r'##  __|    \####  / ## /  ## |## | ####### |## |## |  ## |')
    print('\t\t\t', r'## |       ##  ##<  ## |  ## |## |##  __## |## |## |  ## |')
    print('\t\t\t', r'########\ ##  /\##\ #######  |## |\####### |## |## |  ## |')
    print('\t\t\t', r'\________|\__/  \__|##  ____/ \__| \_______|\__|\__|  \__|')
    print('\t\t\t', r'                    ## |                                  ')
    print('\t\t\t', r'                    ## |  command line tool for model     ')
    print('\t\t\t', r'                    \__|  interrogation                   ')

# Perturbative analysis method
# performs a local PCA around the prediction in  the embedding space
# to understand the degrees of freedom that the prediction is likely to
# have. Then returns the information

def plot_similar(e,E,scale,encoder,decoder, encoder_backward):
  # algorithm for describe locally the distribution in the embedding
  f     = encoder_backward.predict(e)
  base_predict = decoder.decode(e)

  E_loc = E[jnp.linalg.norm(E - e, axis = 1) < scale]
  Y_loc = jax.vmap(decoder.decode)(E_loc)
  X_loc = jax.vmap(encoder_backward.predict)(E_loc)

  axs   = plt.figure(figsize = (10,5)).subplots(nrows = 1, ncols = 2).flatten()
  axs[0].plot(Y_loc.T, color = 'black', alpha = 0.1)
  axs[0].plot(base_predict, color = 'black', lw = 3.0)

  axs[1].boxplot(X_loc, vert=True, patch_artist=True) # box plot of close data
  plt.show()



def perturbative_analysis(e, E, scale, encoder, decoder, encoder_backward):
  base_predict = decoder.decode(e)

  E_loc = E[jnp.linalg.norm(E - e, axis = 1) < scale]
  E_loc = E_loc - e                                                 # ball centered in the computed embedding

  print("Neighbours = %d" % len(E_loc))
  U,s,Vt    = np.linalg.svd(E_loc, full_matrices = False)
  print("s = %s" % s, Vt.shape)


  axs = plt.figure(figsize = (len(s) * 5,5)).subplots(nrows = 1, ncols = len(s))
  if(type(axs).__name__ == 'ndarray'):
    axs = axs.flatten()
  else:
    axs = [axs]


  for i in range(len(s)):
    axs[i].set_title("s = %.3f" % s[i])
    E_vr      = np.linspace(- s[i] , s[i] ,100)[:,None] * Vt.T[:,i]
    E_vr     += e

    # compute the perturbed trajectories
    Y_hat     = jax.vmap(lambda e_loc: decoder.decode(e_loc))(E_vr)

    for j in range(100):
      axs[i].plot(Y_hat[j], color = [j / 100.,0.,1. - j / 100.])
    axs[i].plot(base_predict, lw = 3., color = 'black')
  plt.show()

# Main
def main():
  if len(sys.argv) == 1:
    print("")
    print("[!] Error: no model loaded. Run the application as")
    print("  $ conda run conda run --no-capture-output python3 -u ./explain.py modelname.pkl")
    print("")
    return

  model_file = sys.argv[1]
  encoder, decoder, encoder_backward =  loadmodel(model_file)

  logo()

  while True:
    query = input(">> ")
    tokens = query.split()

    if(tokens[0] == 'q'):
      break
    if(tokens[0] == 'predict'):
      featurevector = np.array(tokens[1:]).astype(float)
      embedding     = encoder.predict(featurevector)
      y_hat         = decoder.decode(embedding)
      plt.plot(y_hat)
      plt.show()

    if(tokens[0] == 'pertanalysis'):
      scale = float(tokens[1])
      featurevector = np.array(tokens[2:]).astype(float)
      # predict the embedding
      embedding     = encoder.predict(featurevector)
      E     = decoder.E
      perturbative_analysis(embedding,E,scale,encoder,decoder,encoder_backward)

    if(tokens[0] == 'similar'):
      scale = float(tokens[1])
      featurevector = np.array(tokens[2:]).astype(float)
      # predict the embedding
      embedding     = encoder.predict(featurevector)
      E     = decoder.E
      plot_similar(embedding,E,scale,encoder,decoder,encoder_backward)

    if(tokens[0] == 'clear'):
      os.system('clear')
      logo()

if __name__ == '__main__':
  main()
