#!usr/bin/env python

# classic imports
import pickle
import sys
import matplotlib.pyplot as plt

# custom imports
import libsfinder
from nwcompression.compute import NWCompression as nwc
from jaxneuralnetworks import network as jnn
from jaxneuralnetworks import resnetwork as jrnn
from loadmodel import loadmodel
from peaks_data import peaks
from interactive_prompt import DrawPrompt

def main():
  if len(sys.argv) == 1:
    print("\t\t +--------------------------------------------+")
    print("\t\t | [Error] Missing model data. Run as         |")
    print("\t\t | >> $ conda run control.py namemodel.pkl    |")
    print("\t\t +--------------------------------------------+")
    return


  file_model = sys.argv[1]
  encoder, decoder, encoder_backward = loadmodel(file_model)

  X_test,Y_test    = peaks(9, seed = 45)

  # try to reconstruct data
  embedding = encoder.batch_predict(X_test)
  Y_hat     = decoder.batch_decode(embedding)


  axs = plt.figure(figsize = (9,9)).subplots(nrows = 3,ncols = 3).flatten()
  plt.suptitle("some predictions...")
  for i,ax in enumerate(axs):
    ax.plot(Y_hat[i])
    ax.plot(Y_test[i])

  plt.show()

  # control

  sample = 6
  x_0                = X_test[sample]
  starting_embedding = embedding[sample]
  prompt = DrawPrompt(Y_hat[sample])
  y_wanted = prompt.data

  # Find the optimal embedding
  e_final = (decoder.backward_predict(y_wanted, starting_embedding))

  y_obtained = decoder.decode(e_final)
  plt.plot(y_obtained)
  plt.plot(Y_hat[sample])
  plt.plot(y_wanted)
  plt.show()


  x_optimal = encoder_backward.predict(e_final)

  # bring it back to feature space
  for i,x_0_i, x_f_i in zip(range(len(e_final)),x_0, x_optimal):
    print(" Feature [%d] \t\t\t %.4f \t -> \t %.4f" % (i,x_0_i,x_f_i) )

if __name__ == '__main__':
  main()
