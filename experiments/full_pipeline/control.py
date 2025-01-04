#!usr/bin/env python

# classic imports
import pickle
import sys

# custom imports
import libsfinder
from nwcompression.compute import NWCompression as nwc
from jaxneuralnetworks import network as jnn
from jaxneuralnetworks import resnetwork as jrnn
from loadmodel import loadmodel
from peaks_data import peaks

def main():
  if len(sys.argv) == 1:
    print("\t\t +--------------------------------------------+")
    print("\t\t | [Error] Missing model data. Run as         |")
    print("\t\t | >> $ conda run control.py namemodel.pkl    |")
    print("\t\t +--------------------------------------------+")
    return


  file_model = sys.argv[1]
  encoder, decoder = loadmodel(file_model)

  X_test,Y_test    = peaks(100, seed = 44)

  # try to reconstruct data
  embedding = encoder.batch_predict(X_test)
  Y_hat     = decoder.batch_predict(embedding)

if __name__ == '__main__':
  main()
