#!usr/bin/env python

# classic imports
import pickle
import sys

# custom imports
import libsfinder
from nwcompression.compute import NWCompression as nwc
from jaxneuralnetworks import network as jnn
from jaxneuralnetworks import resnetwork as jrnn


def main():
  if len(sys.argv) == 1:
    print("\t\t +--------------------------------------------+")
    print("\t\t | [Error] Missing model data. Run as         |")
    print("\t\t | >> $ conda run control.py namemodel.pkl    |")
    print("\t\t +--------------------------------------------+")
    return


  file_model = sys.argv[1]
  with open(file_model, 'rb') as file:
    model = pickle.load(file)  # Deserialize and load

  # rebuild the encoder
  encoder = jrnn.resnetwork(  model['encoder']['forward']['params'],  model['encoder']['forward']['W'] )
  encoder_backward = jrnn.resnetwork(  model['encoder']['backward']['params'],  model['encoder']['backward']['W'] )

  # rebuild the decoder
  embedding = model['decoder']['embedding']
  timeseries = model['decoder']['timeseries']
  decoder = nwc(embedding, timeseries)

  print(encoder.network.params[0].shape)
if __name__ == '__main__':
  main()
