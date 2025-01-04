#!usr/bin/env python

# classic imports
import pickle
import sys

# custom imports
import libsfinder
from nwcompression.compute import NWCompression as nwc
from jaxneuralnetworks import network as jnn
from jaxneuralnetworks import resnetwork as jrnn

def loadmodel(file_model):
  with open(file_model, 'rb') as file:
    model = pickle.load(file)  # Deserialize and load

  # rebuild the encoder
  encoder = jrnn.resnetwork(  model['encoder']['forward']['params'],  model['encoder']['forward']['W'] )
  encoder_backward = jrnn.resnetwork(  model['encoder']['backward']['params'],  model['encoder']['backward']['W'] )

  # rebuild the decoder
  embedding = model['decoder']['embedding']
  timeseries = model['decoder']['timeseries']
  decoder = nwc(embedding, timeseries)

  return encoder, decoder
