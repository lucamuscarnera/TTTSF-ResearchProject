#!usr/bin/env python

#  classic imports
import pickle
import sys
import matplotlib.pyplot as plt
import os 
import readline

import warnings
warnings.filterwarnings("ignore")

# custom imports
import libsfinder
from nwcompression.compute import NWCompression as nwc
from jaxneuralnetworks import network as jnn
from jaxneuralnetworks import resnetwork as jrnn
from loadmodel import loadmodel
from peaks_data import peaks
from interactive_prompt import DrawPrompt
import numpy as np

# small graphics
def logo():
 print('\t\t\t', r'  /$$$$$$    Time Series Control /$$  Toolkit                /$$')
 print('\t\t\t', r' /$$__  $$                      | $$                        | $$')
 print('\t\t\t', r'| $$  \__/  /$$$$$$  /$$$$$$$  /$$$$$$    /$$$$$$   /$$$$$$ | $$')
 print('\t\t\t', r'| $$       /$$__  $$| $$__  $$|_  $$_/   /$$__  $$ /$$__  $$| $$')
 print('\t\t\t', r'| $$      | $$  \ $$| $$  \ $$  | $$    | $$  \__/| $$  \ $$| $$')
 print('\t\t\t', r'| $$    $$| $$  | $$| $$  | $$  | $$ /$$| $$      | $$  | $$| $$')
 print('\t\t\t', r'|  $$$$$$/|  $$$$$$/| $$  | $$  |  $$$$/| $$      |  $$$$$$/| $$')
 print('\t\t\t', r' \______/  \______/ |__/  |__/   \___/  |__/       \______/ |__/')
                                                               
def main():
  if len(sys.argv) == 1:
    print("")
    print("[!] Error: no model loaded. Run the application as")
    print("  $ conda run conda run --no-capture-output python3 -u ./explain.py modelname.pkl")
    print("")
    return

  logo()
  print("Loading...")

  file_model = sys.argv[1]
  encoder, decoder, encoder_backward = loadmodel(file_model)

  # control
  while True:
    query = input(">> ")
    tokens = query.split()
    
    if tokens[0]  == 'control':
      x_0 = np.array(tokens[1:]).astype(float)
      starting_embedding  = encoder.predict(x_0)
      starting_trajectory = decoder.decode(starting_embedding)
      
      prompt = DrawPrompt(starting_trajectory)
      y_wanted = prompt.data
      
      # Find the optimal embedding
      e_final = (decoder.backward_predict(y_wanted, starting_embedding))
      
      y_obtained = decoder.decode(e_final)
      plt.plot(y_obtained, label = 'best approximant')
      plt.plot(starting_trajectory, label = 'initial trajectory')
      plt.plot(y_wanted, label = 'input trajectory')
      plt.legend()
      plt.show()

      # bring it back to feature space
      x_optimal = encoder_backward.predict(e_final)
      for i,x_0_i, x_f_i in zip(range(len(e_final)),x_0, x_optimal):
        print(" Feature [%d] \t\t\t %.4f \t -> \t %.4f" % (i,x_0_i,x_f_i) )

    if(tokens[0] == 'clear'):
      os.system('clear')
      logo()
      
    if tokens[0] == 'q':
      break


if __name__ == '__main__':
  main()
