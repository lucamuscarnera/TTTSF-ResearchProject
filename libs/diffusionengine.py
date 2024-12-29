#!/usr/bin/env python

# imports
import numpy as np

def homotopy(X_0,X_f,a):
	return X_0 * (1 - a) + X_f * a

def homotopy_interval(steps):
	return np.linspace(0,1, steps)
