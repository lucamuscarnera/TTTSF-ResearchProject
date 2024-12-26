# Knots Instability

We provide an example of a Timeseries Forecasting (from static features) problem where  we have an apparent 
contradiction. The problem is in fact solvable by a linear model, but knots position suffers from high variance.

This would imply that a classifier based on B-Splines could be ineffective in capturing the shape of the time series
even in the paradoxical case in which a linear classifier would be able to obtain high accuracy.

