# Knots Instability

We provide an example of a Timeseries Forecasting (from static features) problem where  we have an apparent 
contradiction. The problem is in fact solvable by a linear model, but knots position suffers from high variance.

This would imply that a classifier based on B-Splines could be ineffective in capturing the shape of the time series
even in the paradoxical case in which a linear classifier would be able to obtain high accuracy. 
Moreover, in a scenario where knots are "approximately" equal for each time series (e.g. they tend to cluster around specific
mean values) knots have a specific physical interpretation, linked to the behaviour of the function. A further contradiction
appears in the moment where such interpretation is lost, and thus the final knots are not representative of the entire 
set of Time series


## The experiment

We propose a model of Dynamical Linear System with Quenched Disorder in the transition matrix $A$.

