# Full Pipeline 

The following folder hosts a self contained playground to try the toolkits presented in the last 2 sections of the article.
It provides, moreover, two additional folders.

- ```sampledata``` contains the python scripts that are able to generate the synthetic data used in the experiments
- ```sampleconfig``` contains the json files that can be used to run the training procudure


# Script launch structure

- ```./sampledata/*gen.py <name>  <seed>``` --> creates two csv files <name>_covariates.csv and  <name>_timeseries.csv 
- ```./training <covariates_csv> <timeseries_csv> <output> <config> ``` the first two fields describe the input dataset (see previous command for an example)  and <output> is instead the pkl  file that will contain the model. Config finally is the json configuration file that manages the training 
