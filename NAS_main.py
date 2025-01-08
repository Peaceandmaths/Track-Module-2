
from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.datasets import fashion_mnist
from keras.models import Sequential
import numpy as np
from keras.utils import np_utils
import nni 
from keras.callbacks import Callback
from nni.experiment import Experiment
#import fashion_mnist_nas

# Create the search space

search_space = {
    'filter_size_c1':{'_type': 'choice', '_value': [32,64,128]},#frist convolution 
   'filter_size_c2':{'_type': 'choice', '_value': [32,64,128]}, # second convolution
    
  'kernel_size_c1':{'_type': 'choice', '_value': [3,5]},
   'kernel_size_c2':{'_type': 'choice', '_value': [3,5]}, 
    
    'learning_rate':{'_type': 'uniform', '_value': [0.001, 0.01]}, # uniform type because continuous to choose from this range 
    'nb_units': {'_type': 'choice', '_value': [80, 100, 120]}
}

# Update original code : fashion_mnist_nas.py
    # remove hard coded values 
    # Get parameters from nni 
    # report intermediate results 
    # report final results 


experiment = Experiment('local')
experiment.config.trial_command = 'python fashion_mnist_nas.py'
experiment.config.trial_code_directory ='/home/golubeka/Trackmodule2' # same directory
experiment.config.trial_concurrency = 1
experiment.config.max_trial_number = 10
experiment.config.debug = True
# specify search space 
experiment.config.search_space = search_space

# Specify search strategy 
experiment.config.tuner.name = 'TPE' 

# Run the experiment 
experiment.config.experiment_name = 'fashion_mnist_nas'
experiment.run(6027) # specify port on which you want to run 