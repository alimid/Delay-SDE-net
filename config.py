# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:17:25 2023

@author: alise
"""

#----- Change these 
# f = drift, ga = aleatoric diffusion and ge = epistemic diffusion

# Learning rate
lr_f = 0.0005
lr_ga = 0.000005
lr_ge = 0.01

# Number of epochs
n_epochs_f = 500
n_epochs_ga = 500
n_epochs_ge = 20

# Batch size: Number of days/time points in your data (e.g. number of days in a year)
batch_size = int(365)

# Iterations: Number of years/iterations in training and test data
Iter = 5
Iter_test = 2

# Name used in data_loader to load your dataset
dataset_name = 'load_wind_3sound'

# Random state / seed
state = 111

# Number of lags to use (excluding the present day)
p = 3

# Number of variables/features (including time)
m = 9

# Number of days forward to predict: Keep zero for real-time inference
l = 0

#-- Settings for the epistemic uncertainty

# Sigma_max, maximum value of epistemic uncertainty (maximum standard deviation due to epistemic uncertainty)
const_ep = 40

# Minimum distanse of new OOD point to real dataset
d_min=6.0

# Distance new OOD point will "jump" randomly from the original point
d_off=5.8

# Softness, which decides how the OOD points creep in pockets or the original data
softness = 0.5

incl_y = False