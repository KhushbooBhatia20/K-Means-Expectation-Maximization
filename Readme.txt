--------------------------------------------------------------------------
Implementation of K-Means & Expectation Maximization
--------------------------------------------------------------------------

In this project, I have implemented K-Means & Expectation Maximization on two data sets. My first data set is 'SGEMM GPU Kernel Performance Prediction' and my second data set is 'Rain in Australia'.

-----------------
Dataset Source:
-----------------

We have used the SGEMM GPU kernel performance Data Set available for download at -

https://archive.ics.uci.edu/ml/datasets/SGEMM+GPU+kernel+performance 

Second Data set can be obtained from Kaggle, link to the dataset is given below –

https://www.kaggle.com/jsphyg/weather-dataset-rattle-package

---------------
Prerequisites: 
---------------

Below Packages are prerequisites to run K-Means & Expectation Maximization-

import numpy as np
import pandas as pd
from pandas import *
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import accuracy_score
from sklearn import metrics
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier  
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector
