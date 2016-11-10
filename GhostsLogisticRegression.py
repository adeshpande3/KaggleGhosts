from __future__ import division
import scipy
import numpy as np
from sklearn import linear_model
import csv
import sys

# This program will use Logistic Regression as a machine learning algorithm that predicts
# whether a set of input characteristics represents a ghost, ghoul, or goblin

Xtrain=[] # Will contain the characteristic data for each monster. Xtrain will 
		  # contain info about the monster's bone length, rotting flesh, 
		  # hair length, soul, and color. 
Ytrain=[] # WIll contain a binary label for what type of monster it is 
Xtest=[] # Will contain the characteristic data for the monsters in the test set