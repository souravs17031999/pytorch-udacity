# function to calculate probability for given combination of equation -w1*0.4+w2*0.6+b
import numpy as np
def sigmoid(x):
	return 1/(1+np.exp(-x))
def probability(w1,w2,b):
	return w1*0.4+w2*0.6+b