import numpy as np
 def softmax(x):
	s = sum(np.exp(x))
	for i in range(len(x)):
		x[i] = np.exp(x[i])/s
	return x

