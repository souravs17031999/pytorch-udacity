import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(y,p):
    loglist=[]
    for i in range(len(y)):
        if y[i]==1:
            loglist.append(-(np.log(p[i])))
        else:
            loglist.append(-(np.log(1-p[i])))
    return(sum(loglist))
	