import numpy as np
def sigm(l):
	l1=[]
	for i in l:
		l1.append(1/(1+np.exp(-i)))
	print(l1)
	input()
if __name__ == '__main__':
	sigm(list(map(int,input().strip().split())))	