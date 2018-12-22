import numpy as np
def softmax(l):
    s=sum(np.exp(l))
    l1=[]
    for i in l:
        l1.append(np.exp(i)/s)
    print(l1)
if __name__ == '__main__':
	softmax(list(map(int,input().strip().split())))
