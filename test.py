import numpy as np
a = np.array([[0,1,0],[1,1,2]])
def featuredecoder(arr):
    return np.argmax(arr,axis=1)
print(featuredecoder(a))