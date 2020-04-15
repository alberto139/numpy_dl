import numpy as np 
import pickle

weights_file = 'weights_file.pkl'


with open(weights_file, 'rb') as handle:
            b = pickle.load(handle)


for d in b:
    #print (d)
    if d:
        for key, val in d.items():
            print (key, val.shape)
            print( np.max(val))
            print( np.min(val))