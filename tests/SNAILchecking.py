"""
Created on Thu Jun 12 17:09:51 2025

@author: alex
"""
import numpy as np
import matplotlib.pyplot as plt

stringlist = [0,37,74,111,148,185]

single = np.load('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/Bt.npy')

for i in stringlist:
    parallel = np.load('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/Bt{}.npy'.format(i))
    if i != 185:
        print(np.where(np.isclose(single[i:i+37],parallel)==False)[0].size)
    else:
        print(np.where(np.isclose(single[i:],parallel)==False))
    


parallel1 = np.load('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/Bt0.npy')

# parallel3 = np.load('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/pst74.npy')

# parallel4 = np.load('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/pst111.npy')

# parallel5 = np.load('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/pst148.npy')

# parallel6 = np.load('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/pst185.npy')






# print(single)
# print('\n')
# print(paral)


# print(np.where(np.isclose(single,parallel1)==False))

# print(np.where(np.isclose(single,paral)==False))

