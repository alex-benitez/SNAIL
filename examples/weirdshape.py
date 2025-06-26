"""
Created on Fri Jun 20 12:22:32 2025

@author: alex
"""
import numpy as np
import matplotlib.pyplot as plt


r = np.zeros(16000)
t = np.arange(16000)
r[:2500] = np.square(t[:2500]/5000)
r[2500:5000] = np.exp(t[2500:5000]/2500-1)-(1-r[2499])
r[5000:7500] = np.sin(np.pi*t[5000:7500]/2500) + r[4999]
r[7500:10000] = np.sin(5.3*np.pi*t[7500:10000]/2500) + r[7499]
r[10000:14000] = np.cos(6*np.pi*t[10000:14000]/4000) + r[9999]
r[14000:] = np.flip(np.square(t[:2000]/4000))+r[13999]-(np.square(2000/4000))
r = r-max(r)/2
plt.plot(t,r)
plt.show()


