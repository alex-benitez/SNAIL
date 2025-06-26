"""
Created on Fri Jun 20 05:09:08 2025

@author: alex
"""
import numpy as np
import matplotlib.pyplot as plt
import general_tools as gt
import plotting_tools as pt

config = gt.Tilaser
cutoff = []
for i in range(7):
    config.Ip += i*0.3
    config.peak_intensity += 0.12e14*i
    field = gt.generate_pulse(config)
    resp = gt.dipole_response(field,config,logscale=True)
    cutoff.append(cutoff, gt.cutoff(config))
    pt.harmonic_plot(resp + 40*i)

final = np.array(cutoff)
plt.plot(final[:,0],final[:,1],'r--')    

[omega1,response1] = gt.dipole_response([[0,0,,driving_field,config)
pt.driving_spectra(response1,driving_field,style='horizontal')





