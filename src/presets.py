"""
Created on Fri Jun 20 20:06:56 2025

@author: alex

File which provides a number of presets for ease of use, if you find any missing which you think
should be here, email me at: alexbenitezcanelles@gmail.com
"""
import numpy as np
import matplotlib .pyplot as plt
import general_tools as gt



Tilaser = gt.config
Tilaser.calculation_cycles = 60
Tilaser.ppcycle = 200
Tilaser.padding = 10
Tilaser.wavelength = 0.8e-3
Tilaser.peak_intensity = 1e14
Tilaser.pulse_shape = 'gaussian'
Tilaser.pulse_duration = 35 # In fs
Tilaser.ionization_potential = 12.13
Tilaser.tau_window_length = 1.5
Tilaser.tau_dropoff_pts = 0.3

field = gt.generate_pulse(Tilaser)
t = gt.generate_t(Tilaser)
resp = gt.dipole_response([[0,0,0]],field,Tilaser)
plt.plot(t,field)





