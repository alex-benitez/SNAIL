'''
This is an example of generating single atom response with a simple cos^2 pulse
'''

'''
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                        INITIALIZATION
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
import numpy as np
import matplotlib.pyplot as plt
import general_tools
from numpy import pi, sqrt, cos, sin, log
from numpy.linalg import norm
import time 
import plotting_tools as pt

firststart = time.time()
config = general_tools.config()
sau =  general_tools.sau_convert


'''
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                        LASER PROPERTIES:
-Cycles is the number of laser cycles that the pulse will have
-ppcycle is the number of points per cycle, will determine the quality of the output, but increase processing time
-Pulse duration (in femtoseconds) 
-Pulse shape, can be constant, cos_sqr, gaussian, super_gaussian,

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''


config.calculation_cycles = 40
config.ppcycle = 200
config.wavelength = 1e-3
config.peak_intensity = 1e14
config.pulse_shape = 'gaussian'
config.pulse_duration = 35# In fs

'''
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                INTEGRATION AND TARGET PROPERTIES
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''




config.ionization_potential = 12.13

config.tau_window_length = 1# How far back over excursion time to integrate over, as a fraction of a cycle
config.tau_dropoff_pts = 0.3 # Fraction of the integration window which the integrands drop off to prevent artifacts

xv,yv,zv = [np.array([0]) for i in range(3)]


start = time.time()
valrang = [0,1000]
lawvals = []
hbar = 1.05457181e-34/(1.6*1e-19)
c = 3e8

w = c*2*pi/(config.wavelength*1e-3)
responsebig = []



'''
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                      PLOTTING FUNCTIONS
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
Up = 9.33*(config.peak_intensity/1e14)*(config.wavelength/1e-3)**2
cutoff = (config.ionization_potential + 3.17*Up )/(hbar*w)


    

driving_field = general_tools.generate_pulse(config)


start = time.time()
[omega1,response1] = general_tools.dipole_response([[0,0,0]],driving_field,config)
end = time.time()

pt.driving_harmonics(driving_field,response1,config,harmonic_range=valrang,save_location='/home/alex/Desktop/Python/SNAIL/images/driving_response.png')


plt.show()

print('That took {} seconds'.format(end-start))

plt.clf()
plt.close('all')

