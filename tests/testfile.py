'''
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                        INITIALIZATION
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
import numpy as np
import matplotlib.pyplot as plt
import HHGBenitezFinal
from numpy import pi, sqrt, cos, sin, log
from numpy.linalg import norm
import time 

config = HHGBenitezFinal.config()
sau =  HHGBenitezFinal.sau_convert


'''
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                        LASER PROPERTIES:
-Cycles is the number of laser cycles that the pulse will have
-ppcycle is the number of points per cycle, will determine the quality of the output, but increase processing time

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''


config.cycles = 40
config.ppcycle = 200
config.wavelength = 1e-3
config.peak_intensity = 1e14
config.pulse_shape = 'gaussian'
config.pulse_duration = 120
[t, pulse_omega, pulse_coefficients] = HHGBenitezFinal.generate_pulse(config)
config.omega = pulse_omega
config.pulse_coefficients = pulse_coefficients
print(pulse_omega)

'''
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                INTEGRATION AND TARGET PROPERTIES
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''




config.ionization_potential = 12.13

config.tau_interval_length = 0.1
config.tau_dropoff_pts = 0.5

# config.parallelize = True
xv,yv,zv = [np.array([0]) for i in range(3)]


start = time.time()
valrang = [0,100]
lawvals = []
hbar = 1.05457181e-34/(1.6*1e-19)
c = 3e8

w = c*2*pi/(config.wavelength*1e-3)
responsebig = []

# This is just the plane wave driving field function from the main module Ive repurposed to draw the plot
def pwdf(omega,lconfig):
    # omega = lconfig.omega
    a = np.fft.ifft(np.conjugate(lconfig.pulse_coefficients),  axis=1)
    # pulse_coefficients = lconfig.pulse_coefficients
    E0_SI = np.sqrt(2*lconfig.peak_intensity*10000/299792458/8.854187817e-12)
    E0 = sau(E0_SI, 'E', 'SAU', lconfig)
    return E0 * a

[omega1,response1] = HHGBenitezFinal.dipole_response(t,[[0,0,0]],config)





'''
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                      PLOTTING FUNCTIONS
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''


omega1 = omega1[np.where(omega1>valrang[0])]
response1 = response1[np.where(omega1>valrang[0])]
response1 = np.log(np.abs(response1[np.where(omega1<valrang[1])])**2)
print(response1)
print('That took {} seconds'.format(time.time()-start)) 

Up = 9.33*(config.peak_intensity/1e14)*(config.wavelength/1e-3)**2
cutoff = (config.ionization_potential + 3.17*Up )/(hbar*w)

fig,axs = plt.subplots(2,1)

axs[0].plot(omega1[np.where(omega1<valrang[1])],response1)

axs[0].set_xlabel('High Harmonic Order')
axs[0].set_ylabel('Intensity (arbitary log scale)')
axs[0].set_title('Plot of Harmonic Response in Xenon and Pulse')

axs[1].plot(t,pwdf(pulse_omega,config)[0])

# axs[1].set_xlabel('Time')
# axs[1].set_ylabel('Intensity (arbitary log scale)')
axs[1].set_title('Plot of Pulse')

plt.tight_layout()
plt.show()
plt.clf()
plt.close('all')