'''
This is an example of generating single atom response with a simple cos^2 pulse, documentation coming soon
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
from scipy.io import loadmat

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


config.calculation_cycles = 60
config.ppcycle = 200
config.wavelength = 1e-3
config.peak_intensity = 1e14
config.pulse_shape = 'cos_sqr'
config.pulse_duration = 35


# print(pulse_omega.size)

'''
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                INTEGRATION AND TARGET PROPERTIES
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''




config.ionization_potential = 12.13

config.tau_window_length = 1.1# How far back over excursion time to integrate over, as a fraction of a cycle
config.tau_dropoff_pts = 0.4 # Fraction of the integration window past which the integrands drop off to prevent artifacts

# config.parallelize = True
xv,yv,zv = [np.array([0]) for i in range(3)]


start = time.time()
valrang = [0,100]
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
matdata = open('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/d_t.txt')
matlabarray = []
for line in matdata.readlines():
    matlabarray.append(line.split(','))
matlabarray = matlabarray[0]
matlabarray[-1] = matlabarray[-1][:-2]
matlabarray = [float(i) for i in matlabarray]

    

[t, driving_field] = general_tools.generate_pulse(config)
# print(driving_field)
start = time.time()
[omega1,response1] = general_tools.dipole_response(t,[[0,0,0]],driving_field,config)

omega1 = omega1[np.where(omega1>valrang[0])]
response1 = response1[np.where(omega1>valrang[0])]
response1 = np.log(np.abs(response1[np.where(omega1<valrang[1])])**2)
response1 = response1[np.where(omega1<valrang[1])]
output = np.load('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/output.npy')
diff = abs(matlabarray[:-1]-output)
# print(diff[np.where(diff>1e-2)])

# response1 = np.log(np.abs(response1)**2)
# response1 = np.imag(response1)
print('That took {} seconds'.format(time.time()-firststart)) 


fig,axs = plt.subplots(2,1,figsize=(7,4))

# axs[0].plot(omega1,response1)
axs[0].plot(omega1[np.where(omega1<valrang[1])],response1)
axs[0].set_title('Harmonic Response of a Gaussian Pulse')
axs[0].set_xlabel('Harmonic Order')
axs[0].set_ylabel('Int. (Arb. log Scale)')

# axs[0].vlines(int(cutoff),min(response1),max(response1),'k',linewidth=0.5)

# axs[0].set_xlabel('High Harmonic Order')`
t_fs = general_tools.sau_convert(t,'t','SI',config)/1e-15
axs[1].plot(t_fs,driving_field[0])
axs[1].set_xlabel('Time(fs)')
axs[1].set_ylabel('Int. (Arb. Scale)')
# axs[1].set_title('Gaussian Pulse')

plt.savefig('/home/alex/Desktop/Python/SNAIL/Latex/simpleharmonic.png',dpi=300)
plt.tight_layout()
plt.show()

plt.clf()
plt.close('all')

