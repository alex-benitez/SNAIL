import numpy as np
import matplotlib.pyplot as plt
import HHGBenitezFinal
from numpy import pi, sqrt, cos, sin, log
from numpy.linalg import norm
import time 
from scipy.io import loadmat

firststart = time.time()
config = HHGBenitezFinal.config()
sau =  HHGBenitezFinal.sau_convert


'''
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                                                                        LASER PROPERTIES:
-Cycles is the number of laser cycles that the pulse will have
-ppcycle is the number of points per cycle, will determine the quality of the output, but increase processing time
-Pulse duration (in femtoseconds) 
-Pulse shape, can be cos_sqr, gaussian, super_gaussian,

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''


config.cycles = 160
config.ppcycle = 100
config.wavelength = 1e-3
config.peak_intensity = 1e14
config.pulse_shape = 'cos_sqr'
config.pulse_duration = 160
config.ionization_potential = 12.13

# config.ppcycle=int(config.points*499/23000)
# [t, pulse_omega, pulse_coefficients] = HHGBenitezFinal.generate_pulse(config)
# config.omega = pulse_omega
# config.pulse_coefficients = pulse_coefficients
# start = time.time()
# [omega1,response1] = HHGBenitezFinal.dipole_response(t,[[0,0,0]],config)


# print(response1)

# Up = 9.33*(config.peak_intensity/1e14)*(config.wavelength/1e-3)**2
# cutoff = (config.ionization_potential + 3.17*Up )/(hbar*w)

fig,axs = plt.subplots(1,1)

# axs.plot(omega1[np.where(omega1<valrang[1])],response1)

# axs.set_xlabel('High Harmonic Order')
# axs.set_ylabel('Intensity (arbitary log scale)')
# axs.set_title('Plot of Harmonic Response in Different Mediums')
# slowmark = [0.1438131332397461, 0.13239431381225586, 0.15754342079162598, 0.1707460880279541, 0.21477699279785156, 0.2715623378753662, 0.326305627822876, 0.40293335914611816, 0.5154860019683838, 0.6218922138214111, 0.7987756729125977,1.058110237121582, 1.5042004585266113, 1.4410650730133057]
# points = [5000,6000,7000,8000,10000,12000,16000,20000,26000,32000,40000,48000,55000,64000]

# maxmark = [0.21,0.23,0.28,0.23,0.36,0.44,0.71,1.15,1.83,2.84,4.3,6.31,7.87,10.25]
# plt.plot(points,maxmark,'bs',markerfacecolor='none',linewidth=1.5,label='HHGMax')
# plt.plot(points,slowmark,'rs',markerfacecolor='none',linewidth=1.5,label='SLOW')
# plt.scatter(points,maxmark)

def pwdf(omega,lconfig):
    # omega = lconfig.omega
    a = np.fft.ifft(np.conjugate(lconfig.pulse_coefficients),  axis=1)
    # pulse_coefficients = lconfig.pulse_coefficients
    E0_SI = np.sqrt(2*lconfig.peak_intensity*10000/299792458/8.854187817e-12)
    E0 = sau(E0_SI, 'E', 'SAU', lconfig)
    return E0 * a

valrang = [0,100]
points = [i*1000 for i in range(5,60)]
# timetaken = np.zeros((55,10))
# for pos,point in enumerate(points):
#     for i in range(10):
#         start = time.time()
#         config.ppcycle = int(point/config.cycles)
#         if point < 10000:
#             config.tau_interval_length = 1
#             config.tau_dropoff_pts = 0.5
#         else:
#             config.tau_interval_length = 0.6
#             config.tau_dropoff_pts = 0.5
#         [t, pulse_omega, pulse_coefficients] = HHGBenitezFinal.generate_pulse(config)
#         config.omega = pulse_omega
#         config.pulse_coefficients = pulse_coefficients
#         [omega1,response1] = HHGBenitezFinal.dipole_response(t,[[0,0,0]],config)
#         timetaken[pos,i] = time.time()-start
#         if point%10000 == 0:
#             print(timetaken)
        # if point%10000 == 0:
        #     fig,axs = plt.subplots(1,1)
        #     omega1 = omega1[np.where(omega1>valrang[0])]
        #     response1 = response1[np.where(omega1>valrang[0])]
        #     response1 = np.log(np.abs(response1[np.where(omega1<valrang[1])])**2)
        #     axs.plot(omega1[np.where(omega1<valrang[1])],response1)
    
        #     axs.set_xlabel('Time')
        #     axs.set_ylabel('Intensity (arbitary log scale)')
        #     axs.set_title('Plot of Pulse')
    
        #     plt.tight_layout()
        #     plt.show()
        #     plt.clf()
        #     plt.close('all')
            
        # print('For {} points it took {} seconds'.format(t.size,time.time()-start))
    
    
# np.save('/home/alex/Desktop/Python/HHGBenitez/benchsnail.npy',timetaken)
timetaken = np.load('/home/alex/Desktop/Python/HHGBenitez/benchsnail.npy')
maxtaken = loadmat('/home/alex/Desktop/Python/HHGBenitez/benchmax.m',appendmat=False)['biglist'][0][1:-1]
maxtaken[0] = 0.20
print(maxtaken.size)
# color='tab:blue'
# markerfacecolor="none", markersize="15" and markeredge color="red"
# plt.plot(points,maxtaken,label='HHGMax',color='tab:orange')
fig,axs = plt.subplots(1,1)
plt.plot(points,maxtaken,'ko',label='HHGMax', markerfacecolor='none', markersize=4,markeredgecolor="orange",markeredgewidth=1.5)
plt.plot(points,np.sum(timetaken,axis=1)/10,'ks',label='SNAIL', markerfacecolor='none', markersize=4,markeredgecolor="blue",markeredgewidth=0.8)
plt.rcParams.update({'font.size': 16})
plt.title('Benchmark of HHGMax against SNAIL')
plt.xlabel('Number of points in the driving field')
plt.ylabel('Time taken to process (s)')
plt.legend()
fig.set_size_inches(7, 5)
plt.savefig('/home/alex/Desktop/Benchmarkplot2.png',dpi=300)
# # axs[1].plot(t,pwdf(pulse_omega,config)[0])

# # axs[1].set_xlabel('Time')
# # axs[1].set_ylabel('Intensity (arbitary log scale)')
# # axs[1].set_title('Plot of Pulse')

plt.tight_layout()
plt.show()
plt.clf()
plt.close('all')
