import numpy as np
import matplotlib.pyplot as plt
import general_tools
pi = np.pi
config = general_tools.config()
sau =  general_tools.sau_convert
plt.ioff()


config.calculation_cycles = 50
config.ppcycle = 200
config.wavelength = 1e-3
config.peak_intensity = 1e14
config.pulse_shape = 'gaussian'
config.pulse_duration = 40
hbar = 1.05457181e-34/(1.6*1e-19)
config.tau_window_length = 1.1
config.tau_dropoff_pts = 0.4
config.parallel = False
c = 3e8
w = c*2*pi/(config.wavelength*1e-3)
# points = [i*1000 for i in range(5,60)]
# timetaken = np.zeros((55,10))
valrang = [0,100]

# results = np.zeros((10,4998))
cutoff = []
fig,axs = plt.subplots(1,1,figsize=(6,4))
for i in range(7):
    config.ionization_potential = 12 + 0.3*i
    config.peak_intensity = 1e14 + 0.12e14*i
    [driving_field] = general_tools.generate_pulse(config)
    [omega1,response1] = general_tools.dipole_response([[0,0,0]],driving_field,config)

    omega1 = omega1[np.where(omega1>valrang[0])]
    omega1 = omega1[np.where(omega1<valrang[1])]
    
    response1 = response1[np.where(omega1>valrang[0])]
    response1 = np.log(np.abs(response1[np.where(omega1<valrang[1])])**2) + 40*i
    # axs.plot(omega1[np.where(omega1<valrang[1])],response1)
    Up = 9.33*(config.peak_intensity/1e14)*(config.wavelength/1e-3)**2
    tempoff = (config.ionization_potential + 3.17*Up )/(hbar*w)
    # tempoff= round(tempoff)
    # tempoff = tempoff + (tempoff%2-1)
    cutoff.append((tempoff,response1[np.argmin(abs(omega1-tempoff))]))
    
    # print(cutoff)
    ratio = i/7
    color = [0.1,0.1+ratio*0.9,0.8]
    current = axs.plot(omega1,response1,color=color)

    # axs.vlines(int(cutoff),min(response1)+10,max(response1),current[-1].get_color(),linewidth=1)
xvals = [i[0] for i in cutoff]
yvals = [i[1] for i in cutoff]

axs.plot(xvals,yvals,'k--',label='Predicted Cutoff')
plt.legend()


    


   
axs.set_xlabel('Harmonic Order')
axs.set_ylabel('Intensity (arbitary log scale)')
axs.set_title('Cutoff Law Demonstration With Simulated Data')
plt.tight_layout() 
plt.savefig('/home/alex/Desktop/Python/SNAIL/images/cutofflaw.png',dpi=300)
plt.show()

config.ionization_potential = 12.13
 # Fraction of the integration window past which the integrands drop off to prevent artifacts
xv,yv,zv = [np.array([0]) for i in range(3)]

lawvals = []

responsebig = []





