# Generate test datasets for Anne
import numpy as np
import matplotlib.pyplot as plt
import General_tools
from numpy import pi


config = General_tools.config()
sau =  General_tools.sau_convert

for i in range(35):
    
    config.calculation_cycles = 30 + i
    config.ppcycle = 200
    config.wavelength = 1e-3 
    config.peak_intensity = 1e14 + (i*3e14)/35
    config.pulse_shape = 'cos_sqr'
    config.pulse_duration = 35 + i
    config.ionization_potential = 12.13
    config.tau_window_length = 1.1# How far back over excursion time to integrate over, as a fraction of a cycle
    config.tau_dropoff_pts = 0.4 # Fraction of the integration window past which the integrands drop off to prevent artifacts
    xv,yv,zv = [np.array([0]) for i in range(3)]
    

    valrang = [0,100]
    lawvals = []
    hbar = 1.05457181e-34/(1.6*1e-19)
    c = 3e8
    
    w = c*2*pi/(config.wavelength*1e-3)
    responsebig = []
    
    
        
    
    [t, driving_field] = General_tools.generate_pulse(config)
    np.savetxt('/home/alex/Desktop/Python/SNAIL/AnneData/cos_cycles{}Inten{}'.format(config.calculation_cycles,round(config.peak_intensity/1e14,2)),driving_field[0])

    [omega1,response1] = General_tools.dipole_response(t,[[0,0,0]],driving_field,config)
    omega1 = omega1[np.where(omega1>valrang[0])]
    response1 = response1[np.where(omega1>valrang[0])]
    response1 = np.log(np.abs(response1[np.where(omega1<valrang[1])])**2)
    response1 = response1[np.where(omega1<valrang[1])]

    fig,axs = plt.subplots(2,1,figsize=(7,4))

    axs[0].plot(omega1[np.where(omega1<valrang[1])],response1)
    axs[0].set_title('Harmonic Response of a Cos Pulse')
    axs[0].set_xlabel('Harmonic Order')
    axs[0].set_ylabel('Int. (Arb. log Scale)')

    # axs[0].vlines(int(cutoff),min(response1),max(response1),'k',linewidth=0.5)

    # axs[0].set_xlabel('High Harmonic Order')`
    t_fs = General_tools.sau_convert(t,'t','SI',config)/1e-15
    axs[1].plot(t_fs,driving_field[0])
    axs[1].set_xlabel('Time(fs)')
    axs[1].set_ylabel('Int. (Arb. Scale)')

    plt.tight_layout()
    plt.savefig('/home/alex/Desktop/Python/SNAIL/AnneData/cos_cycles{}Inten{}.png'.format(config.calculation_cycles,round(config.peak_intensity/1e14,2)),dpi=100)
    plt.show()

    plt.clf()
    plt.close('all')

for i in range(35):
    
    config.calculation_cycles = 30 + i
    config.ppcycle = 200
    config.wavelength = 1e-3 
    config.peak_intensity = 1e14 + (i*1e14)/35
    config.pulse_shape = 'gaussian'
    config.pulse_duration = 35 + i
    config.ionization_potential = 12.13
    config.tau_window_length = 1.1# How far back over excursion time to integrate over, as a fraction of a cycle
    config.tau_dropoff_pts = 0.4 # Fraction of the integration window past which the integrands drop off to prevent artifacts
    xv,yv,zv = [np.array([0]) for i in range(3)]
    

    valrang = [0,100]
    lawvals = []
    hbar = 1.05457181e-34/(1.6*1e-19)
    c = 3e8
    
    w = c*2*pi/(config.wavelength*1e-3)
    responsebig = []
    
    
        
    
    [t, driving_field] = General_tools.generate_pulse(config)
    np.savetxt('/home/alex/Desktop/Python/SNAIL/AnneData/gaussian_cycles{}Inten{}'.format(config.calculation_cycles,round(config.peak_intensity/1e14,2)),driving_field[0])

    [omega1,response1] = General_tools.dipole_response(t,[[0,0,0]],driving_field,config)
    omega1 = omega1[np.where(omega1>valrang[0])]
    response1 = response1[np.where(omega1>valrang[0])]
    response1 = np.log(np.abs(response1[np.where(omega1<valrang[1])])**2)
    response1 = response1[np.where(omega1<valrang[1])]

    fig,axs = plt.subplots(2,1,figsize=(7,4))

    axs[0].plot(omega1[np.where(omega1<valrang[1])],response1)
    axs[0].set_title('Harmonic Response of a Gaussian Pulse')
    axs[0].set_xlabel('Harmonic Order')
    axs[0].set_ylabel('Int. (Arb. log Scale)')

    # axs[0].vlines(int(cutoff),min(response1),max(response1),'k',linewidth=0.5)

    # axs[0].set_xlabel('High Harmonic Order')`
    t_fs = General_tools.sau_convert(t,'t','SI',config)/1e-15
    axs[1].plot(t_fs,driving_field[0])
    axs[1].set_xlabel('Time(fs)')
    axs[1].set_ylabel('Int. (Arb. Scale)')

    plt.tight_layout()
    plt.savefig('/home/alex/Desktop/Python/SNAIL/AnneData/gaussian_cycles{}Inten{}.png'.format(config.calculation_cycles,round(config.peak_intensity/1e14,2)),dpi=100)
    plt.show()

    plt.clf()
    plt.close('all')
