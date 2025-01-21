'''
There is so much technical debt in this project that a complete rewrite is better,
lets see how long it takes, starting at 10:48 Jan 21 2025
'''
import numpy as np
import lewenstein

lewenstein = lewenstein.lewenstein

def generate_pulse(pulsefig):
    
    Npoints = pulsefig.ppcycle*pulsefig.cycles
    times = pulsefig.wavelength*pulsefig.cycles/(3*(10**8))
    t = np.linspace(-times/2,times/2,Npoints)
    pulse_list = ['constant','gaussian','super_gaussian','cos_sqr','sin_sqr']
    
    if not hasattr(pulsefig,'pulse_shape') or (pulsefig.pulse_shape.lower() not in pulse_list):
        raise ValueError('You must specify a pulse shape from the following: {}'.format(pulse_list))
    
    