'''
There is so much technical debt in this project that a complete rewrite is better,
lets see how long it takes, starting at 10:48 Jan 21 2025
'''
import numpy as np
import lewenstein
import warnings

lewenstein = lewenstein.lewenstein
class lconfig:
    pass

def sau_convert(value,quantity,target,pulsefig):
    '''
    Converts between standard SI units and atomic units, currently supports:
        e - Electric field
        u - Energy
        s - Length
        a - Area
        vol - Volume
        v - velocity
    '''
    c = 299792458;
    hbar = 1.054571726e-34;
    eq = 1.602176565e-19; # electron charge
    a0 = 5.2917721092e-11; # Bohr radius
    Ry = 13.60569253*eq; # Rydberg unit of energy
    
    # scaled atomic unit quantities expressed in SI units
    t_unit_SI = (pulsefig.wavelength*1e-3) / c / (2*np.pi);
    omega_unit_SI = 1/t_unit_SI
    
    U_unit_SI = hbar * omega_unit_SI 
    q_unit_SI = eq
    s_unit_SI = a0 * np.sqrt(2*Ry/U_unit_SI)
    	
    E_unit_SI = U_unit_SI / q_unit_SI / s_unit_SI
    
    factors = {'e':E_unit_SI,'u':U_unit_SI,'s':s_unit_SI,'a':s_unit_SI**2,'vol':s_unit_SI**3,'t':t_unit_SI,'v':s_unit_SI/t_unit_SI}

    	

    target = target.lower()
    quantity = quantity.lower()
    
    if target == 'si':
        return value*factors[quantity]
    
    elif target == 'sau':
        return value/factors[quantity]
    
    else:
        raise ValueError("Invalid quantity or target")

def generate_pulse(pulsefig):
    
    '''
    Generates the driving pulse, forces the user to set a pulse type,
    currently supports pulse types:
        
        Constant - Self explanatory
        Gaussian - Gaussian beam with no cutoff
        Super Gaussian - Gaussian with a faster decline
        Cos Squared - 
    '''
    
    Npoints = pulsefig.ppcycle*pulsefig.cycles
    times = pulsefig.wavelength*pulsefig.cycles/(3*(10**8))
    # t = np.linspace(-times/2,times/2,Npoints)
    # t = sau_convert(t, 't', 'sau', pulsefig)
    t = 4*np.pi*np.arange(-40,40,0.005)
    pult = sau_convert(pulsefig.pulse_duration*1e-15, 't', 'sau', pulsefig)

    pulse_list = ['constant','gaussian','super_gaussian','cos_sqr','sin_sqr']
    
    if not hasattr(pulsefig,'pulse_shape') or (pulsefig.pulse_shape.lower() not in pulse_list):
        raise ValueError('You must specify a pulse shape from the following: {}'.format(pulse_list))
        
    if pulsefig.pulse_shape == 'constant':
        envelope = 1
        
    elif pulsefig.pulse_shape.lower() == 'gaussian':
        tau = pult / 2 / np.sqrt(np.log(np.sqrt(2)))
        envelope = np.exp(-(t / tau) ** 2)
        
    elif pulsefig.pulse_shape.lower() == 'super-gaussian':
        tau = pult / 2 / np.sqrt(np.sqrt(np.log(np.sqrt(2))))
        envelope = np.exp(-(t / tau) ** 4)
        
    elif pulsefig.pulse_shape.lower() == 'cos_sqr':
        tau = pult / 2 / np.arccos(1 / np.sqrt(np.sqrt(2)))
        envelope = np.cos(t / tau) ** 2
        envelope[t / tau <= -np.pi / 2] = 0
        envelope[t / tau >= np.pi / 2] = 0
        
    elif pulsefig.pulse_shape.lower() == 'sin_sqr': 
        tau = pult / 2 / np.arccos(1 / np.sqrt(np.sqrt(2)))
        envelope = 1-np.cos(np.pi/2 +0.5*t / tau) ** 6
        envelope[t / tau <= -np.pi ] = 0
        envelope[t / tau >= np.pi ] = 0
    
    else:
        raise ValueError('Wrong type of carrier, use one of the following: {}'.format(pulse_list))
        
    if not hasattr(pulsefig, 'carrier') or pulsefig.carrier.lower() == 'cos':
           carrier = np.cos
    elif pulsefig.carrier.lower() == 'exp':
        carrier = lambda x: np.exp(1j * x)
    else:
        raise ValueError("Invalid carrier: must be 'cos' or 'exp'")
    print(envelope)
    amplitude = np.array([envelope*carrier(t)])

    # Setup frequency axis
    domega = 2 * np.pi / (t[1] - t[0]) / len(t)
    temp = np.arange(len(t))
    temp[temp >= np.ceil(len(temp) / 2)] -= len(temp)
    omega = temp * domega
    
    # Fourier transform
    coefficients = np.conj(np.fft.fft(np.conj(amplitude), axis=1))
    
    return [t,omega,coefficients]



def get_omega_axis(t, lconfig):

    dt = t[1] - t[0]
    domega = 2*np.pi/dt/len(t)
    temp = np.arange(0,len(t))
    temp[int(len(temp)/2)-1:] = temp[int(len(temp)/2)-1:] - len(temp)
    omega = temp*domega
    
    return omega



def plane_wave_driving_field(x,y,z,lconfig):
    omega = lconfig.omega
    pulse_coefficients = lconfig.pulse_coefficients
    E0_SI = np.sqrt(2*lconfig.peak_intensity*10000/299792458/8.854187817e-12)
    E0 = sau_convert(E0_SI, 'E', 'SAU', lconfig)
    return E0 * np.fft.ifft(np.conjugate(pulse_coefficients),  axis=1)



def dipole_response(t,points,lconfig):
    dt = abs(t[1] - t[0])
    pi = np.pi
    
    '''
    To avoid integration artifacts, a soft integration window is applied through
    the weights which multiply each integrand later, how it works is the weights
    are equal to 1 [0 -> tau_window_pts] and then drop off as cos^2
    '''
    if not hasattr(lconfig, 'tau_window_length'):
        lconfig.tau_window_length = 0.5
    if not hasattr(lconfig, 'tau_dropoff_pts'):
        lconfig.tau_dropoff_pts = 0.1
    
        
    tau_window_pts    = int(lconfig.ppcycle*lconfig.tau_window_length) # The number of cycles to integrate over (can be less than one)
    tau_dropoff_pts  = int(lconfig.tau_dropoff_pts*tau_window_pts) # The range of the soft window
    tau_window_pts   -= tau_dropoff_pts
    
    
    
    weights = np.ones((1, tau_dropoff_pts + tau_window_pts))[0]
    
    if tau_dropoff_pts > 1:
      
        dropoff_factor = pi/2 / (tau_dropoff_pts-1)
      
    else:  # avoid division by zero
        dropoff_factor = 0.5
    
    dropoff = np.cos(dropoff_factor*np.arange(0,max(tau_dropoff_pts,2)))**2  
    weights[tau_window_pts:] = weights[tau_window_pts:] * dropoff
    
    lconfig.weights = weights
    Ip = sau_convert(lconfig.ionization_potential*1.602176565e-19, 'u', 'sau', lconfig)
    lconfig.Ip = Ip
    lconfig.alpha = 2*Ip
    
    t_window = np.cos(0.5 * np.arange(len(t))) ** 2

    omega = get_omega_axis(t,lconfig)
    final = np.array([])
 
    for point in points:
        xi,yi,zi = point
        Et_cmc = np.real(plane_wave_driving_field(xi,yi,zi,lconfig))
        d_t = lewenstein(t,Et_cmc,lconfig)*t_window
        # d_t(:,win_start:win_end) = d_t(:,win_start:win_end) .* repmat(t_window,components,1);
        
        
        d_omega = np.conj(np.fft.fft(d_t))
        
        omega = omega[:d_omega.size]
        d_omega = d_omega*np.exp(-1j*omega*t[0])*dt
        
        
        
        final = np.append(final,d_omega)
                # final = d_omega
    response_cmc = final   
                
                
    return [omega,response_cmc]
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    