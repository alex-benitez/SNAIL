import numpy as np
from integration_tools import lewenstein
# from slowenstein import slowenstein as lewenstein
# from parallellewenstein import parallel_lewenstein as lewenstein

class config:
    """
    This is the class that stores all of the information on laser and target properties, as well as calculation parameters. By default has parallelization enabled,
    however, if you want to use a single core version which only requires numpy, set config.parallel to False.
    """
    parallel = True
    pass

def generate_t(config):
    """
    A function to generate the time axis in femtoseconds given the specific pulse information.
    """
    l = config.pulse_duration*(1+2*config.padding)
    return np.arange(-l/2,l/2,step=config.timestep) 
    

def get_omega_axis(t, config):
    dt = t[1] - t[0]
    domega = 2*np.pi/dt/len(t)
    temp = np.arange(0,len(t))
    temp[int(len(temp)/2)-1:] = temp[int(len(temp)/2)-1:] - len(temp)
    omega = temp*domega
    
    return omega


def au_convert(value,quantity,target):
    """
    Converts between standard SI units and atomic units or time to femtoseconds, currently supports:
        e - Electric field   i - intensity (W/cm^2)  u - Energy   s - Length   a - Area   
        vol - Volume   t - time   v - velocity
    
    Args:
        value (float): The quantity to convert.
        quantity: The physical parameter that is being converted, e.g. e for electric field.
        target (string): 'si' to convert to standard units, 'au' to convert to atomic units, 'fs' to convert au time to fs time (only works with time).
        
    Returns:
        
        float: Value converted to the corresponding unit system.
    
    """
    c = 299792458;
    hbar = 1.054571726e-34
    eq = 1.602176565e-19 # electron charge
    a0 = 5.2917721092e-11 # Bohr radius
    Ry = 13.60569253*eq # Rydberg unit of energy
    t_unit_SI = 2.418884326509e-17
    e0 = 8.8541878188e-12

    omega_unit_SI = 1/t_unit_SI
    U_unit_SI = hbar * omega_unit_SI 
    q_unit_SI = eq
    s_unit_SI = a0 * np.sqrt(2*Ry/U_unit_SI)
    	
    E_unit_SI = U_unit_SI / q_unit_SI / s_unit_SI
    I_unit_SI = 0.5*c*e0*E_unit_SI**2
    
    factors = {'e':E_unit_SI,'i':I_unit_SI,'u':U_unit_SI,'s':s_unit_SI,'a':s_unit_SI**2,'vol':s_unit_SI**3,'t':t_unit_SI,'v':s_unit_SI/t_unit_SI}

    	
    target = target.lower()
    quantity = quantity.lower()
    
    if target == 'si':
        return value*factors[quantity]
    
    elif target == 'au':
        return value/factors[quantity]
    
    elif target == 'fs':
        if quantity != 't':
            raise ValueError("Can only convert time to fs")
        else:
            return value * 0.02418884 
    
    else:
        raise ValueError("Invalid quantity or target")

def generate_pulse(config):
    
    '''
    Generates the driving pulse, forces the user to set a pulse type,
    currently supports pulse types:
        
        Constant       - A constant envelope\n
        Gaussian       - Gaussian beam with no cutoff\n
        Super Gaussian - Gaussian with a faster decline\n
        Cos Squared    - Cos squared envelope\n
        Sin 6          - A very flat sin 6 pulse for sharper harmonics\n 
    
    Args:
        config (class):
            - Calculation_cycles
            - Points per cycle
            - Pulse duration
            - Pulse shape
    
    Returns:
        driving_field (array): The electric field amplitude over time, same size as t
    '''
    
    t = generate_t(config)
    # pult = au_convert(config.pulse_duration*1e-15, 't', 'au')
    pult = config.pulse_duration

    

    pulse_list = ['constant','gaussian','super_gaussian','cos_sqr','sin_6']
    
    if not hasattr(config,'pulse_shape') or (config.pulse_shape.lower() not in pulse_list):
        raise ValueError('You must specify a pulse shape from the following: {}'.format(pulse_list))
        
    if config.pulse_shape == 'constant':
        envelope = 1
        
    elif config.pulse_shape.lower() == 'gaussian':
        tau = pult / 2 / np.sqrt(np.log(np.sqrt(2)))
        envelope = np.exp(-(t / tau) ** 2)
        
    elif config.pulse_shape.lower() == 'super-gaussian':
        tau = pult / 2 / np.sqrt(np.sqrt(np.log(np.sqrt(2))))
        envelope = np.exp(-(t / tau) ** 4)
        
    elif config.pulse_shape.lower() == 'cos_sqr':
        tau = pult / 2 / np.arccos(1 / np.sqrt(np.sqrt(2)))
        envelope = np.cos(t / tau) ** 2
        envelope[t / tau <= -np.pi / 2] = 0
        envelope[t / tau >= np.pi / 2] = 0
        
    elif config.pulse_shape.lower() == 'sin_6': 
        t = t - t[0] 
        tau = pult / 2 / np.arccos(1 / np.sqrt(np.sqrt(2)))
        envelope = 1-np.sin(np.pi/2 +0.5*t / tau) ** 6
        # envelope[np.pi/2 +0.5*t / tau <= 0 ] = 0
        # envelope[np.pi/2 +0.5*t / tau >= 2*np.pi ] = 0
    
    else:
        raise ValueError('Wrong type of carrier, use one of the following: {}'.format(pulse_list))
        
    if not hasattr(config, 'carrier') or config.carrier.lower() == 'cos':
           carrier = np.cos
    elif config.carrier.lower() == 'exp':
        carrier = lambda x: np.exp(1j * x)
    else:
        raise ValueError("Invalid carrier: must be 'cos' or 'exp'")
    # print(envelope)
    amplitude = envelope*carrier(t*2*137*np.pi/config.wavelength)
    print(au_convert(config.peak_intensity, 'i', 'au'))
    E0 = np.sqrt(config.peak_intensity)
    print(E0)
    driving_field = amplitude*E0
    return np.squeeze(driving_field)


def dipole_response(points,driving_field,config,t=np.array([])):
    if t.size == 0:
        t = generate_t(config)
        # print(t)
    pi = np.pi
    
    '''
    To avoid integration artifacts, a soft integration window is applied through
    the weights which multiply each integrand later, how it works is the weights
    are equal to 1 [0 -> tau_window_pts] and then drop off as cos^2

    '''
    if not hasattr(config, 'tau_window_length'):
        config.tau_window_length = 1
        print('Tau window length was not defined, setting to 1.0 as default')
    if not hasattr(config, 'tau_dropoff_pts'):
        config.tau_dropoff_pts = 0.2
        print('Tau dropoff length was not defined, setting to 0.2 as default')
    
        
    tau_window_pts    = int(config.ppcycle*config.tau_window_length) # The number of cycles to integrate over (can be less than one)
    tau_dropoff_pts  = int(config.tau_dropoff_pts*tau_window_pts) # The range of the soft window
    tau_window_pts   -= tau_dropoff_pts
    
    
    
    weights = np.ones((1, tau_dropoff_pts + tau_window_pts))[0]
    if tau_dropoff_pts > 1:
      
        dropoff_factor = pi/2 / (tau_dropoff_pts-1)
      
    else:  # avoid division by zero
        dropoff_factor = 0.5
    if tau_dropoff_pts < 2:
        print('The value for tau_interval_length is too small, setting points integrated over to 2')
    dropoff = np.cos(dropoff_factor*np.arange(0,max(tau_dropoff_pts,2)))**2  
    weights[tau_window_pts:] = weights[tau_window_pts:] * dropoff
    
    config.weights = weights
    Ip = au_convert(config.ionization_potential*1.602176565e-19, 'u', 'au')
    config.Ip = Ip
    config.alpha = 2*Ip
    
    wstart = t.size - 5*config.ppcycle
    t_window = np.cos(0.5 * np.arange(t.size-wstart)) ** 2
    wind = np.sin((np.pi*t)/t[-1])**2

    omega = get_omega_axis(t,config)
    final = np.array([])
    
    if config.parallel == True:
        from parallellewenstein import parallel_lewenstein as lewenstein
    elif config.parallel == False:
        from integration_tools import lewenstein 
        
    
    # TODO: Implement a generator that uses numpy arrays for 3D spaces in the future, the current implementation is quite archaic
    '''
    https://stackoverflow.com/questions/44854593/any-object-that-exists-with-memory-usage-like-a-generator-but-can-return-a-numpy
    '''
    
    for point in points:
        xi,yi,zi = point
        # t = au_convert(t,'t','SI',config)
        d_t = lewenstein(t,driving_field,config)#*t_window
        np.save('/home/alex/Desktop/Python/SNAIL/Benflattop/responsestore/response.npy',d_t)
        np.save('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/single.npy',d_t)
        np.save('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/time.npy',t)
        
        d_t = d_t*wind

        

        d_omega = np.conj(np.fft.fft(d_t)) 
        d_omega = d_omega*np.exp(-1j*omega*t[0])*(t[1]-t[0])
        omega = omega[:d_omega.size]

        
        
        
        final = np.append(final,d_omega)

    
                
                
    return [omega,final]
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    