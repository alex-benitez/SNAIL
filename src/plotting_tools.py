import numpy as np
import matplotlib.pyplot as plt
import general_tools as gt

'''
What would a fast integration library be without a fast plotting module!

This is a collection of functions to quickly call and view your results,
hopefully quick and intuitive to use!
'''


def driving(field, config, t=np.array([]), timeaxis='fs',save_location='./src/stored_data/driving_field.png',grid=True):
    '''
    A function to quickly plot your driving field
    '''
    if t.size == 0:
        t = gt.generate_t(config)
    
    timeaxis = timeaxis.lower()
    if timeaxis != 'au' and timeaxis != 'fs':
        raise ValueError("Time axis has to be in fs or au")
    if timeaxis == 'fs':
        t = gt.sau_convert(t,'t','SI',config)/1e-15
        
    fig, axs = plt.subplots(1,1,figsize=(4,3))
    axs.plot(t,field,'r')
    axs.set_title('Laser Pulse')
    axs.set_xlabel('Time({})'.format(timeaxis))
    axs.set_ylabel('Intensity (Arbitrary Scale)')
    axs.grid(grid)
    plt.tight_layout()
    
    
    try:
        plt.savefig(save_location,dpi=300)
    except:
        print('Not saving due to invalid save location')
        
    

           
def driving_harmonics(field,response, config, t=np.array([]), timeaxis='fs',save_location='./src/stored_data/driving_response.png',grid=True,harmonic_range=[0,100]):
    '''
    A function that plots the driving field and the harmonic spectrum by taking the logarithm
    of the square of the response.
    '''
    if t.size == 0:
        t = gt.generate_t(config)
        
        
    fig,axs = plt.subplots(1,2,figsize=(8,3))
    
    omega = gt.get_omega_axis(t,config)
    response = response[(omega>harmonic_range[0]) & (omega<harmonic_range[1])]
    omega    =    omega[(omega<harmonic_range[1]) & (omega>harmonic_range[0])]
    response = np.log(np.abs(response)**2)
    
    timeaxis = timeaxis.lower()
    if timeaxis != 'au' and timeaxis != 'fs':
        raise ValueError("Time axis has to be in fs or sau")

    if timeaxis == 'fs':
        t = gt.sau_convert(t,'t','SI',config)/1e-15
        
    axs[0].plot(t,field,'r')
    axs[0].set_title('Laser Pulse')
    axs[0].set_xlabel('Time({})'.format(timeaxis))
    axs[0].set_ylabel('Intensity (Arbitrary Scale)')
    
    axs[1].plot(omega,response,'c')
    axs[1].set_title('Harmonic Response')
    axs[1].set_xlabel('Harmonic Order')
    axs[1].set_ylabel('Logarithmic Intensity(Arbitrary Scale)')
    
    axs[0].grid(grid)
    axs[1].grid(grid)
    plt.tight_layout()
    try:
        plt.savefig(save_location,dpi=300)
    except:
        print('Not saving due to invalid save location')


