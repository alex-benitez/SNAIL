'''
This is to check at what point the new lewenstein diverges from HHGMax and benchmark speed
'''
import numpy as np
# import General_tools
# from General_tools import sau_convert
# import matplotlib.pyplot as plt

# import time
# config = General_tools.config()
# from scipy.interpolate import interp1d
# from scipy.integrate import cumtrapz
# config.calculation_cycles = 50
# config.ppcycle = 200
# config.wavelength = 1e-3
# config.peak_intensity = 1e14
# config.pulse_shape = 'cos_sqr'
# config.pulse_duration = 40
# config.ionization_potential = 12.13
# config.tau_window_length = 0.65# How far back over excursion time to integrate over, as a fraction of a cycle
# config.tau_dropoff_pts = 0.4 # Fraction of the integration window past which the integrands drop off to prevent artifacts
# Ip = sau_convert(config.ionization_potential*1.602176565e-19, 'u', 'sau', config)
# config.Ip = Ip
# config.alpha = 2*Ip
# tau_window_pts    = int(config.ppcycle*config.tau_window_length) # The number of cycles to integrate over (can be less than one)
# tau_dropoff_pts  = int(config.tau_dropoff_pts*tau_window_pts) # The range of the soft window
# tau_window_pts   -= tau_dropoff_pts
# config.weights = np.ones((1, tau_dropoff_pts + tau_window_pts))[0]
# weights = config.weights
# t = 2*np.pi*np.arange(config.calculation_cycles,step=1/config.ppcycle)+1/config.ppcycle
# # t = np.load('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/t.npy')
# driving_field = np.load('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/driving_field.npy')


def slowenstein(t,Et,config,at=None,epsilon_t=1e-4):
    N = t.size
    if at is None: at = np.ones_like(t)
    SQR = np.square
    Et = Et[0]

    alpha = config.alpha
    Ip = config.Ip
    prefactor = (2**3.5) * (alpha**1.25) / np.pi 
    
    def dp(p):
        return 1j*prefactor*p/((np.square(p) + alpha)**3)
       
    dt = (-np.roll(t,1) + t)*0.5
    dt[0] = 0
    
    At = -(np.roll(Et,1) + Et)*dt 
    

    At[0] = 0
    At = np.cumsum(At)


    Bt = (np.roll(At,1) + At)*dt
    Bt[0] = 0
    Bt = np.cumsum(Bt)

    Ct = (np.square(np.roll(At,1)) + np.square(At))*dt
    Ct[0] = 0
    Ct = np.cumsum(Ct)
    weights = config.weights
    ws = weights.size

    bigAt = np.reshape(np.tile(At,ws),(ws,At.size))
    temptAt = bigAt[np.c_[:bigAt.shape[0]], (np.r_[:bigAt.shape[1]] - np.c_[:ws]) % bigAt.shape[1]]

    bigCt = np.reshape(np.tile(Ct,ws),(ws,Ct.size))
    temptCt = bigCt[np.c_[:bigCt.shape[0]], (np.r_[:bigCt.shape[1]] - np.c_[:ws]) % bigCt.shape[1]]
    
    c = (np.pi/(epsilon_t + 0.5*1j*t[:ws]))**1.5
    
    bigBt = Bt*np.c_[np.ones(ws)]
    temptBt = bigBt[np.c_[:bigBt.shape[0]], (np.r_[:bigBt.shape[1]] - np.c_[:ws]) % bigBt.shape[1]]
    
    pst = (bigBt - temptBt)/np.c_[t[:ws]]
    pst[0] = At
    correction = np.r_[:pst.shape[1]] > np.c_[:pst.shape[0]]
    
    pst = pst*correction
   
    print('over here')
    argdstar = pst - bigAt
    argdstar = argdstar*correction
    argdnorm = pst - temptAt
    argdnorm = argdnorm*(correction)

    dstar = np.conjugate(dp(argdstar))
    dnorm = 1j*prefactor*argdnorm/((np.square(argdnorm) + alpha)**3)
    
    dnorm = dnorm*correction
    dstar = dstar*correction
     

    Sst = -(0.5/np.c_[t[:ws]])*SQR(bigBt - temptBt) + 0.5*(bigCt-temptCt) + Ip*np.c_[t[:ws]]
    
    Sst[0] = Sst[0]*0
    Sst = Sst*correction
    

    
    bigEt = np.reshape(np.tile(Et,ws),(ws,Et.size))
    temptEt = bigEt[np.c_[:bigEt.shape[0]], (np.r_[:bigEt.shape[1]] - np.c_[:ws]) % bigEt.shape[1]]
    
    bigat = np.reshape(np.tile(at,ws),(ws,at.size))
    temptat = bigat[np.c_[:bigat.shape[0]], (np.r_[:bigat.shape[1]] - np.c_[:ws]) % bigat.shape[1]]
    
    integral = dstar*dnorm*np.exp(-1j*Sst)
    
    columnc = np.c_[c] + np.c_[c]*0j
    imagEt = temptEt + temptEt*0j
    imaweights = np.c_[weights] + 0j*np.c_[weights]
    imat = bigat + 0j*bigat
    imatat = temptat + 0j*temptat
    
    integral *= imagEt*columnc*imaweights*imat*imatat
    # integral *= temptEt*(np.c_[weights.astype(np.complex128)])*(np.c_[c.astype(np.complex128)])*(bigat)*temptat

    timeinterval  = np.array([np.ones(N,dtype='complex128')*(t[i] - t[i-1]) for i in range(ws)])
    integral = integral*timeinterval
    integral[0] = integral[0]*0
     
    output = np.cumsum(integral,0)[-1]

    output = 2*np.imag(output)
    np.save('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/output',output)

    # print(output)
    # print(output[4000:4050])
    # print(foutput[4000:4050])
    # fast = time.time()-start
    # temp = abs(output-foutput)
    # etemp = np.where(temp>1e-10)
    # print(etemp[0].size)
    # print(output-foutput)
    # np.save('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/output',output)
    # print('The difference is slow: {} , fast: {}'.format(slow,fast))
    return output

    



# matdata = open('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/Electric.txt')
# matlabarray = []
# for line in matdata.readlines():
#     matlabarray.append(line.split(','))
# matlabarray = matlabarray[0]
# matlabarray[-1] = matlabarray[-1][:-2]
# matlabarray = [float(i) for i in matlabarray]
# slowenstein(t, np.array([matlabarray[:-1]]), config)
# alpha = config.alpha
# Ip = config.Ip
# prefactor = (2**3.5) * (alpha**1.25) / np.pi 
# def dp(p):
#     return 1j*prefactor*p/((np.square(p) + alpha)**3)
# output = slowenstein(t,driving_field, config)
# plt.plot(output)
# output = np.load('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/output.npy')
# foutput = lewenstein(t.size,t,np.array([matlabarray[:-1]]),weights.size,weights,np.ones_like(t),Ip,1e-4,dp,1)
# plt.plot(output)
# plt.savefig('/home/alex/Desktop/Python/SNAIL/Latex/coolgraph.png')
# plt.plot(driving_field[0])
# plt.plot(matlabarray)





