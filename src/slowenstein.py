'''
This is to check at what point the new lewenstein diverges from HHGMax and benchmark speed
'''
import numpy as np
import General_tools
from General_tools import sau_convert
import matplotlib.pyplot as plt

import time
config = General_tools.config()
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
config.calculation_cycles = 50
config.ppcycle = 200
config.wavelength = 1e-3
config.peak_intensity = 1e14
config.pulse_shape = 'cos_sqr'
config.pulse_duration = 40
config.ionization_potential = 12.13
config.tau_window_length = 0.65# How far back over excursion time to integrate over, as a fraction of a cycle
config.tau_dropoff_pts = 0.4 # Fraction of the integration window past which the integrands drop off to prevent artifacts
Ip = sau_convert(config.ionization_potential*1.602176565e-19, 'u', 'sau', config)
config.Ip = Ip
config.alpha = 2*Ip
tau_window_pts    = int(config.ppcycle*config.tau_window_length) # The number of cycles to integrate over (can be less than one)
tau_dropoff_pts  = int(config.tau_dropoff_pts*tau_window_pts) # The range of the soft window
tau_window_pts   -= tau_dropoff_pts
config.weights = np.ones((1, tau_dropoff_pts + tau_window_pts))[0]
weights = config.weights
t = np.load('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/t.npy')
driving_field = np.load('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/driving_field.npy')


def slowenstein(t,Et,config,at=None,epsilon_t=1e-4):
    N = t.size
    if at is None: at = np.ones_like(t)
    SQR = np.square
    # Et = Et[0]

    alpha = config.alpha
    Ip = config.Ip
    prefactor = (2**3.5) * (alpha**1.25) / np.pi 
    
    def dp(p):
        return 1j*prefactor*p/((np.square(p) + alpha)**3)
       
    # dt = (-np.roll(t,1) + t)*0.5
    # dt[0] = 0
    
    # At = -(np.roll(Et,1) + Et)*dt 
    

    # At[0] = 0
    # At = np.cumsum(At)


    # Bt = (np.roll(At,1) + At)*dt
    # Bt[0] = 0
    # Bt = np.cumsum(Bt)

    # Ct = (np.square(np.roll(At,1)) + np.square(At))*dt
    # Ct[0] = 0
    # Ct = np.cumsum(Ct)
    
    
    # np.save('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/output',At)
    weights = config.weights
    ws = weights.size
    fAt = np.zeros(t.size)
    fBt = np.zeros(t.size)
    fCt = np.zeros(t.size)
    # fpst = np.zeros((ws,N))
    # fargdstar = np.zeros((ws,N))
    # fargdnorm = np.zeros((ws,N))
    # fdstar = np.zeros((ws,N),dtype='complex')
    # fdnorm = np.zeros((ws,N),dtype='complex')
    # fSst = np.zeros((ws,N))
    # fintegral = np.zeros((ws,N),dtype='complex')
    # start = time.time()
    foutput = np.zeros(t.size)
    IAt = 0
    IBt = 0
    ICt = 0
    for i in range(1,N):
        dt = t[i]-t[i-1]
        IAt -= (Et[i-1]+Et[i])*dt/2
        fAt[i] = IAt
        IBt += (fAt[i-1]+fAt[i])*dt/2
        fBt[i] = IBt
        ICt += (np.square(fAt[i-1]) + np.square(fAt[i]))*dt/2
        fCt[i] = ICt

    for i in range(1,N):
        inde = ws
        if i<inde:
            inde = i+1
        integral = 0
        last_integrand = 0
        for tau in range(inde):
            fpst = (fBt[i]-fBt[i-tau])/t[tau]
            if tau == 0:
                fpst = fAt[i]
                
            fargdstar = fpst - fAt[i]
            fargdnorm = fpst - fAt[i-tau]
            fdnorm = dp(fargdnorm)
            fdstar = np.conj(dp(fargdstar))
            c = (np.pi/(epsilon_t + 0.5*1j*t[tau]))**1.5
            
            fSst = Ip*t[tau] - 0.5/t[tau]*SQR(fBt[i]-fBt[i-tau]) + 0.5*(fCt[i]-fCt[i-tau])
            if tau == 0:
                fSst = 0
            integrand13 = fdstar
            integrand13 *= (fdnorm*Et[i-tau])*(c**1.5)*(np.cos(fSst) + 1j*np.sin(fSst))*weights[tau]*at[i]*at[i-tau]
            dt = t[tau] - t[tau-1]
            if tau == 0:
                dt = 0
            integral += (last_integrand + integrand13)*dt/2
            last_integrand = integrand13
        # print(integral)
        foutput[i] = 2*np.imag(integral)   

                
    # slow = time.time()-start


    # bigAt = np.reshape(np.tile(At,ws),(ws,At.size))
    # temptAt = bigAt[np.c_[:bigAt.shape[0]], (np.r_[:bigAt.shape[1]] - np.c_[:ws]) % bigAt.shape[1]]

    # bigCt = np.reshape(np.tile(Ct,ws),(ws,Ct.size))
    # temptCt = bigCt[np.c_[:bigCt.shape[0]], (np.r_[:bigCt.shape[1]] - np.c_[:ws]) % bigCt.shape[1]]
    
    # c = (np.pi/(epsilon_t + 0.5*1j*t[:ws]))**1.5
    
    # bigBt = Bt*np.c_[np.ones(ws)]
    # temptBt = bigBt[np.c_[:bigBt.shape[0]], (np.r_[:bigBt.shape[1]] - np.c_[:ws]) % bigBt.shape[1]]
    
    # pst = (bigBt - temptBt)/np.c_[t[:ws]]
    # pst[0] = At
    # correction = np.r_[:pst.shape[1]] > np.c_[:pst.shape[0]]
    
    # pst = pst*correction
   
    # start = time.time()
    # argdstar = pst - bigAt
    # argdstar = argdstar*correction
    # argdnorm = pst - temptAt
    # argdnorm = argdnorm*(correction)

    # dstar = np.conjugate(dp(argdstar))
    # dnorm = 1j*prefactor*argdnorm/((np.square(argdnorm) + alpha)**3)

    # dnorm = dnorm*correction
    # dstar = dstar*correction
        

    # Sst = -(0.5/np.c_[t[:ws]])*SQR(bigBt - temptBt) + 0.5*(bigCt-temptCt) + Ip*np.c_[t[:ws]]
    
    # Sst[0] = Sst[0]*0
    # Sst = Sst*correction
    

    
    # bigEt = np.reshape(np.tile(Et,ws),(ws,Et.size))
    # temptEt = bigEt[np.c_[:bigEt.shape[0]], (np.r_[:bigEt.shape[1]] - np.c_[:ws]) % bigEt.shape[1]]
    
    # bigat = np.reshape(np.tile(at,ws),(ws,at.size))
    # temptat = bigat[np.c_[:bigat.shape[0]], (np.r_[:bigat.shape[1]] - np.c_[:ws]) % bigat.shape[1]]
    
    # integral = dstar*dnorm*np.exp(-1j*Sst)*temptEt*(np.c_[weights.astype(np.complex128)])*(np.c_[c.astype(np.complex128)])#*(bigat)*temptat

    # timeinterval  = np.array([np.ones(N)*(t[i] - t[i-1]) for i in range(ws)])
    # integral = integral*timeinterval
    # integral[0] = integral[0]*0
     
    # output = np.cumsum(integral,0)[-1]

    # output = 2*np.imag(output)
    # np.save('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/output',np.imag(np.exp(-1j*Sst)[40])) 

    # print(output)
    # print(output[4000:4050])
    # # print(foutput[4000:4050])
    # fast = time.time()-start
    # # temp = abs(output-foutput)
    # etemp = np.where(temp>1e-10)
    # print(etemp[0].size)
    # # print(output-foutput)
    # np.save('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/output',output)
    # print('The difference is slow: {} , fast: {}'.format(slow,fast))
    return foutput

    
# 

def lewenstein(N, t, Et_data, weight_length, weights, at, Ip, epsilon_t, dp, dim):
    # Initialize arrays
    Et = Et_data.reshape((N, dim))
    At = np.zeros((N, dim))
    Bt = np.zeros((N, dim))
    Ct = np.zeros(N)
    output = np.zeros((N, dim), dtype=np.float64)
    
    # Constants
    pi = np.pi
    i = 1j

    # Precompute At, Bt, Ct
    for t_i in range(1, N):
        dt = t[t_i] - t[t_i - 1]
        Et_avg = (Et[t_i - 1] + Et[t_i]) * (dt / 2)
        At[t_i] = At[t_i - 1] - Et_avg
        Bt[t_i] = Bt[t_i - 1] + (At[t_i - 1] + At[t_i]) * (dt / 2)
        Ct[t_i] = Ct[t_i - 1] + 0.5 * dt * (np.sum(At[t_i - 1]**2) + np.sum(At[t_i]**2))

    # Main loop
    for t_i in range(1, N):
        inde = min(weight_length, t_i + 1)
        integral = np.zeros(dim, dtype=np.complex128)
        last_integrand = np.zeros(dim, dtype=np.complex128)

        for tau_i in range(inde):
            tau = t[tau_i]
            pst = (Bt[t_i] - Bt[t_i - tau_i]) / tau if tau_i > 0 else At[t_i]

            argdstar = pst - At[t_i]
            argdnorm = pst - At[t_i - tau_i] if tau_i > 0 else np.zeros(dim)

            dnorm = dp(argdnorm)
            dstar = np.conj(dp(argdstar))

            Sst = (Ip * tau - 0.5 / tau * np.sum((Bt[t_i] - Bt[t_i - tau_i])**2) +
                   0.5 * (Ct[t_i] - Ct[t_i - tau_i])) if tau_i > 0 else 0

            c = pi / (epsilon_t + 0.5 * i * tau)
            phase = np.exp(-1j * Sst)
            prefactor = c * np.sqrt(c)

            integrand13 = dstar * (dnorm * Et[t_i - tau_i]) * prefactor * phase * weights[tau_i] * at[t_i] * at[t_i - tau_i]

            dt = t[tau_i] - t[tau_i - 1] if tau_i > 0 else 0
            integral += (last_integrand + integrand13) * dt / 2
            last_integrand = integrand13

        output[t_i] = 2.0 * np.imag(integral)

    output[0] = 0
    return output

matdata = open('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/Electric.txt')
matlabarray = []
for line in matdata.readlines():
    matlabarray.append(line.split(','))
matlabarray = matlabarray[0]
matlabarray[-1] = matlabarray[-1][:-2]
matlabarray = [float(i) for i in matlabarray]
alpha = config.alpha
Ip = config.Ip
prefactor = (2**3.5) * (alpha**1.25) / np.pi 
def dp(p):
    return 1j*prefactor*p/((np.square(p) + alpha)**3)
# output = np.load('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/output.npy')
foutput = lewenstein(t.size,t,np.array([matlabarray[:-1]]),weights.size,weights,np.ones_like(t),Ip,1e-4,dp,1)
plt.plot(foutput)
# plt.savefig('/home/alex/Desktop/Python/SNAIL/Latex/coolgraph.png')
# plt.plot(driving_field[0])
# plt.plot(matlabarray)





