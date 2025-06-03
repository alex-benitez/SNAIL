import numpy as np
  
def lewenstein(t,Et_data,lconfig,at=None,epsilon_t=1e-4):
    '''
    
    Calculates the dipole response, with the provided dipole elements dp,
    N - number of timesteps
    t - the timesteps to calculate (doesn't have to be equally spaced)
    Et - Electric field data
    
    
    '''

    Et = np.squeeze(Et_data)
    weights = lconfig.weights
    if at is None: at = np.ones_like(t)
    Ip = lconfig.Ip
    alpha = lconfig.alpha
    
    N = Et_data.size
    

    prefactor = (2**3.5) * (alpha**1.25) / np.pi 
    def dp(p):
        return 1j*prefactor*p/((np.square(p) + alpha)**3)
    

    
    dt = (-np.roll(t,1) + t)*0.5
    dt[0] = 0
    
    At = -(np.roll(Et,1) + Et)*dt 
    
    At = np.squeeze(At)
    At[0] = 0
    At = np.cumsum(At)


    Bt = (np.roll(At,1) + At)*dt
    Bt[0] = 0
    Bt = np.cumsum(Bt)

    Ct = (np.square(np.roll(At,1)) + np.square(At))*dt
    Ct[0] = 0
    Ct = np.cumsum(Ct)
    
    # c**1.5 is 10x faster than c*np.sqrt(c)
    # Section above is 2 orders of magnitude faster than using for loops (0.04641/0.00031)

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
    correction = np.r_[:pst.shape[1]]+1 > np.c_[:pst.shape[0]]
    pst = pst*correction
    
    # Section above is 4 times faster than with for loops: (0.46973681449890137/0.10493278503417969)
    
    argdstar = pst - bigAt
    argdstar = argdstar*correction
    argdnorm = pst - temptAt
    argdnorm = argdnorm*(correction)

    dstar = np.conjugate(dp(argdstar))
    dnorm = dp(argdnorm)
    dnorm = dnorm*correction
    dstar = dstar*correction
    
    # Section above is two orders of magnitude faster: (7.2791759967803955/0.1377711296081543)
    
    SQR = np.square
    integral = np.zeros((ws,N))
    dt = np.diff(t)

    Sst = -(0.5/np.c_[t[:ws]])*SQR(bigBt - temptBt) + 0.5*(bigCt-temptCt) + Ip*np.c_[t[:ws]]
    
    Sst[0] = Sst[0]*0
    
    del bigBt
    del temptBt
    del temptCt


    Sst = Sst*correction
    bigEt = np.reshape(np.tile(Et,ws),(ws,Et.size))
    temptEt = bigEt[np.c_[:bigEt.shape[0]], (np.r_[:bigEt.shape[1]] - np.c_[:ws]) % bigEt.shape[1]]
    
    bigat = np.reshape(np.tile(at,ws),(ws,at.size))
    temptat = bigat[np.c_[:bigat.shape[0]], (np.r_[:bigat.shape[1]] - np.c_[:ws]) % bigat.shape[1]]
    
    integral = dstar*dnorm*np.exp(-1j*Sst)*temptEt*(np.c_[weights])*(np.c_[c])*(bigat)*temptat
    # for tau in range(ws):
    #     integral[tau] = dstar[tau]*dnorm[tau]*np.roll(Et,tau)*(c[tau])*np.exp(-1j*Sst)*weights[tau]*at*np.roll(at,tau)
       
        # integral[tau-1] = dstar[tau-1]*dnorm[tau-1]*(np.exp(-1j*Sst[tau-1]))
        # integral[tau-1] = integral[tau-1]*np.roll(Et,tau)*weights[tau]*at*np.roll(at,tau)*(c[tau])
    
    timeinterval  = np.array([np.ones(N)*(t[i] - t[i-1]) for i in range(ws)])




    # for tau in range(2,ws):
    #     output[tau:] += ((integral[tau-2])[tau:]+ (integral[tau-1])[tau:])*(t[tau-1]-t[tau-2]) 
    integral = integral*timeinterval
    
    # integral = integral*error
    output = 2*np.imag(np.cumsum(integral,0)[-1])

    return output


