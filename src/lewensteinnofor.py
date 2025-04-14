import numpy as np
  
'''
This is Lewenstein without for loops, need to test why it gives odd results

'''
def lewenstein(t,Et_data,lconfig,at=None,epsilon_t=1e-4):
    '''
    
    Calculates the dipole response, with the provided dipole elements dp,
    N - number of timesteps
    t - the timesteps to calculate (doesn't have to be equally spaced)
    Et - Electric field data
    
    
    '''

    Et = Et_data
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

    ws = weights.size
    c = (np.pi/(epsilon_t + 0.5*1j*t[:ws]))**1.5

    bigAt = np.reshape(np.tile(At,ws),(ws,At.size))
    temptAt = bigAt[np.c_[:bigAt.shape[0]], (np.r_[:bigAt.shape[1]] - np.c_[:ws]) % bigAt.shape[1]]
    
    bigBt = np.reshape(np.tile(Bt,ws),(ws,Bt.size))
    temptBt = bigBt[np.c_[:bigBt.shape[0]], (np.r_[:bigBt.shape[1]] - np.c_[:ws]) % bigBt.shape[1]]
    
    bigCt = np.reshape(np.tile(Ct,ws),(ws,Ct.size))
    temptCt = bigCt[np.c_[:bigCt.shape[0]], (np.r_[:bigCt.shape[1]] - np.c_[:ws]) % bigCt.shape[1]]
    
    
    pst = (bigBt - temptBt)/np.c_[t[:ws]]

    correction = np.r_[:pst.shape[1]] > np.c_[:pst.shape[0]]
        
    pst = pst*correction

    argdstar = pst - bigAt
    argdstar = argdstar*correction
    argdnorm = pst - temptAt
    argdnorm = argdnorm*(correction)
    
    del bigAt
    del temptAt

    dstar = np.conjugate(dp(argdstar))
    dnorm = dp(argdnorm)
    dnorm = dnorm*correction
    dstar = dstar*correction

    SQR = np.square
    Sst = np.zeros((ws,N),dtype='complex')
    integral = np.zeros((ws,N))
    dt = np.diff(t)

    Sst = -(0.5/np.array([t[:ws]]).T)*SQR(bigBt - temptBt) + 0.5*(bigCt-temptCt) + Ip*np.reshape(t[:ws],(1,ws)).T
    
    del bigBt
    del temptBt
    
    del bigCt
    del temptCt


    Sst = Sst*correction
    
    bigEt = np.reshape(np.tile(Et,ws),(ws,Et.size))
    temptEt = bigEt[np.c_[:bigEt.shape[0]], (np.r_[:bigEt.shape[1]] - np.c_[:ws]) % bigEt.shape[1]]
    
    bigat = np.reshape(np.tile(at,ws),(ws,at.size))
    temptat = bigat[np.c_[:bigat.shape[0]], (np.r_[:bigat.shape[1]] - np.c_[:ws]) % bigat.shape[1]]
    
    integral = dstar*dnorm*np.exp(-1j*Sst)*temptEt*(np.c_[weights])*(np.c_[c])*(bigat)*temptat
    # for tau in range(1,ws):
        

    #     # integral[tau-1] = dstar[tau-1]*dnorm[tau-1]*np.roll(Et,tau)*(c[tau])*(np.cos(Sst[tau-1]) - 1j*np.sin(Sst[tau-1]))*weights[tau]*at*np.roll(at,tau)
    #     integral[tau-1] = dstar[tau-1]*dnorm[tau-1]*(np.exp(-1j*Sst[tau-1]))
    #     integral[tau-1] = integral[tau-1]*np.roll(Et,tau)*weights[tau]*at*np.roll(at,tau)*(c[tau])
    
    timeinterval  = np.array([np.ones(N)*(t[i] - t[i-1]) for i in range(ws)])




    # for tau in range(2,ws):
    #     output[tau:] += ((integral[tau-2])[tau:]+ (integral[tau-1])[tau:])*(t[tau-1]-t[tau-2]) 
    integral = integral*timeinterval
    # integral = integral*error
    output = np.cumsum(integral,0)[-1]
    # print(output)

    return output


