'''
This is probably a stupid idea, but my hate for Matlab is stronger than my regard for my own time and sanity
'''

import numpy as np
import time
from numpy import pi, sqrt, cos, sin, log
from numpy.linalg import norm
import scipy.constants as constants
import matplotlib.pyplot as plt

    
    
    
    
def lewenstein(t,Et_data,lconfig,at=None,epsilon_t=1e-4):
    '''
    
    Calculates the dipole response, with the provided dipole elements dp
    N - number of timesteps
    t - the timesteps to calculate (doesn't have to be equally spaced)
    Et - Electric field data
    
    
    '''

    # start = time.time()
    Et = Et_data
    weights = lconfig.weights
    if at is None: at = np.ones_like(t)
    Ip = lconfig.Ip
    # epsilon_t = lconfig.epsilon_t
    alpha = lconfig.alpha
    
    N = Et_data.size
    
    # print('That took {}'.format(time.time()-start))
    prefactor = (2**3.5) * (alpha**1.25) / pi 
    #d(p) = i * 2^3.5*alpha^1.25/pi * p/(p^2 + alpha)^3
    def dp(p):
        return 1j*prefactor*p/((np.square(p) + alpha)**3)
    

    
    dt = (-np.roll(t,1) + t)*0.5
    dt[0] = 0
    
    At = -(np.roll(Et,1) + Et)*dt 
    
    At = At[0]

    At[0] = 0
    At = np.cumsum(At)


    Bt = (np.roll(At,1) + At)*dt

    Bt[0] = 0
    Bt = np.cumsum(Bt)

    Ct = (np.square(np.roll(At,1)) + np.square(At))*dt

    Ct[0] = 0
    Ct = np.cumsum(Ct)

    
    # print('That took {}'.format(time.time()-start))
    
    # Now the meat and bones, am I Linus Torvalds, or just some schmuck
    # c**1.5 is 10x faster than c*np.sqrt(c)

    ws = weights.size
    c = (pi/(epsilon_t + 0.5*1j*t[:ws]))**1.5

    # print('That took {}'.format(time.time()-start))
    
    '''
# =============================================================================
#     Currently technically finished, need to generate the data to check if it works
#     Spoiler alert: He was not in fact finished (Written 2 months later whilst still fixing it)
# =============================================================================
    '''

    pst = np.array([(-np.roll(Bt,i)+Bt)/t[i] for i in range(1,ws)])
    error = np.ones(pst.shape)
    error_complex = np.ones(pst.shape,dtype='complex')
    for i in range(ws-1):
        error[i,:(i+1)] = 0 
        error_complex[i,:(i+1)] = 0 + 0j 
        
    pst = pst*error
    '''
    Need to sit down and work through this loop again,
    probably doing it wrong.
    
    '''
    # print(pst[np.where(pst!=0)])
    
    argdstar = pst - np.reshape(np.tile(At,ws-1),(ws-1,At.size))
    argdstar = argdstar*error
    argdnorm = pst - np.array([(np.roll(At,i)) for i in range(1,ws)])
    argdnorm = argdnorm*(error)

    dstar = np.conjugate(dp(argdstar))
    dnorm = dp(argdnorm)
    dnorm = dnorm*error_complex
    dstar = dstar*error_complex

    SQR = np.square
    Sst = np.zeros((ws-1,N),dtype='complex')
    integral = np.zeros((ws-1,N))
    dt = np.diff(t)
    temptBt = np.array([(np.roll(Bt,i)) for i in range(1,ws)])
    temptCt = np.array([(np.roll(Ct,i)) for i in range(1,ws)])
    Sst = -(0.5/np.reshape(t[1:ws],(1,ws-1)).T)*SQR(np.reshape(np.tile(Bt,ws-1),(ws-1,Bt.size))-temptBt) + 0.5*(np.reshape(np.tile(Ct,ws-1),(ws-1,Ct.size))-temptCt) + Ip*np.reshape(t[1:ws],(1,ws-1)).T
    
    del temptBt
    del temptCt
    # for tau in range(1,ws):
    #     tempt = t[tau]+0j
    #     Sst[tau-1,:] = Ip*(t[tau]) - (0.5/t[tau])*SQR(Bt - np.roll(Bt,tau)) + 0.5*(Ct - np.roll(Ct,tau))

    Sst = Sst*error_complex


    for tau in range(1,ws):
        

        integral[tau-1] = dstar[tau-1]*dnorm[tau-1]*np.roll(Et,tau)*(c[tau])*(np.cos(Sst[tau-1]) - 1j*np.sin(Sst[tau-1]))*weights[tau]*at*np.roll(at,tau)
       
    
    timeinterval  = np.array([np.ones(N)*(t[i] - t[i-1]) for i in range(1,ws)])




    # for tau in range(2,ws):
    #     output[tau:] += ((integral[tau-2])[tau:]+ (integral[tau-1])[tau:])*(t[tau-1]-t[tau-2]) 
    integral = integral*timeinterval
    # integral = integral*error
    output = np.cumsum(integral,0)[-1]
    # print(output)

    return output


# N = 1000000
# # Et = np.random.rand(N)

# weights = np.random.random(6)
# t = np.linspace(0,1,N)
# Et = np.cos(t*0.1)
# at = np.random.rand(N)
# Ip = 1
# epsilon_t = 0.1



# lewenstein(N,t,Et,weights,at,Ip,epsilon_t,0.5)
    
# def lewenstein(N,t,Et_data,weights,at,Ip,epsilon_t,alpha):
#     '''
    
#     Calculates the dipole response, with the provided dipole elements dp
#     N - number of timesteps
#     t - the timesteps to calculate (doesn't have to be equally spaced)
#     Et - Electric field data
    
    
#     '''
#     # start = time.time()
    
    
#     # print('That took {}'.format(time.time()-start))
#     prefactor = (2**3.5) * (alpha**1.25) / pi * 1j
    
#     def dp(p):
#         return prefactor/((np.square(p) + alpha)**3)
    
#     At,Bt,Ct = np.zeros((3,N))
#     dt = (-np.roll(t,1) + t)*0.5
#     At = -(np.roll(Et,1) + Et)*dt
#     At[0] = 0
#     At = np.cumsum(At)

#     Bt = (np.roll(At,1) + At)*dt
#     Bt[0] = 0
#     Bt = np.cumsum(Bt)

#     Ct = (np.square(np.roll(At,1)) + np.square(At))*dt
#     Ct[0] = 0
#     Ct = np.cumsum(Ct)

    
#     # print('That took {}'.format(time.time()-start))
    
#     # Now the meat and bones, am I Linus Torvalds, or just some schmuck
#     # c**1.5 is 10x faster than c*np.sqrt(c)
#     # I don't think c needs to be calculated every loop, just saying
#     # Do the loop for values smaller or equal to the weight lengths afterwards
#     # First try to do it without a loop at all
#     ws = weights.size
#     c = (pi/(epsilon_t + 0.5*1j*t[:ws]))**1.5

#     # print('That took {}'.format(time.time()-start))
    
#     '''
# # =============================================================================
# #     Currently technically finished, need to generate the data to check if it works
# # =============================================================================
#     '''
    
#     pst = np.array([(-np.roll(Bt,i)+Bt)/t[i] for i in range(1,ws)])
#     error = np.ones(pst.shape)
#     for i in range(ws):
#         error[i:i+1,:i+1] = 0 
#     pst = pst*error
    
#     argdstar = pst - np.reshape(np.tile(At,ws-1),(ws-1,At.size))
#     argdstar = argdstar*error
#     argdnorm = pst + np.array([(-np.roll(At,i)) for i in range(1,ws)])
#     argdnorm = argdnorm*error
    
#     dstar = np.conjugate(dp(argdstar))
#     dnorm = dp(argdnorm)
#     SQR = np.square
#     Sst = np.zeros((ws-1,N))

#     dt = np.diff(t)
#     integral = np.zeros((ws-1,N))
#     for tau in range(1,ws):
#         Sst[tau-1,:] = Ip*t[tau] - (0.5/t[tau])*SQR(Bt - np.roll(Bt,tau)) + 0.5*(Ct - np.roll(Ct,tau))

#         integral[tau-1] = np.imag(dstar[tau-1]*dnorm[tau-1]*np.roll(Et,tau)*(c[tau])*(np.cos(Sst[tau-1]) - 1j*np.sin(Sst[tau-1]))*weights[tau]*at*np.roll(at,tau))
    

#     output = np.zeros(N)
#     for tau in range(1,ws):
#         output += integral[tau-1]*(t-np.roll(t,tau)) 
        
#     # output = (np.sum(integral[:,1:]*dt,axis=0))
   
#     # print('That took {}'.format(time.time()-start))

#     return output
