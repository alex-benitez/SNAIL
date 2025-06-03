import numpy as np
import time
from numpy import pi, sqrt, cos, sin, log
from multiprocessing import cpu_count,Pool
from multiprocessing import shared_memory

'''

The current operational plans are:
    1. Identify the number of available CPU cores and the amount of RAM
    2. Split up the calculations into neat little chunks to increase parallelization capabilities
    To do this I have to take the available cores and be like: Ok you get the first 1/4 of the tau integrals etc etc.

'''

def barebones_lewenstein(weights,start,N,lconfig,at=None,epsilon_t=1e-4):
    wp = weights.size
    end = start+wp
    Ip = lconfig.Ip
    existing_shm = shared_memory.SharedMemory(name='general_buffer')
    numerators = np.arange(start,start+wp-1)


    # Note that a.shape is (6,) and a.dtype is np.int64 in this example
    
    # Et,At,Bt,Ct,t = np.ndarray((5,), dtype=np.float64, buffer=existing_shm.buf)
    Et,At,Bt,Ct,t = np.ndarray((5,N), dtype=np.float64, buffer=existing_shm.buf)
    if at is None: at = np.ones_like(t)
    
    
    c = (pi/(epsilon_t + 0.5*1j*t[start:end]))**1.5

    # print('That took {}'.format(time.time()-start))
    
    pst = np.array([(-np.roll(Bt,i)+Bt)/t[i] for i in range(start+1,end)])
    error = np.ones(pst.shape)
    error_complex = np.ones(pst.shape,dtype='complex')
    for pos,val in enumerate(numerators):
        error[pos,:(val+1)] = 0 
        error_complex[pos,:(val+1)] = 0 + 0j 
        
    pst = pst*error


    
    # print(pst[np.where(pst!=0)])
    alpha = lconfig.alpha
    prefactor = (2**3.5) * (alpha**1.25) / pi 
    #d(p) = i * 2^3.5*alpha^1.25/pi * p/(p^2 + alpha)^3
    def dp(p):
        return 1j*prefactor*p/((np.square(p) + alpha)**3)
    print(wp)
    argdstar = pst - np.reshape(np.tile(At,wp-1),(wp-1,At.size))
    argdstar = argdstar*error
    argdnorm = pst - np.array([(np.roll(At,i)) for i in range(start+1,end)])
    argdnorm = argdnorm*(error)

    dstar = np.conjugate(dp(argdstar))
    dnorm = dp(argdnorm)
    dnorm = dnorm*error_complex
    dstar = dstar*error_complex

    SQR = np.square
    try:
        Sst = np.zeros((wp-1,N),dtype='complex')
    except:
        print('Ran into memory issues, oops! Going to try something different now')
    integral = np.zeros((wp-1,N))
    dt = np.diff(t)
    temptBt = np.array([(np.roll(Bt,i)) for i in range(start+1,end)])
    temptCt = np.array([(np.roll(Ct,i)) for i in range(start+1,end)])
    Sst = -(0.5/np.array([t[start+1:end]]).T)*SQR(np.reshape(np.tile(Bt,wp-1),(wp-1,Bt.size))-temptBt) + 0.5*(np.reshape(np.tile(Ct,wp-1),(wp-1,Ct.size))-temptCt) + Ip*np.array([t[start+1:end]]).T
    
    del temptBt
    del temptCt


    Sst = Sst*error_complex

    for tau,val in enumerate(numerators[:-1]):
        
        tau = tau+1
        integral[tau-1,:] = dstar[tau-1]*dnorm[tau-1]*np.roll(Et,tau)*(c[tau])*(np.cos(Sst[tau-1,:]) - 1j*np.sin(Sst[tau-1,:]))*weights[tau]*at*np.roll(at,tau)
       
    
    timeinterval  = np.array([np.ones(N)*(t[i] - t[i-1]) for i in range(start+1,end)])




    # for tau in range(2,ws):
    #     output[tau:] += ((integral[tau-2])[tau:]+ (integral[tau-1])[tau:])*(t[tau-1]-t[tau-2]) 
    integral = integral*timeinterval
    return integral
    # integral = integral*error
    # output = np.cumsum(integral,0)[:,-1]
    # # print(output)
    # print(output)
    # return output
    
def parallel_lewenstein(t,Et_data,lconfig,at=None,epsilon_t=1e-4):
    '''
    
    Calculates the dipole response, with the provided dipole elements dp
    N - number of timesteps
    t - the timesteps to calculate (doesn't have to be equally spaced)
    Et - Electric field data
    
    
    '''
    if hasattr(lconfig,'cores'):
        cores = lconfig.cores
    else:
        cores = cpu_count()-2
    try:
        # Deletes the cache if it already exists
        shm = shared_memory.SharedMemory(create=False, size=1,name='general_buffer')
        shm.close()
        shm.unlink()
        del shm
    except:
        pass
    # start = time.time()
    Et = Et_data
    # Here lies 30h of my time debugging, always remember to check for redundant dimensions!
    weights = lconfig.weights
    
    Ip = lconfig.Ip

    # epsilon_t = lconfig.epsilon_t
    alpha = lconfig.alpha
    
    N = Et_data.size
    ws = weights.size
    
    split = int(ws/cores)
    
    # print('That took {}'.format(time.time()-start))

    

    
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
    
    t = t-t[0]

    general = np.vstack((Et,At,Bt,Ct,t))
    

    shm = shared_memory.SharedMemory(create=True, size=general.nbytes,name='general_buffer')
    b = np.ndarray(general.shape, dtype=general.dtype, buffer=shm.buf)
    b[:] = general[:]
    del b

    
    
    
    # print(b)
    info = [(weights[split*i:split*(i+1)],split*i,N,lconfig) for i in range(cores-1)]
    info.append((weights[split*(cores-1):],split*(cores-1),N,lconfig))
    print('Here in')
    # Create a process pool
    with Pool(processes=cores) as pool:
        print('In here')
        results = pool.starmap(barebones_lewenstein, info)
        
    results = np.vstack((results))
    results = results[1:]
    
    results = np.cumsum(results,0)
    shm.close()
    shm.unlink()
    print(results[-1][20000:20100])
        
    return results[-1]
    # print('That took {}'.format(time.time()-start))
    
    # Now the meat and bones, am I Linus Torvalds, or just some schmuck
    # c**1.5 is 10x faster than c*np.sqrt(c)

    
    