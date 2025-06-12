import numpy as np
import time
from numpy import pi, sqrt, cos, sin, log
from multiprocessing import cpu_count,Pool
from multiprocessing import shared_memory


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
    alpha = lconfig.alpha
    prefactor = (2**3.5) * (alpha**1.25) / np.pi 
    def dp(p):
        return 1j*prefactor*p/((np.square(p) + alpha)**3)
    
    ws = weights.size
    bigAt = np.reshape(np.tile(At,ws),(ws,At.size))
    temptAt = bigAt[np.c_[:bigAt.shape[0]], (np.r_[:bigAt.shape[1]] - np.c_[start:ws+start]) % bigAt.shape[1]]

    bigCt = np.reshape(np.tile(Ct,ws),(ws,Ct.size))
    temptCt = bigCt[np.c_[:bigCt.shape[0]], (np.r_[:bigCt.shape[1]] - np.c_[start:ws+start]) % bigCt.shape[1]]
    
    c = (np.pi/(epsilon_t + 0.5*1j*t[start:ws+start]))**1.5
    
    bigBt = Bt*np.c_[np.ones(ws)] # Alternate method of generating big matrix
    temptBt = bigBt[np.c_[:bigBt.shape[0]], (np.r_[:bigBt.shape[1]] - np.c_[start:ws+start]) % bigBt.shape[1]]
    pst = (bigBt - temptBt)/np.c_[t[start:ws+start]]
    if start == 0:
        pst[0] = At
    correction = np.r_[:pst.shape[1]]+1 > np.c_[:pst.shape[0]]+start
    
    np.save('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/pst{}.npy'.format(start),pst)
    pst = pst*correction
    
    
    argdstar = pst - bigAt
    argdstar = argdstar*correction
    argdnorm = pst - temptAt
    argdnorm = argdnorm*(correction)

    dstar = np.conjugate(dp(argdstar))
    dnorm = dp(argdnorm)
    dnorm = dnorm*correction
    dstar = dstar*correction

    
    SQR = np.square
    integral = np.zeros((ws,N))
    dt = np.diff(t)

    Sst = -(0.5/np.c_[t[start:start+ws]])*SQR(bigBt - temptBt) + 0.5*(bigCt-temptCt) + Ip*np.c_[t[start:start+ws]]
    
    Sst[0] = Sst[0]*(1-start==0)
    
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

    Et = np.squeeze(Et_data)
    t = t-t[0] +0.000001


    weights = lconfig.weights
    
    Ip = lconfig.Ip

    
    N = Et_data.size
    ws = weights.size
    
    split = round(ws/cores) # Evenly split the tasks amongs the cores
    
    # print('That took {}'.format(time.time()-start))

    

    
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
    

    general = np.vstack((Et,At,Bt,Ct,t))
    

    shm = shared_memory.SharedMemory(create=True, size=general.nbytes,name='general_buffer')
    b = np.ndarray(general.shape, dtype=general.dtype, buffer=shm.buf)
    b[:] = general[:]
    del b

    
    
    
    # print(b)
    info = [(weights[split*i:split*(i+1)],split*i,N,lconfig) for i in range(cores-1)]
    info.append((weights[split*(cores-1):],split*(cores-1),N,lconfig))
    print('Running on {} cores'.format(cores))
    print(len(info))
    # Create a process pool
    with Pool(processes=cores) as pool:

        results = pool.starmap(barebones_lewenstein, info)
        print('Yipeee')

    results = np.vstack((results))
    print(results.shape)


    results = 2*np.imag(np.cumsum(results,0)[-1])
    np.save('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/parallel.npy',results)    
    shm.close()
    shm.unlink()

        
    return results
    # print('That took {}'.format(time.time()-start))
    
    # Now the meat and bones, am I Linus Torvalds, or just some schmuck
    # c**1.5 is 10x faster than c*np.sqrt(c)

    
    