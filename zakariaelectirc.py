import numpy as np

def electric_field(t, params, env):
    w = params['w0']         # Frequency
    phi = params['phi']         # Phase
    E0 = params['E0']         # Amplitude
    tau = params['tau']         # Pulse width
    tc = params['tc']
    Tp = 2.751 * tau

    if(env == "gaussian"):
        Enev = np.exp(-(t - tc)**2 / (2 * tau * tau))  # Gaussian envelope  
    
    if(env == "cos2"):
        # active only in [0, Tp], where Tp=2.751*tau
        if (t.all() > 0 or t.all() < Tp):
            cosTerm = np.cos(np.pi * t/Tp + np.pi/2)
        else:
            cosTerm = 0.0
        Enev = cosTerm * cosTerm  # Cos2 envelope  

    if(env == "sin6"):
        # active only in [0, Tp], where Tp=2.751*tau
        if (t.all() > 0 or t.all() < Tp):
            cosTerm = np.cos(np.pi * t/Tp) 
        else:
            cosTerm = 0.0
        Enev = 1-cosTerm**6    # Sin6 envelope  

    if(env == "trapez"):
        T0 = 2.0 * np.pi * w
        tau = 10.0 * T0         # Pulse width
        n = 10
        trapzTerm = np.zeros_like(t)
        mask1 = (t >= 0) & (t < T0)
        mask2 = (t >= T0) & (t < (n-1) * T0)
        mask3 = (t >= (n-1) * T0) & (t < n * T0)

        trapzTerm[mask1] = t[mask1] / T0
        trapzTerm[mask2] = 1.0
        trapzTerm[mask3] = (10 * T0 - t[mask3]) / T0
        Enev =  trapzTerm      # Trapez envelope  

    print(tau)

    return E0 * Enev * np.sin(w * t + phi)  


params = {'w0': 0.057, 'phi': 0.0, 'E0': 0.053, 'tau': 125.0 * 2.9, 'tc': 500}
envelope = {'gaussian', 'cos2', 'sin6'}
t = np.linspace(0, 1000, 6000)
np.save('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/zakariaelectric.npy',electric_field(t, params, "sin6"))
np.save('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/zakariatime.npy',t)

# plt.plot(t, )
# plt.show()