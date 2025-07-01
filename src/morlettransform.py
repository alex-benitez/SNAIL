"""
Created on Sat Jun 14 13:04:00 2025

Wavelet transform for further analysis of SNAIL

@author: alex
"""
import numpy as np
import matplotlib.pyplot as plt
from math import erf
from general_tools import sau_convert
from general_tools import config

def morlet(k,W,s):
    xf = 2*np.pi*s
    stat = 4*np.sqrt(np.pi)/(1+erf(W))
    pref = stat*s
    
    return pref*np.exp(-((W - xf*k)**2)/2)

def wavelet_transform(signal, t=None, maxF=0.5, maxT=0.01, JN=500, W=6):
    """
    Performs a continuous wavelet transform (CWT) using the Morlet wavelet.
    
    Parameters:
    - signal: Input signal (1D numpy array)
    - t: Time array (1D numpy array), same length as signal
    - maxF: Max frequency as a fraction of the Nyquist frequency (default 0.2)
    - maxT: Max period as a fraction of the total time range (default 0.1)
    - JN: Number of scales (default 300)
    - W: Wavelet parameter (default 7)
    """
    
    if t is None:
        t = 2*np.pi*np.linspace(0,60,signal.size)
        
    # t = t - t[0] +0.00001
    
    lambda0 = 1000e-9  # Laser wavelength [m]
    C = 3e8           # Speed of light [m/s]
    T0 = lambda0 / C
    omega0 = 2 * np.pi / T0

    N = len(t)
    T = t[-1] - t[0]

    Js = np.log2(W / (2.0 * np.pi * maxF))
    Jf = np.log2(maxT * N * W / (2.0 * np.pi))

    dj = (Jf - Js) / JN
    js = int(np.floor(Js / dj))

    s0 = 1 / N
    s = np.zeros(JN)
    wave = np.zeros((JN, N), dtype=np.complex128)

    fsig = np.fft.fft(signal)
    print(fsig[np.where(fsig!=0)].size)
    for n in range(JN):
        j = n + js
        s[n] = s0 * 2**(j * dj)
        M = morlet(np.r_[0:N], W, s[n])  # Assumes morlet returns frequency-domain filter
        Msig = M * fsig

        wave[n, :] = np.flip(np.fft.ifft(np.flip(Msig)))
          # matches MATLAB fliplr

    S = W / (2 * np.pi * s) * (1 / T)
    Snew = S / omega0
    # 
    # Plotting
    plt.figure(figsize=(6, 4))
    
    plt.imshow(np.log(np.abs(wave)**2), extent=[t[0], t[-1], S[-1], S[0]],
               aspect='auto', cmap='hsv')
    # clim = plt.gci().get_clim()
    # plt.clim(clim[1] - 10, clim[1])
    plt.xlabel('Time [fs]', fontsize=12)
    plt.ylabel('Harmonic order', fontsize=12)
    # plt.xlim(-25,25)
    plt.ylim(0,32)
    plt.xlim(-20,20)
    
    
    field = np.load('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/driving.npy')
    xvals = t[np.where(np.abs(t)<25)]
    yvals =  2.5*field[np.where(np.abs(t)<25)]/max(field[np.where(np.abs(t)<25)])+13
    # plt.plot(xvals,yvals,'k--')
    plt.gca().invert_yaxis()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.gcf().set_facecolor('white')
    plt.tight_layout()
    plt.colorbar()
    
    
    plt.savefig('/home/alex/Desktop/Python/SNAIL/images/gaborplot.png',dpi=300)
    plt.show()
    # return wave, t, S

# response1 = np.load('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/single.npy')
# time = np.load('/home/alex/Desktop/Python/SNAIL/src/stored_arrays/time.npy')
response1 = np.loadtxt('/home/alex/Desktop/Python/SNAIL/dt3.dat')
time = np.loadtxt('/home/alex/Desktop/Python/SNAIL/time3.dat')
wavelet_transform(response1,time)

