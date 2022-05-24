from matplotlib import pyplot as plt 
import numpy as np
from scipy.special import pbwa

def U(a,x):
    return pbwa(a,x)[0]

T = np.linspace(5e-2,3,num=300)
J_lin = 1/(np.sqrt(T*np.pi))
J_sph = (1+np.sqrt(T*np.pi))/(np.sqrt(T*np.pi))
J_band = (np.pi*T)**(-0.5) + 1.0 - ((2.0**0.75)/np.pi)*(T**0.75)*np.exp(-1.0/(8.0*T)) * (U(2.0,(2.0*T)**(-0.5)))
plt.plot(T,J_lin,label='Linear')
plt.plot(T,J_sph,label='spherical')
plt.plot(T,J_band,label='Band electrode')
plt.legend()

plt.show()

