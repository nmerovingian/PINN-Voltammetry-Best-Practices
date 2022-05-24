import numpy as np 
from matplotlib import pyplot as plt 



T = np.linspace(1e-3,1)
J_lin = 1/(np.sqrt(T*np.pi))
J_sph = (1+np.sqrt(T*np.pi))/(np.sqrt(T*np.pi))

plt.plot(T,J_lin,label='Linear')
plt.plot(T,J_sph,label='spherical')
plt.legend()

plt.show()