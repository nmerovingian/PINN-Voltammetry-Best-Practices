import numpy as np 
from matplotlib import pyplot as plt
from cycler import cycler
default_cycler = (cycler(color=["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"]))
plt.rc('axes', prop_cycle=default_cycler)

num_train_samples = 1000

maxT = 1.0
Re = 1.0


R_e = np.random.rand(num_train_samples)*Re
theta_e = np.random.rand(num_train_samples)*np.pi/2
TXYZ_e = np.random.rand(num_train_samples,4)
TXYZ_e[:,0] *= maxT
TXYZ_e[:,2] = np.sqrt(R_e)*np.cos(theta_e) 
TXYZ_e[:,1] = np.sqrt(R_e)*np.sin(theta_e) 
TXYZ_e[:,3] = 0.0


fig = plt.figure()
ax = fig.add_subplot(projection='3d')


ax.scatter(TXYZ_e[:,1],TXYZ_e[:,2],TXYZ_e[:,3],alpha=0.6)



R_e_1 = np.random.rand(num_train_samples)*0.4 + 0.6
theta_e_1 = np.random.rand(num_train_samples)*np.pi/2
TXYZ_e_1 = np.random.rand(num_train_samples,4)
TXYZ_e_1[:,0] *= maxT
TXYZ_e_1[:,2] = R_e_1*np.cos(theta_e_1) 
TXYZ_e_1[:,1] = R_e_1*np.sin(theta_e_1) 
TXYZ_e_1[:,3] = 0.0
ax.scatter(TXYZ_e_1[:,1],TXYZ_e_1[:,2],TXYZ_e_1[:,3],alpha=0.6)




TXYZ_bnd_0 = np.random.rand(num_train_samples,4)
TXYZ_bnd_0[:,0] *= maxT
TXYZ_bnd_0[:,2] *= 3 
TXYZ_bnd_0[:,1] *= 3 
TXYZ_bnd_0[:,3] = 0.0

TXYZ_bnd_0 = TXYZ_bnd_0[TXYZ_bnd_0[:,1]**2+TXYZ_bnd_0[:,2]**2>1.0]

print(TXYZ_bnd_0.shape)

while len(TXYZ_bnd_0) < num_train_samples:
    TXYZ_bnd_temp = np.random.rand(int(num_train_samples/5),4)
    TXYZ_bnd_temp[:,0] *= maxT
    TXYZ_bnd_temp[:,1] *= 3 
    TXYZ_bnd_temp[:,2] *= 3 
    TXYZ_bnd_temp[:,3] = 0.0

    TXYZ_bnd_temp = TXYZ_bnd_temp[TXYZ_bnd_temp[:,1]**2+TXYZ_bnd_temp[:,2]**2>1.0]
    
    TXYZ_bnd_0  = np.concatenate((TXYZ_bnd_0,TXYZ_bnd_temp),axis=0)

TXYZ_bnd_0 = TXYZ_bnd_0[:num_train_samples]

print(TXYZ_bnd_0.shape)



ax.scatter(TXYZ_bnd_0[:,1],TXYZ_bnd_0[:,2],TXYZ_bnd_0[:,3],alpha=0.6)

ax.legend()



plt.show()
