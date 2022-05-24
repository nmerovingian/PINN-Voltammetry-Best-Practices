import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from lib.pinn import PINN
from lib.network import Network
import pandas as pd
import math
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

import tensorflow.keras.backend as K
linewidth = 4
fontsize = 20

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs

# build a core network model
network = Network.build()
network.summary()
# build a PINN model
pinn = PINN(network).build()
pinn.compile(optimizer='adam',loss='mse')

default_weight_name =f"./weights/default.h5"
pinn.save_weights(default_weight_name)

CV_Potential = list()
CV_Current =  list()

t_s = list()

x_s = list()

c_s = list()


fig,axes = plt.subplots(nrows=2,figsize=(8,9))
fig.tight_layout()
def main(epochs=800,sigma=40,maxT=1.0,startT=0.0,dT=1.0,train=True,CV_Potential=None,CV_Current=None):
    """
    Using PINN to solve 1D linear diffusion equation
    """

    # number of training samples
    n_train_samples = int(1e6*(dT/maxT))
    # number of test samples
    num_test_samples = int(1e3)




    maxX = 3 * np.sqrt(maxT)  # the max diffusion length 


    file_name = f'sigma={sigma:.2E} startT={startT:.2E} epochs={epochs:.2E} n_train={n_train_samples}'

    # create training input
    tx_eqn = np.random.rand(n_train_samples, 2)
    tx_eqn[...,0] = tx_eqn[...,0] * dT + startT         
    tx_eqn[..., 1] = tx_eqn[..., 1]*maxX + 1.0

    x_eqn1 = 2.0/tx_eqn[..., 1]            

    tx_ini = np.random.rand(n_train_samples, 2)  
    tx_ini[..., 0] = startT
    tx_ini[...,1] = tx_ini[...,1]*maxX + 1.0

    tx_bnd = np.random.rand(n_train_samples, 2)         
    tx_bnd[...,0] = tx_bnd[...,0] *dT + startT
    tx_bnd[..., 1] =  1.0

    tx_bnd1 = np.random.rand(n_train_samples, 2)         
    tx_bnd1[...,0] = tx_bnd1[...,0] *dT + startT
    tx_bnd1[..., 1] =  maxX+1.0


    # create training output
    c_eqn = np.zeros((n_train_samples, 1))
    if startT < dT:              
        c_ini = np.ones((n_train_samples,1))
    else:
        c_ini = network.predict(tx_ini)    

    c_bnd=np.zeros((n_train_samples,1))
    c_bnd1=np.ones((n_train_samples,1))



    # apply Nernst equation in boundary conditions
    for i in range(n_train_samples):
        if tx_bnd[i,0] < maxT/2.0:
            c_bnd[i] = 1.0/(1.0+np.exp(-(theta_i-sigma*tx_bnd[i,0])))
        elif tx_bnd[i,0] >= maxT/2.0:
            c_bnd[i] = 1.0/(1.0+np.exp(-(theta_v+sigma*(tx_bnd[i,0]-maxT/2.0))))
        else:
            c_bnd[i] = 1.0
    

    # train the model using Adama
    x_train = [tx_eqn,x_eqn1, tx_ini, tx_bnd,tx_bnd1]
    y_train = [ c_eqn,  c_ini,  c_bnd,c_bnd1]

    weights_list = [K.variable(1),K.variable(1),K.variable(1),K.variable(1)]




    pinn.compile(optimizer='adam',loss='mse',loss_weights=weights_list)

    pinn.load_weights(default_weight_name)
    if train:
        pinn.fit(x=x_train,y=y_train,epochs=epochs,verbose=2)
        pinn.save_weights(f'./weights/weights {file_name}')
    else:
        try:
            print(f'weights {file_name}.h5')
            pinn.load_weights(f'./weights/weights {file_name}.h5')
        except:
            print(f'weights {file_name}.h5')
            print('weights does not exist\n start training')
            pinn.fit(x=x_train,y=y_train,epochs=epochs,verbose=2)
            pinn.save_weights(f'./weights/weights {file_name}.h5')


    

    t_flat = np.linspace(startT, startT+dT, num_test_samples)
    cv_flat = np.where(t_flat<maxT/2.0,theta_i-sigma*t_flat,theta_v+sigma*(t_flat-maxT/2.0))
    x_flat = np.linspace(1.0, maxX, num_test_samples) 
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    c = network.predict(tx, batch_size=num_test_samples)
    c = c.reshape(t.shape)

    x_i = x_flat[1] - x_flat[0]
    flux = -(c[1,:] - c[0,:])/x_i






    CV_Potential += list(cv_flat)
    CV_Current += list(flux)

    
    t_s.append(t)
    x_s.append(x)
    c_s.append(c)
    


if __name__ == "__main__":
    # set train to True if you prefer to start fresh training.

    
    num_time_steps = 10
    epochs = 150
    for sigma in [0.5]:

        theta_i = 10.0
        theta_v = -10.0
        maxT = 2.0*abs(theta_v-theta_i)/sigma  # total time of voltammetric scan
        time_steps = np.linspace(0,maxT,endpoint=False,num=num_time_steps)
        if num_time_steps == 1:
            dT = maxT
        else:
            dT = time_steps[1] - time_steps[0]
        for i in range(len(time_steps)): 


            main(epochs=epochs,sigma=sigma,train=False,maxT=maxT,startT=time_steps[i],dT=dT,CV_Potential=CV_Potential,CV_Current=CV_Current)
            #main(epochs=150,sigma=sigma,train=True,maxT=maxT,startT=0.0,dT=maxT,CV_Potential=CV_Potential,CV_Current=CV_Current)

            df = pd.DataFrame({'Potential':CV_Potential,'Flux':CV_Current})

            df.to_csv(f'sigma={sigma:.2E} time_stepes = {num_time_steps} epochs = {epochs}.csv',index=False)


        ax = axes[1]
        df.plot(x='Potential',y='Flux',color='b',lw=3,alpha=0.8,ax=ax,label='PINN w/ Seq to Seq')
        sqe2seq_peak_flux = df.iloc[:,1].max()
        df = pd.read_csv("sigma=5.00E-01 time_stepes = 1 epochs = 150.csv")
        no_sqe2seq_peak_flux = df.iloc[:,1].max()
        df.plot(x='Potential',y='Flux',ax=ax,color='k',lw=3,alpha=0.8,label='PINN w/o Seq to Seq')
        df_FD = pd.read_csv('FD sigma=0.5.txt',header=None)
        df_FD.plot(x=0,y=1,ax=ax,label='Finite Difference',color='r',ls='--')
        fd_peak_flux = df_FD.iloc[:,1].max()
        ax.annotate("", xy=(2.5, -0.5), xytext=(10, -0.5),arrowprops=dict(facecolor='black', shrink=0.05))
        ax.set_xlabel(r'Potential, $\theta$')
        ax.set_ylabel(r'Flux,$J$')
        ax.legend(fontsize=(12))




        ax = axes[0]

        for i in range(len(t_s)):
            mappable = ax.pcolormesh(t_s[i], x_s[i], c_s[i], cmap='viridis',shading='auto')
            mappable.set_clim(0,1)

        ax.set_xlabel('T',fontsize='large',fontweight='bold')
        ax.set_ylabel('R',fontsize='large',fontweight='bold')
        ax.set_ylim(1,10)
        cbar = plt.colorbar(mappable,pad=0.05, aspect=10,ax=ax)
        cbar.set_label(r'$C(T,R)$',fontsize='medium',fontweight='bold')


        CV_Potential.clear()
        CV_Current.clear()

        fig.text(0.01,0.95,'A)',fontsize=30)
        fig.text(0.01,0.45,'B)',fontsize=30)
        fig.savefig(f'sigma={sigma:.2E} time_stepes = {num_time_steps} epochs = {epochs}.csv.png',bbox_inches='tight',dpi=250)


