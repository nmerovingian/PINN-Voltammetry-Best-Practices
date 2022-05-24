import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from lib.pinn import PINN
from lib.network import Network
import pandas as pd
import math
import tensorflow as tf
from math import isclose

linewidth = 4
fontsize = 20
font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs


def main(epochs=800,maxT=1.0,train=True):
    """
    Using PINN to solve 1D linear diffusion equation
    """

    # number of training samples
    n_train_samples = 200000
    # number of test samples
    num_test_samples = 1000


    maxT = 1.0  # total time of voltammetric scan 
    stall_time = maxT*0.1
    maxX = 4.0 * np.sqrt(maxT) + 1.0  # the max diffusion length 

    # build a core network model
    network = Network.build()
    network.summary()
    # build a PINN model
    pinn = PINN(network).build()

    # create training input
    tx_eqn = np.random.rand(n_train_samples, 2)
    tx_eqn[...,0] = tx_eqn[...,0] * maxT         
    tx_eqn[..., 1] = tx_eqn[..., 1]*maxX + 1.0




    tx_ini = np.random.rand(n_train_samples, 2)  
    tx_ini[..., 0] = 0
    tx_ini[...,1] = tx_ini[...,1]*maxX + 1.0                                   
    tx_bnd0 = np.random.rand(n_train_samples, 2)         
    tx_bnd0[...,0] = tx_bnd0[...,0] *maxT
    tx_bnd0[..., 1] =  1.0

    tx_bnd1 = np.random.rand(n_train_samples, 2)         
    tx_bnd1[...,0] = tx_bnd1[...,0] *maxT
    tx_bnd1[..., 1] =  maxX+1.0
    

    # create training output
    c_eqn = np.zeros((n_train_samples, 1))              
    c_ini = np.ones((n_train_samples,1))    

    c_bnd0=np.ones((n_train_samples,1))
    c_bnd1 = np.ones((n_train_samples,1))

    # apply Nernst equation in boundary conditions
    for i in range(n_train_samples):
        c_ini[i] = tx_ini[i,1] * 1.0
        c_bnd1[i] = (maxX+1.0) 
        if tx_bnd0[i,0] > stall_time:
            c_bnd0[i] = 0.0
    

    # train the model using Adama
    x_train = [tx_eqn, tx_ini, tx_bnd0,tx_bnd1]
    y_train = [ c_eqn,  c_ini,  c_bnd0,c_bnd1]

    pinn.compile(optimizer='adam',loss='mse')

    if train:
        pinn.fit(x=x_train,y=y_train,epochs=epochs,verbose=1)
        pinn.save_weights(f'./weights/weights maxT={maxT:.2E} epochs={epochs:.2E} n_train={n_train_samples}.h5')
    else:
        try:
            print(f'weights maxT={maxT:.2E} epochs={epochs:.2E} n_train={n_train_samples}.h5')
            pinn.load_weights(f'./weights/weights maxT={maxT:.2E} epochs={epochs:.2E} n_train={n_train_samples}.h5')
        except:
            print('weights does not exist\n start training')
            pinn.fit(x=x_train,y=y_train,epochs=epochs,verbose=1)
            pinn.save_weights(f'./weights/weights maxT={maxT:.2E} epochs={epochs:.2E} n_train={n_train_samples}.h5')


    

    t_flat = np.linspace(0, maxT*3, num_test_samples)

    x_flat = np.linspace(1.0, maxX+1.0, num_test_samples) 
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    u = network.predict(tx, batch_size=num_test_samples)
    c = u/(x.flatten().reshape(-1,1))
    u = u.reshape(t.shape)
    c = c.reshape(t.shape)

    x_i = x_flat[1]-x_flat[0]
    flux = -(c[1,:] - c[0,:])/x_i
    df = pd.DataFrame({'Time':t_flat,'Flux':flux})
    df.to_csv(f'Chronoamperogram scan maxT={maxT:.2E} epochs={epochs:.2E} n_train={n_train_samples}.csv',index=False)
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.plot(t_flat,flux,label='PINN prediction')
    T = np.linspace(8e-3,maxT*3-stall_time,num=200)
    J_cottrell = -(1.0+np.sqrt(T*np.pi))/(np.sqrt(T*np.pi))
    ax.plot(T+stall_time,J_cottrell,label='Cottrell Equation')
    ax.set_xlabel('Time, T')
    ax.set_ylabel('Flux,J')
    ax.legend()
    fig.savefig(f'Chronoamperogram maxT={maxT:.2E} epochs={epochs:.2E} n_train={n_train_samples}.png')
    
    
    # plot u(t,x) distribution as a color-map
    fig,ax12 = plt.subplots(figsize=(8,13.5),nrows=3)
    ax = ax12[0]
    arrs = np.load("Chrono Sphere.npy")
    axes = ax.pcolormesh(arrs[0], arrs[1], arrs[2], cmap='viridis',shading='auto')
    ax.set_xlabel('T',fontsize='large',fontweight='bold')
    ax.set_ylabel('R',fontsize='large',fontweight='bold')
    ax.annotate('Extrapolation\nStarts',xy=(1,1),xytext=(0.6,2),arrowprops=dict(facecolor='black', shrink=0.05))
    cbar = plt.colorbar(axes,pad=0.05, aspect=10,ax=ax)
    cbar.set_label(r'$C_{Net\ A}(T,R)$',fontsize='large',fontweight='bold')

    axes.set_clim(0.0, 1)

    ax = ax12[1]
    axes = ax.pcolormesh(t, x, u, cmap='inferno',shading='auto')

    ax.set_xlabel('T',fontsize='large',fontweight='bold')
    ax.set_ylabel('R',fontsize='large',fontweight='bold')
    ax.set_ylim(1,4)
    ax.annotate('Extrapolation\nStarts',xy=(1,1),xytext=(0.6,2),arrowprops=dict(facecolor='white', shrink=0.05))

    cbar = plt.colorbar(axes,pad=0.05, aspect=10,ax=ax)
    cbar.set_label(r'$U_{Net\ B}$(T,R)',fontsize='large',fontweight='bold')

    axes.set_clim(0.0, 4)

    ax = ax12[2]

    df = pd.read_csv("Chrno radial.csv")
    df.plot(x='Time',y='Flux',ax=ax,color='b',lw=3,alpha=0.8,label='PINN, Net A')
    
    df = pd.read_csv(f'Chronoamperogram scan maxT={maxT:.2E} epochs={epochs:.2E} n_train={n_train_samples}.csv')
    df.plot(x='Time',y='Flux',ax=ax,color='k',lw=4,alpha=0.8,label='PINN, Net B')

    ax.plot(T+stall_time,J_cottrell,label='Analytical Equation',lw=3,color='r',ls='--')
    ax.annotate('Extrapolation\nStarts',xy=(1,-1.8),xytext=(0.6,-5),arrowprops=dict(facecolor='black', shrink=0.05))
    ax.set_xlabel(r'T',fontsize='large',fontweight='bold')
    ax.set_ylabel(r'Flux, $J$',fontsize='large',fontweight='bold')
    ax.legend(fontsize=15)


    fig.tight_layout()
    fig.text(0.05,0.95,'A)',fontsize=30)
    fig.text(0.05,0.62,'B)',fontsize=30)
    fig.text(0.05,0.29,'C)',fontsize=30)
    fig.savefig(f'Solving Diffusion equation and compare maxT={maxT:.2E} epochs={epochs:.2E} n_train={n_train_samples}.png',dpi=250,bbox_inches='tight')

    
    plt.close('all')
    tf.keras.backend.clear_session()



    


    plt.close('all')
    tf.keras.backend.clear_session()
if __name__ == "__main__":
    # set train to True if you prefer to start fresh training.


    main(epochs=400,maxT=1.0,train=False)