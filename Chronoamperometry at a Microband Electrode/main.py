from re import I
import re
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib.pinn import PINN
from lib.network import Network
import pandas as pd
from matplotlib.patches import Rectangle
import tensorflow as tf 
import os
from scipy.special import pbwa

linewidth = 4
fontsize = 20

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs

errors = list()

D =  1.0
# build a core network model
network = Network.build()
network.summary()
# build a PINN model
pinn = PINN(network, D).build()
pinn.compile(optimizer='Adam',loss='mse')

default_weight_name = "./weights/default.h5"
pinn.save_weights(default_weight_name)

def main(epochs=50,batch_size=1000,maxT=1.0,train=True,saving_directory = './Data',rElectrode=0.5,loss_weights = [1,1,1,1,1,1,1,1],alpha=1.0,lambda_ratio=2.0):
    """
    epochs: number of epoch for the training 
    maxT: dimensionless scan time 
    train: If true, always train a new neural network. Otherwise just 
    alpha: the decay of learning rate. If alpha = 1.0, no decay of learning rate. alpha<1 if decay of learning rate.

    
    """
    if not os.path.exists(saving_directory):
        os.mkdir(saving_directory)

    # define the learning rate scheduler 
    def scheduler(epoch,lr):
        
        if epoch < 400:
            return lr
        else:
            return lr*alpha


    # number of training samples
    num_train_samples = int(2e6)
    # number of test samples
    num_test_samples = 100

    batch_size = batch_size




    file_name = f'maxT={maxT:.2E} epochs={epochs:.2E} n_train={num_train_samples:.2E} batch_size = {batch_size:.2E}'




    stall_T = 0.1

    maxX = lambda_ratio* np.sqrt(maxT)  # the max diffusion length 



    # the training feature enforcing Fick's second law of diffusion in 2D
    txy_dmn0 = np.random.rand(num_train_samples,3)
    txy_dmn0[...,0] = txy_dmn0[...,0] * maxT
    txy_dmn0[...,1] = txy_dmn0[...,1] *maxX 
    txy_dmn0[...,2] = txy_dmn0[...,2] *(maxX+rElectrode)



    # boundary 1
    txy_bnd1 = np.random.rand(num_train_samples,3)
    txy_bnd1[...,0] = txy_bnd1[...,0] * maxT
    txy_bnd1[...,1] = 0.0 
    txy_bnd1[...,2] = txy_bnd1[...,2] * maxX + rElectrode

    # boundary 2 
    txy_bnd2 = np.random.rand(num_train_samples,3)
    txy_bnd2[...,0] = txy_bnd2[...,0] * maxT
    txy_bnd2[...,1] = txy_bnd2[...,1] * maxX 
    txy_bnd2[...,2] = (maxX + rElectrode)

    # boundary 3 
    txy_bnd3 = np.random.rand(num_train_samples,3)
    txy_bnd3[...,0] = txy_bnd3[...,0] * maxT
    txy_bnd3[...,1] = maxX
    txy_bnd3[...,2] = txy_bnd3[...,2] * (maxX + rElectrode) 

    #boundary 4
    txy_bnd4 = np.random.rand(num_train_samples,3)
    txy_bnd4[...,0] = txy_bnd4[...,0] * maxT
    txy_bnd4[...,1] = txy_bnd4[...,1] * maxX
    txy_bnd4[...,2] = 0.0

    # initial condition 
    txy_ini = np.random.rand(num_train_samples,3)
    txy_ini[...,0]  = 0.0
    txy_ini[...,1] = txy_ini[...,1] *maxX 
    txy_ini[...,2] = txy_ini[...,2] *(maxX+rElectrode) 


    # boundary 0, the electrode surface 
    txy_bnd0 = np.random.rand(num_train_samples,3) 
    txy_bnd0[...,0] = txy_bnd0[...,0] * maxT
    txy_bnd0[...,1] = 0.0
    txy_bnd0[...,2] = txy_bnd0[...,2] * rElectrode


    # creating training output 

    c_dmn = np.zeros((num_train_samples,1))


    c_bnd0 = np.zeros((num_train_samples,1))
    c_bnd1 = np.zeros((num_train_samples,1))
    c_bnd2 = np.ones((num_train_samples,1))
    c_bnd3 = np.ones((num_train_samples,1))
    c_bnd4 = np.zeros((num_train_samples,1))

    c_ini = np.ones((num_train_samples,1))

    for i in range(num_train_samples):
        if txy_bnd0[i,0] < stall_T:
            c_bnd0[i] = 1.0
        else:
            c_bnd0[i] = 0.0
            

    






    x_train = [txy_dmn0,txy_ini,txy_bnd1,txy_bnd4,txy_bnd0,txy_bnd2,txy_bnd3]
    y_train = [c_dmn,c_ini,c_bnd1,c_bnd4,c_bnd0,c_bnd2,c_bnd3]






    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    pinn.load_weights(default_weight_name)
    # If train = True, fit the NN, else, use saved weights 
    if train:
        pinn.fit(x=x_train,y=y_train,epochs=epochs,batch_size =batch_size,verbose=2,callbacks=[callback])
        pinn.save_weights(f'./weights/weights {file_name}.h5')
    else:
        try:
            pinn.load_weights(f'./weights/weights {file_name}.h5')
        except:
            print('Weights does not exist\nStart training')
            print(f'./weights/weights {file_name}.h5')
            pinn.fit(x=x_train,y=y_train,epochs=epochs,batch_size =batch_size,verbose=2)
            pinn.save_weights(f'./weights/weights {file_name}.h5')


    """
    time_sects = np.linspace(0,maxT,num=11)

    for index,time_sect in enumerate(time_sects):
        txy_test = np.zeros((int(num_test_samples**2),3))
        txy_test[...,0] = time_sect
        x_flat = np.linspace(0,maxX,num_test_samples)
        y_flat = np.linspace(0,maxX+rElectrode,num_test_samples)
        x,y = np.meshgrid(x_flat,y_flat)
        txy_test[...,1] = x.flatten()
        txy_test[...,2] = y.flatten()

        c = network.predict(txy_test)

        c = c.reshape(x.shape)

        plt.figure()
        plt.pcolormesh(x,y,c,shading='auto')
        plt.title(f'time={time_sect:.2f}')
        cbar = plt.colorbar(pad=0.05, aspect=10)
        cbar.set_label('c(t,x,y)')
        cbar.mappable.set_clim(0, 1)
        plt.savefig(f'{saving_directory}/{file_name} t={index}.png')
        plt.close('all')
    """
    
    time_steps = np.linspace(0.0,maxT*1.5,num=int(num_test_samples*10))


    fluxes = np.zeros_like(time_steps)


    for index, time_step in enumerate(time_steps):
        x_flat = np.linspace(0,1e-5,25)
        y_flat = np.linspace(0,rElectrode,num=num_test_samples)
        txy_test = np.zeros((int(len(x_flat)*len(y_flat)),3))
        txy_test[...,0] = time_step
        x,y = np.meshgrid(x_flat,y_flat)
        txy_test[...,1] = x.flatten()
        txy_test[...,2] = y.flatten()

        x_i = x_flat[1] -x_flat[0]
        y_i = y_flat[1] - y_flat[0]
        c = network.predict(txy_test)
        c = c.reshape(x.shape)

        J = - sum((c[:,5] - c[:,0])/ (5*x_i) * y_i)



        fluxes[index] = J

    



    #Aoki_fluxes = 1.0/(np.sqrt(np.pi*time_steps)) +0.97 - 1.10*np.exp(-9.9/np.abs(np.log(12.37*time_steps)))
    #Aoki_fluxes = - Aoki_fluxes
    df = pd.DataFrame({'Time':time_steps,'Flux':fluxes,})


    df.to_csv(f'{saving_directory}/{file_name}.csv',index=False)
    
    
    
    time_sects = [maxT/2.0]

    for index,time_sect in enumerate(time_sects):
        txy_test = np.zeros((int(num_test_samples**2),3))
        txy_test[...,0] = time_sect
        x_flat = np.linspace(0,maxX,num_test_samples)
        y_flat = np.linspace(0,maxX+rElectrode,num_test_samples)
        x,y = np.meshgrid(x_flat,y_flat)
        txy_test[...,1] = x.flatten()
        txy_test[...,2] = y.flatten()

        c = network.predict(txy_test)

        c = c.reshape(x.shape)

        fig,ax12 = plt.subplots(figsize=(8,9),nrows=2)

        ax = ax12[0]
        axes = ax.pcolormesh(x, y, c,shading='auto')
        ax.set_xlabel('X',fontsize='large',fontweight='bold')
        ax.set_ylabel('Y',fontsize='large',fontweight='bold')

        cbar = plt.colorbar(axes,pad=0.05, aspect=10,ax=ax)
        cbar.set_label('C(T=0.5,X,Y)',fontsize='large',fontweight='bold')
        axes.set_clim(0.0, 1)
        ax.contour(x,y,c,10,colors='white',linewidths=2,alpha=0.5)
        ax.add_patch(Rectangle((-0.1,0),0.1,0.5,facecolor='r',edgecolor='k'))
        ax.add_patch(Rectangle((-0.1,0.5),0.1,maxX,facecolor='white',edgecolor='k'))
        #ax.set_ylim(0,4)
        ax.set_xlim(-0.1,maxX)
        plt.subplots_adjust(hspace=0.35)
        ax = ax12[1]
        df = pd.read_csv(f'{saving_directory}/{file_name}.csv')
        df['Flux'] /= rElectrode
        #from scipy.signal import savgol_filter  
        #df['Flux'] = savgol_filter(df['Flux'],15,3)
        df.plot(x='Time',y='Flux',ax=ax,color='b',lw=3,alpha=0.8,label='PINN, $m=4000$')

        df = pd.read_csv(f'./Data/maxT=1.00E+00 epochs=1.00E+02 n_train=2.00E+06 batch_size = 2.40E+01.csv')
        df['Flux'] /= rElectrode
        df.plot(x='Time',y='Flux',ax=ax,color='k',lw=3,alpha=0.8,label='PINN, $m=24$')

        PINN_peak_current = df['Flux'].min()
        def U(a,x):
            return pbwa(a,x)[0]

        T = np.linspace(1e-4,maxT*1.5-0.1,num=500)
        J_band = (np.pi*T)**(-0.5) + 1.0 - ((2.0**0.75)/np.pi)*(T**0.75)*np.exp(-1.0/(8.0*T)) * (U(2.0,(2.0*T)**(-0.5)))
        ax.plot(T+0.1,-J_band,color='r',ls='--',lw=3,label='Analytical Equation')

        ax.annotate('Extrapolation\nStarts',xy=(1,-1.5),xytext=(0.25,-5.0),arrowprops=dict(facecolor='black', shrink=0.05))

        ax.set_xlabel(r'Time,$T$',fontsize='large',fontweight='bold')
        ax.set_ylabel(r'Flux, $J$',fontsize='large',fontweight='bold')
        ax.legend(fontsize=15)


        fig.text(0.05,0.92,'A)',fontsize=30)
        fig.text(0.05,0.45,'B)',fontsize=30)
        fig.tight_layout()
        fig.savefig(f'{saving_directory}/{file_name} t={index}.png',bbox_inches='tight',dpi=200)

        plt.close('all')
    








    tf.keras.backend.clear_session()




if __name__ == "__main__":



    
    #batch_sizes = [24,32,64,100,200,400,800,1000,4000,8000,16000,32000,64000,64000*2,64000*3,64000*4,64000*5]
    batch_sizes = [4000]

    for batch_size in batch_sizes :

        main(epochs=100,maxT=1.0,batch_size=batch_size,train=False,rElectrode=0.5)



      