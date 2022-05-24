import numpy as np
import os

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


default_weight_name =f"./weights/default.h5"
pinn.save_weights(default_weight_name)


class AdaptiveWeightsCallBack(tf.keras.callbacks.Callback):

    def __init__(self,x_train,y_train,weights_list):
        super().__init__()

        self.x_train = x_train
        self.y_train = y_train
        self.loss_fn = MeanSquaredError()
        self.max_grad_res_list = list()
        self.mean_grad_ini_list = list()
        self.mean_grad_bnd_list = list()
        self.alpha = 0.9
        self.equ_weight =  weights_list[0]
        self.bnd_weight = weights_list[1]
        self.ini_weight = weights_list[2]
        



    def on_epoch_end(self, epoch, logs=None):




        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_weights)
            y_pred = self.model(self.x_train)
            loss_eqn = self.loss_fn(self.y_train[0],y_pred[0])
            loss_ini = self.loss_fn(self.y_train[1],y_pred[1])
            loss_bc = self.loss_fn(self.y_train[2],y_pred[2])

        gradients_eqn = tape.gradient(loss_eqn,self.model.trainable_weights)
        gradients_ini = tape.gradient(loss_ini,self.model.trainable_weights)
        gradients_bnd = tape.gradient(loss_bc,self.model.trainable_weights)

        for i in range(len(gradients_eqn) - 1):
            self.max_grad_res_list.append(tf.reduce_max(tf.abs(gradients_eqn[i])))
            self.mean_grad_ini_list.append(tf.reduce_max(tf.abs(gradients_ini[i])))
            self.mean_grad_bnd_list.append(tf.reduce_max(tf.abs(gradients_bnd[i])))

        self.max_grad_res = tf.reduce_max(tf.stack(self.max_grad_res_list))
        self.mean_grad_ini = tf.reduce_mean(tf.stack(self.mean_grad_ini_list))
        self.mean_grad_bnd = tf.reduce_mean(tf.stack(self.mean_grad_bnd_list))

        self.adaptive_constant_ini = self.max_grad_res / self.mean_grad_ini
        self.adaptive_constant_bnd = self.max_grad_res / self.mean_grad_bnd


        print("\n",self.adaptive_constant_ini,self.adaptive_constant_bnd,"\n")

        bnd_weight = (1.0-self.alpha)*self.bnd_weight + self.alpha*self.adaptive_constant_bnd
        ini_weight = (1.0-self.alpha)*self.ini_weight + self.alpha*self.adaptive_constant_ini

        print('New bnd weights:',self.bnd_weight,"new ini weight",self.ini_weight)
        K.set_value(self.bnd_weight,bnd_weight)
        K.set_value(self.ini_weight,ini_weight)


def main(epochs=800,sigma=40,train=True):
    """
    Using PINN to solve 1D linear diffusion equation
    """

    # number of training samples
    n_train_samples = 200000
    # number of test samples
    num_test_samples = 1000

    theta_i = 10.0
    theta_v = -10.0
    maxT = 2.0*abs(theta_v-theta_i)/sigma  # total time of voltammetric scan 
    maxX = 6.0 * np.sqrt(maxT)  # the max diffusion length 



    # create training input
    tx_eqn = np.random.rand(n_train_samples, 2)
    tx_eqn[...,0] = tx_eqn[...,0] * maxT         
    tx_eqn[..., 1] = tx_eqn[..., 1]*maxX               

    tx_ini = np.random.rand(n_train_samples, 2)  
    tx_ini[..., 0] = 0
    tx_ini[...,1] = tx_ini[...,1]*maxX                                    
    tx_bnd = np.random.rand(n_train_samples, 2)         
    tx_bnd[...,0] = tx_bnd[...,0] *maxT
    tx_bnd[..., 1] =  np.round(tx_bnd[..., 1])*maxX   


    # create training output
    c_eqn = np.zeros((n_train_samples, 1))              
    c_ini = np.ones((n_train_samples,1))    

    c_bnd=np.ones((n_train_samples,1))

    # apply Nernst equation in boundary conditions
    for i in range(n_train_samples):
        if tx_bnd[i,0] < maxT/2.0 and tx_bnd[i,1]<1e-3:
            c_bnd[i] = 1.0/(1.0+np.exp(-(theta_i-sigma*tx_bnd[i,0])))
        elif tx_bnd[i,0] >= maxT/2.0 and tx_bnd[i,1]<1e-3:
            c_bnd[i] = 1.0/(1.0+np.exp(-(theta_v+sigma*(tx_bnd[i,0]-maxT/2.0))))
        else:
            c_bnd[i] = 1.0
    

    # train the model using Adama
    x_train = [tx_eqn, tx_ini, tx_bnd]
    y_train = [ c_eqn,  c_ini,  c_bnd]

    pinn.load_weights(default_weight_name)

    weights_list = [K.variable(1),K.variable(1),K.variable(1)]
    pinn.compile(optimizer='adam',loss='mse',loss_weights=weights_list)

    if train:
        pinn.fit(x=x_train,y=y_train,epochs=epochs,verbose=2,callbacks=[AdaptiveWeightsCallBack(x_train,y_train,weights_list)])
        pinn.save_weights(f'./weights/weights sigma={sigma:.2E} epochs={epochs:.2E} n_train={n_train_samples}.h5')
    else:
        try:
            print(f'weights sigma={sigma:.2E} epochs={epochs:.2E} n_train={n_train_samples}.h5')
            pinn.load_weights(f'./weights/weights sigma={sigma:.2E} epochs={epochs:.2E} n_train={n_train_samples}.h5')
        except:
            print('weights does not exist\n start training')
            pinn.fit(x=x_train,y=y_train,epochs=epochs,verbose=2,callbacks=[AdaptiveWeightsCallBack(x_train,y_train,weights_list)])
            pinn.save_weights(f'./weights/weights sigma={sigma:.2E} epochs={epochs:.2E} n_train={n_train_samples}.h5')


    
    # predict c(t,x) distribution
    t_flat = np.linspace(0, maxT, num_test_samples)
    cv_flat = np.where(t_flat<maxT/2.0,theta_i-sigma*t_flat,theta_v+sigma*(t_flat-maxT/2.0))
    x_flat = np.linspace(0, maxX, num_test_samples) 
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    c = network.predict(tx, batch_size=num_test_samples)
    c = c.reshape(t.shape)
    x_i = x_flat[1]
    flux = -(c[1,:] - c[0,:])/x_i
    df = pd.DataFrame({'Potential':cv_flat,'Flux':flux})
    df.to_csv(f'Voltammogram scan sigma={sigma:.2E} epochs={epochs:.2E} n_train={n_train_samples}.csv',index=False)
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.plot(cv_flat,flux,label='PINN prediction')
    ax.axhline(-0.446*math.sqrt(sigma),label='R-S equation',ls='--',color='r')
    ax.axvline(-1.109,label='Expected Forward Scan Potential',ls='--',color='k')

    ax.set_xlabel('Potential, theta')
    ax.set_ylabel('Flux,J')
    ax.legend()
    fig.savefig(f'Voltmmorgam sigma={sigma:.2E} epochs={epochs:.2E} n_train={n_train_samples}.png')
    

    
    # plot u(t,x) distribution as a color-map
    fig,ax12 = plt.subplots(figsize=(8,9),nrows=2)
    ax = ax12[0]
    arrs = np.load("Concentration profile no call back.npy")
    t_no_callback,x_no_callback,c_no_callback = arrs[0], arrs[1], arrs[2]
    l1 = (c_no_callback-c)**1
    axes = ax.pcolormesh(t, x, l1, cmap='rainbow',shading='auto')

    ax.set_xlabel('T',fontsize='large',fontweight='bold')
    ax.set_ylabel('X',fontsize='large',fontweight='bold')
    ax.set_ylim(0,maxX*0.4)
    cbar = plt.colorbar(axes,pad=0.05, aspect=10,ax=ax)
    cbar.set_label(r'$C_{Net\ A}-C_{Net\ B}$',fontsize='medium',fontweight='bold')
    #axes.set_clim(0.0, 1)

    ax = ax12[1]
    df = pd.read_csv("No callback Voltammogram scan sigma=4.00E+01 epochs=1.00E+01 n_train=200000.csv")
    df.plot(x='Potential',y='Flux',ax=ax,color='b',lw=3,alpha=0.8,label='PINN, Net A')
    df = pd.read_csv(f'Voltammogram scan sigma={sigma:.2E} epochs={epochs:.2E} n_train={n_train_samples}.csv')
    df.plot(x='Potential',y='Flux',ax=ax,color='k',lw=3,alpha=0.8,label='PINN, Net B')
    df_fd = pd.read_csv('FD-sigma=40.txt',header=None)
    df_fd.plot(x=0,y=1,ax=ax,color='r',ls='--',lw=3,alpha=0.9,label='Finite Difference')
    #ax.axhline(-0.446*math.sqrt(sigma),label='R-S Equation',ls='--',color='k',lw=3)
    ax.plot([-5,5],[-0.446*math.sqrt(sigma),-0.446*math.sqrt(sigma)],label='R-S Equation',ls='--',color='k',lw=3)
    ax.annotate("", xy=(2.5, -0.5), xytext=(10, -0.5),arrowprops=dict(facecolor='black', shrink=0.05))
    ax.set_xlabel(r'Potential,$\theta$')
    ax.set_ylabel(r'Flux, $J$')
    ax.legend(fontsize=15)


    fig.tight_layout()
    fig.text(0.05,0.92,'A)',fontsize=30)
    fig.text(0.05,0.45,'B)',fontsize=30)
    fig.savefig(f'Solving Diffusion equation and compare sigma={sigma:.2E} epochs={epochs:.2E} n_train={n_train_samples}.png',dpi=250,bbox_inches='tight')

    
    plt.close('all')
    tf.keras.backend.clear_session()



if __name__ == "__main__":


    for sigma in [40]:
        main(epochs=10,sigma=sigma,train=False)