import sys
import numpy as np 
import os
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tensorflow.keras import callbacks
from tensorflow.python.keras.callbacks import Callback
from lib.pinn import PINN
from lib.network import Network
import pandas as pd
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
"""
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7680)])
"""
from tensorflow.keras.callbacks import LearningRateScheduler

linewidth = 4
fontsize = 15

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs




# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    network = Network.build()
    network.summary()
    pinn = PINN(network).build()
    pinn.compile(optimizer='Adam',loss='mse')

default_weight_name =f"./weights/default.h5"
pinn.save_weights(default_weight_name)



# Estimate the flux
J_0 = list()
J_1 = list()
J_2 = list()
Time_list = list()
Potential_list = list()


def prediction(epochs=10,sigma=40,startT=0.0,endT=1.0,durT=1.0,maxT=1.0,Xe=1.0,Ye=1.0,Ze=1.0,num_train_samples=1e6,batch_size = 1e4, alpha=0.98,initial_weights = None,train=True,saving_directory="./Data",):
    # prefix or suffix of files 
    file_name = f'sigma={sigma:.2E} epochs={epochs:.2E} n_train={num_train_samples:.2E} starT={startT}'

    def schedule(epoch,lr):
        # a learning rate scheduler 
        if epoch<=200:
            return lr 
        else:
            lr *= alpha
            return lr
    # saving directory is where data(voltammogram, concentration profile etc is saved)
    if not os.path.exists(saving_directory):
        os.mkdir(saving_directory)
    
    #weights folder is where weights is saved
    if not os.path.exists('./weights'):
        os.mkdir('./weights')


    theta_i = 10.0 # start/end potential of scan 
    theta_v = -10.0 # reverse potential of scan
    maxT = 2.0*abs(theta_v-theta_i)/sigma  # total time of voltammetric scan 

    
    # number of training samples
    num_train_samples = int(num_train_samples)
    # number of test samples
    num_test_samples = 100
    # batch size
    batch_size = int(batch_size)

    Xsim = 4.0*np.sqrt(maxT) + Xe
    Ysim = 4.0*np.sqrt(maxT) + Ye
    Zsim = 4.0*np.sqrt(maxT) + Ze

    print('Generating electrode surface training')
    """
    # electrode surface
    TXYZ_e_surf_0 = np.random.rand(num_train_samples,4)
    TXYZ_e_surf_0[:,0] = TXYZ_e_surf_0[:,0]*durT + startT
    TXYZ_e_surf_0[:,1] *= Xe
    TXYZ_e_surf_0[:,2] *= Ye
    TXYZ_e_surf_0[:,3] = Ze

    TXYZ_e_surf_1 = np.random.rand(num_train_samples,4)
    TXYZ_e_surf_1[:,0] = TXYZ_e_surf_1[:,0]*durT + startT
    TXYZ_e_surf_1[:,1] *= Xe
    TXYZ_e_surf_1[:,2] = Ye
    TXYZ_e_surf_1[:,3] *= Ze

    TXYZ_e_surf_2 = np.random.rand(num_train_samples,4)
    TXYZ_e_surf_2[:,0] = TXYZ_e_surf_2[:,0]*durT + startT
    TXYZ_e_surf_2[:,1] = Xe
    TXYZ_e_surf_2[:,2] *= Ye
    TXYZ_e_surf_2[:,3] *= Ze

    print('Generating electrode training ')
    # inside the electrode
    TXYZ_e = np.random.rand(num_train_samples,4)
    TXYZ_e[:,0] = TXYZ_e[:,0]*durT + startT
    TXYZ_e[:,1] *= Xe
    TXYZ_e[:,2] *= Ye
    TXYZ_e[:,3] *= Ze

    print('Generating diffusion training ')

    TXYZ_eqn0 = np.random.rand(num_train_samples,4)
    TXYZ_eqn0[:,0] = TXYZ_eqn0[:,0]*durT + startT
    TXYZ_eqn0[:,1] *= Xsim 
    TXYZ_eqn0[:,2] =  TXYZ_eqn0[:,2] * (Ysim-Ye) + Ye 
    TXYZ_eqn0[:,3] *= Zsim

    TXYZ_eqn1 = np.random.rand(num_train_samples,4)
    TXYZ_eqn1[:,0] = TXYZ_eqn1[:,0]*durT + startT
    TXYZ_eqn1[:,1] = TXYZ_eqn1[:,1] * (Xsim-Xe) + Xe 
    TXYZ_eqn1[:,2] *= Ysim  
    TXYZ_eqn1[:,3] *= Zsim

    TXYZ_eqn2 = np.random.rand(num_train_samples,4)
    TXYZ_eqn2[:,0] = TXYZ_eqn2[:,0]*durT + startT
    TXYZ_eqn2[:,1] = TXYZ_eqn2[:,1]*(Xsim-Xe) + Xe 
    TXYZ_eqn2[:,2] *= Ysim 
    TXYZ_eqn2[:,3] *= Zsim

    TXYZ_eqn3 = np.random.rand(num_train_samples,4)
    TXYZ_eqn3[:,0] = TXYZ_eqn3[:,0]*durT + startT
    TXYZ_eqn3[:,1] *= Xsim 
    TXYZ_eqn3[:,2] *= Ysim  
    TXYZ_eqn3[:,3] = TXYZ_eqn3[:,3]*(Zsim-Ze) + Ze

    TXYZ_eqn4 = np.random.rand(num_train_samples,4)
    TXYZ_eqn4[:,0] = TXYZ_eqn4[:,0]*durT + startT
    TXYZ_eqn4[:,1] *= Xsim 
    TXYZ_eqn4[:,2] = TXYZ_eqn4[:,2] * (Ysim -Ye) + Ye
    TXYZ_eqn4[:,3] *= Zsim

    TXYZ_eqn5 = np.random.rand(num_train_samples,4)
    TXYZ_eqn5[:,0] *= TXYZ_eqn5[:,0]*durT + startT
    TXYZ_eqn5[:,1] *= Xsim 
    TXYZ_eqn5[:,2] *= Ysim  
    TXYZ_eqn5[:,3] = TXYZ_eqn5[:,3] * (Zsim-Ze) + Ze



    print('Generating outer training ')

    # Simulation Surface
    # Front surface
    TXYZ_sim_surf_0 = np.random.rand(num_train_samples,4)
    TXYZ_sim_surf_0[:,0] = TXYZ_sim_surf_0[:,0]*durT + startT
    TXYZ_sim_surf_0[:,1] = TXYZ_sim_surf_0[:,1] * (Xsim-Xe) + Xe
    TXYZ_sim_surf_0[:,2] = 0.0
    TXYZ_sim_surf_0[:,3] *= Zsim

    TXYZ_sim_surf_1 = np.random.rand(num_train_samples,4)
    TXYZ_sim_surf_1[:,0] = TXYZ_sim_surf_1[:,0]*durT + startT
    TXYZ_sim_surf_1[:,1] *= Xsim 
    TXYZ_sim_surf_1[:,2] = 0.0 
    TXYZ_sim_surf_1[:,3] = TXYZ_sim_surf_1[:,3] * (Zsim-Ze) + Ze

    # Left surface
    TXYZ_sim_surf_2 = np.random.rand(num_train_samples,4)
    TXYZ_sim_surf_2[:,0] = TXYZ_sim_surf_1[:,0]*durT + startT 
    TXYZ_sim_surf_2[:,1] = 0.0
    TXYZ_sim_surf_2[:,2] = TXYZ_sim_surf_2[:,2] * (Ysim-Ye) + Ye
    TXYZ_sim_surf_2[:,3] *= Zsim 

    TXYZ_sim_surf_3 = np.random.rand(num_train_samples,4)
    TXYZ_sim_surf_3[:,0] = TXYZ_sim_surf_3[:,0]*durT + startT
    TXYZ_sim_surf_3[:,1] = 0.0
    TXYZ_sim_surf_3[:,2] *= Ysim 
    TXYZ_sim_surf_3[:,3] = TXYZ_sim_surf_3[:,3] * (Zsim-Ze) + Ze

    # Bottom surface
    TXYZ_sim_surf_4 = np.random.rand(num_train_samples,4)
    TXYZ_sim_surf_4[:,0] = TXYZ_sim_surf_4[:,0]*durT + startT
    TXYZ_sim_surf_4[:,1] = TXYZ_sim_surf_4[:,1] * (Xsim-Xe) + Xe
    TXYZ_sim_surf_4[:,2] *= Ysim 
    TXYZ_sim_surf_4[:,3] = 0.0

    TXYZ_sim_surf_5 = np.random.rand(num_train_samples,4)
    TXYZ_sim_surf_5[:,0] = TXYZ_sim_surf_5[:,0]*durT + startT
    TXYZ_sim_surf_5[:,1] *= Xsim 
    TXYZ_sim_surf_5[:,2] = TXYZ_sim_surf_5[:,2] * (Ysim-Ye) + Ye
    TXYZ_sim_surf_5[:,3] = 0.0


    # Back surface
    TXYZ_sim_surf_6 = np.random.rand(num_train_samples,4)
    TXYZ_sim_surf_6[:,0] = TXYZ_sim_surf_6[:,0]*durT + startT
    TXYZ_sim_surf_6[:,1] *= Xsim 
    TXYZ_sim_surf_6[:,2] = Ysim
    TXYZ_sim_surf_6[:,3] *= Zsim

    # Right surface # Fix concentration 
    TXYZ_sim_surf_7 = np.random.rand(num_train_samples,4)
    TXYZ_sim_surf_7[:,0] = TXYZ_sim_surf_7[:,0]*durT + startT
    TXYZ_sim_surf_7[:,1] = Xsim 
    TXYZ_sim_surf_7[:,2] *= Ysim 
    TXYZ_sim_surf_7[:,3] *= Zsim

    # Top surface  # Fix concentration 
    TXYZ_sim_surf_8 = np.random.rand(num_train_samples,4)
    TXYZ_sim_surf_8[:,0] = TXYZ_sim_surf_8[:,0]*durT + startT
    TXYZ_sim_surf_8[:,1] *= Xsim 
    TXYZ_sim_surf_8[:,2] *= Ysim
    TXYZ_sim_surf_8[:,3] = Zsim

    # Initial condition  # Fixed concentration
    TXYZ_ini = np.random.rand(num_train_samples,4)
    TXYZ_ini[:,0] = startT
    TXYZ_ini[:,1] *= Xsim
    TXYZ_ini[:,2] *= Ysim
    TXYZ_ini[:,3] *= Zsim



    print('Generating target')

    C_e_surf_0 = np.ones((num_train_samples,1))
    C_e_surf_1 = np.ones((num_train_samples,1))
    C_e_surf_2 = np.ones((num_train_samples,1))
    C_e = np.ones((num_train_samples,1))

    for i in range(num_train_samples):
        if TXYZ_e_surf_0[i,0]<maxT/2.0:
            C_e_surf_0[i] = 1.0/(1.0+np.exp(-(theta_i-sigma*TXYZ_e_surf_0[i,0])))
        elif TXYZ_e_surf_0[i,0]>maxT/2.0:
            C_e_surf_0[i] = 1.0/(1.0+np.exp(-(theta_v+sigma*(TXYZ_e_surf_0[i,0]-maxT/2.0))))
        else: 
            C_e_surf_0[i] = 0.0

        if TXYZ_e_surf_1[i,0]<maxT/2.0:
            C_e_surf_1[i] = 1.0/(1.0+np.exp(-(theta_i-sigma*TXYZ_e_surf_1[i,0])))
        elif TXYZ_e_surf_1[i,0]>maxT/2.0:
            C_e_surf_1[i] = 1.0/(1.0+np.exp(-(theta_v+sigma*(TXYZ_e_surf_1[i,0]-maxT/2.0))))
        else: 
            C_e_surf_1[i] = 0.0

        if TXYZ_e_surf_2[i,0]<maxT/2.0:
            C_e_surf_2[i] = 1.0/(1.0+np.exp(-(theta_i-sigma*TXYZ_e_surf_2[i,0])))
        elif TXYZ_e_surf_2[i,0]>maxT/2.0:
            C_e_surf_2[i] = 1.0/(1.0+np.exp(-(theta_v+sigma*(TXYZ_e_surf_2[i,0]-maxT/2.0))))
        else: 
            C_e_surf_2[i] = 0.0


        if TXYZ_e[i,0] < maxT/2.0:
            C_e[i] = 1.0/(1.0+np.exp(-(theta_i-sigma*TXYZ_e[i,0])))
        elif TXYZ_e[i,0] > maxT/2.0:
            C_e[i] = 1.0/(1.0+np.exp(-(theta_v+sigma*(TXYZ_e[i,0]-maxT/2.0))))
        else:
            C_e[i] = 0.0



    C_eqn0 = np.zeros((num_train_samples,1))
    C_eqn1 = np.zeros((num_train_samples,1))
    C_eqn2 = np.zeros((num_train_samples,1))
    C_eqn3 = np.zeros((num_train_samples,1))
    C_eqn4 = np.zeros((num_train_samples,1))
    C_eqn5 = np.zeros((num_train_samples,1))

    C_sim_surf_0 = np.zeros((num_train_samples,1))
    C_sim_surf_1 = np.zeros((num_train_samples,1))
    C_sim_surf_2 = np.zeros((num_train_samples,1))
    C_sim_surf_3 = np.zeros((num_train_samples,1))
    C_sim_surf_4 = np.zeros((num_train_samples,1))
    C_sim_surf_5 = np.zeros((num_train_samples,1))

    C_sim_surf_6 = np.ones((num_train_samples,1))
    C_sim_surf_7 = np.ones((num_train_samples,1))
    C_sim_surf_8 = np.ones((num_train_samples,1))

    if startT==0.0:
        C_ini = np.ones((num_train_samples,1))
    else: 
        pinn.load_weights(initial_weights)
        C_ini = network.predict(TXYZ_ini)

    x_train = [TXYZ_ini,TXYZ_e,TXYZ_e_surf_0,TXYZ_e_surf_1,TXYZ_e_surf_2,TXYZ_eqn0,TXYZ_eqn1,TXYZ_eqn2,TXYZ_eqn3,TXYZ_eqn4,TXYZ_eqn5,TXYZ_sim_surf_0,TXYZ_sim_surf_1,TXYZ_sim_surf_2,TXYZ_sim_surf_3,TXYZ_sim_surf_4,TXYZ_sim_surf_5,TXYZ_sim_surf_6,TXYZ_sim_surf_7,TXYZ_sim_surf_8,]
    y_train = [C_ini,C_e,C_e_surf_0,C_e_surf_1,C_e_surf_2,C_eqn0,C_eqn1,C_eqn2,C_eqn3,C_eqn4,C_eqn5,C_sim_surf_0,C_sim_surf_1,C_sim_surf_2,C_sim_surf_3,C_sim_surf_4,C_sim_surf_5,C_sim_surf_6,C_sim_surf_7,C_sim_surf_8]



    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
    pinn.load_weights(default_weight_name)
    # If train = True, fit the NN, else, use saved weights 
    if train:
        if initial_weights:
            pinn.load_weights(initial_weights)
        history = pinn.fit(x=x_train,y=y_train,epochs=epochs,batch_size =batch_size,verbose=2,callbacks=[lr_scheduler_callback])
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(f'History {file_name}.csv')
        pinn.save_weights(f'./weights/weights {file_name}.h5')
    else:
        try:
            pinn.load_weights(f'./weights/weights {file_name}.h5')
        except:
            print('Weights does not exist\nStart training')
            if initial_weights:
                pinn.load_weights(initial_weights)
            pinn.fit(x=x_train,y=y_train,epochs=epochs,batch_size =batch_size,verbose=2,callbacks=[lr_scheduler_callback])
            pinn.save_weights(f'./weights/weights {file_name}.h5')


    """

    pinn.load_weights(f'./weights/weights {file_name}.h5')

    time_sects = [maxT*0.0,maxT*0.2,maxT*0.3,maxT*0.5]
    Z_sects = [0.1,0.5,1.0,1.2,2.0]

    for index_time_sect,time_sect in enumerate(time_sects):
        for index_Z_sect, Z_sect in enumerate(Z_sects):
            TXYZ_test = np.zeros((int(num_test_samples**2),4))
            TXYZ_test[:,0] = time_sect
            X_flat = np.linspace(0,Xsim,num_test_samples)
            Y_flat = np.linspace(0,Ysim,num_test_samples)
            X,Y = np.meshgrid(X_flat,Y_flat)
            TXYZ_test[:,1] = X.flatten()
            TXYZ_test[:,2] = Y.flatten()
            TXYZ_test[:,3] = Z_sect

            C = network.predict(TXYZ_test)
            C = C.reshape(X.shape)

            fig,ax = plt.subplots(figsize=(16,9))
            mesh = ax.pcolormesh(X,Y,C,shading='auto')
            cbar = plt.colorbar(mesh,pad=0.05, aspect=10,ax=ax)
            cbar.set_label('$C_R(X,Y)$')
            cbar.mappable.set_clim(0, 1.0)

            # add a red rectangle to represent the surface of electrode
            if Z_sect<=1.0:
                ax.add_patch(Rectangle((0.0,0.0),Xe,Ye,facecolor='white',edgecolor='k'))

            ax.set_title(f'Z={Z_sect:.2f} T={time_sect}')
            ax.set_aspect('equal')
            fig.savefig(f'{saving_directory}/{file_name} t={time_sect:.2f} Z={Z_sect:.2f}.png')
    


    time_steps = np.linspace(startT,endT,num=int(num_test_samples*4),endpoint=False)
    cv_flat = np.where(time_steps<maxT/2.0,theta_i-sigma*time_steps,theta_v+sigma*(time_steps-maxT/2.0))
    delta = 1e-4 
    for time_sect_index,time_sect in enumerate(time_steps):
        TXYZ_test = np.zeros((int(num_test_samples**2),4))
        TXYZ_test[:,0] = time_sect
        X_flat = np.linspace(0,Xe,num_test_samples)
        Y_flat = np.linspace(0,Ye,num_test_samples)
        X,Y = np.meshgrid(X_flat,Y_flat)
        TXYZ_test[:,1] = X.flatten()
        TXYZ_test[:,2] = Y.flatten()
        TXYZ_test[:,3] = Ze


        C_surf_0 = network.predict(TXYZ_test)
        C_surf_0 = C_surf_0.reshape(X.shape)

        TXYZ_test[:,3] = Ze + delta

        C_surf_0_delta = network.predict(TXYZ_test)
        C_surf_0_delta = C_surf_0_delta.reshape(X.shape)

        X_i = X_flat[1] - X_flat[0]
        Y_i = Y_flat[1] - Y_flat[0]
        J_surf_0 = -sum(sum((C_surf_0_delta - C_surf_0)/delta * X_i) * Y_i)

        # surf 1 
        TXYZ_test = np.zeros((int(num_test_samples**2),4))
        TXYZ_test[:,0] = time_sect
        X_flat = np.linspace(0,Xe,num_test_samples)
        Z_flat = np.linspace(0,Ze,num_test_samples)
        X,Z = np.meshgrid(X_flat,Z_flat)
        TXYZ_test[:,1] = X.flatten()
        TXYZ_test[:,2] = Ye
        TXYZ_test[:,3] = Z.flatten()


        C_surf_1 = network.predict(TXYZ_test)
        C_surf_1 = C_surf_1.reshape(X.shape)

        TXYZ_test[:,2] = Ye + delta

        C_surf_1_delta = network.predict(TXYZ_test)
        C_surf_1_delta = C_surf_1_delta.reshape(X.shape)

        X_i = X_flat[1] - X_flat[0]
        Z_i = Z_flat[1] - Z_flat[0]
        J_surf_1 = -sum(sum((C_surf_1_delta - C_surf_1)/delta * X_i) * Z_i)

        # surf 2
        TXYZ_test = np.zeros((int(num_test_samples**2),4))
        TXYZ_test[:,0] = time_sect
        Y_flat = np.linspace(0,Ye,num_test_samples)
        Z_flat = np.linspace(0,Ze,num_test_samples)
        Y,Z = np.meshgrid(Y_flat,Z_flat)
        TXYZ_test[:,1] = Xe
        TXYZ_test[:,2] = Y.flatten()
        TXYZ_test[:,3] = Z.flatten()


        C_surf_2 = network.predict(TXYZ_test)
        C_surf_2 = C_surf_2.reshape(X.shape)

        TXYZ_test[:,1] = Xe + delta

        C_surf_2_delta = network.predict(TXYZ_test)
        C_surf_2_delta = C_surf_2_delta.reshape(X.shape)

        Y_i = Y_flat[1] - Y_flat[0]
        Z_i = Z_flat[1] - Z_flat[0]
        J_surf_2 = -sum(sum((C_surf_2_delta - C_surf_2)/delta * Y_i) * Z_i)


        J_0.append(J_surf_0)
        J_1.append(J_surf_1)
        J_2.append(J_surf_2)
        Time_list.append(time_sect)
        Potential_list.append(cv_flat[time_sect_index])

        if time_sect_index%10 ==0:
            print(f'Progress of prediction: {time_sect_index/len(time_steps):.2%}')

    




        
    
    


    tf.keras.backend.clear_session()
    plt.close('all')

    return f'./weights/weights {file_name}.h5'





if __name__ =='__main__':

    # A list of dimensionless scan rates simulated
    sigmas = [10,20,30,40,60,80]

    for sigma in sigmas:
        epochs = 350
        theta_i = 10.0 # start/end potential of scan 
        theta_v = -10.0 # reverse potential of scan
        maxT = 2.0*abs(theta_v-theta_i)/sigma  # total time of voltammetric scan 
        saving_directory=f'./sigma={sigma:.1f} epochs = {epochs:.1f}'
        num_train_samples =  4e7
        batch_size = 1e5
        
        weights_name = prediction(epochs = epochs,sigma=sigma,startT=0.0,endT=maxT,durT=maxT,maxT=maxT,num_train_samples=num_train_samples,batch_size=batch_size,initial_weights=None,train=True,saving_directory=saving_directory)
        df = pd.DataFrame({'Time':Time_list,'Potential':Potential_list,'J_0':J_0,'J_1':J_1,'J_2':J_2})
        df.to_csv(f'{saving_directory} epochs = {epochs}.csv',index=False)
