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

default_weight_name = "./weights/default.h5"
pinn.save_weights(default_weight_name)






def prediction(epochs=10,maxT=1.0,Xe=1.0,Ye=1.0,Ze=1.0,num_train_samples=1e6,batch_size = 1e4, alpha=0.99,initial_weights = None,train=True,saving_directory="./Data",):
    # prefix or suffix of files 
    file_name = f'maxT={maxT:.2E} epochs={epochs:.2E} n_train={num_train_samples:.2E}'

    def schedule(epoch,lr):
        # a learning rate scheduler 
        if epoch<=400:
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

    # number of training samples
    num_train_samples = int(num_train_samples)
    # number of test samples
    num_test_samples = 100
    # batch size
    batch_size = int(batch_size)

    Xsim = 4.0*np.sqrt(maxT) + Xe
    Ysim = 4.0*np.sqrt(maxT) + Ye
    Zsim = 4.0*np.sqrt(maxT) + Ze
    stall_T = 0.1

    # electrode surface
    TXYZ_e_surf_0 = np.random.rand(num_train_samples,4)
    TXYZ_e_surf_0[:,0] *= maxT
    TXYZ_e_surf_0[:,1] *= Xe
    TXYZ_e_surf_0[:,2] *= Ye
    TXYZ_e_surf_0[:,3] = Ze

    TXYZ_e_surf_1 = np.random.rand(num_train_samples,4)
    TXYZ_e_surf_1[:,0] *= maxT
    TXYZ_e_surf_1[:,1] *= Xe
    TXYZ_e_surf_1[:,2] = Ye
    TXYZ_e_surf_1[:,3] *= Ze

    TXYZ_e_surf_2 = np.random.rand(num_train_samples,4)
    TXYZ_e_surf_2[:,0] *= maxT
    TXYZ_e_surf_2[:,1] = Xe
    TXYZ_e_surf_2[:,2] *= Ye
    TXYZ_e_surf_2[:,3] *= Ze

    TXYZ_e = np.random.rand(num_train_samples,4)
    TXYZ_e[:,0] *= maxT
    TXYZ_e[:,1] *= Xe
    TXYZ_e[:,2] *= Ye
    TXYZ_e[:,3] *= Ze

    TXYZ_eqn0 = np.random.rand(num_train_samples,4)
    TXYZ_eqn0[:,0] *= maxT
    TXYZ_eqn0[:,1] *= Xsim 
    TXYZ_eqn0[:,2] =  TXYZ_eqn0[:,2] * (Ysim-Ye) + Ye 
    TXYZ_eqn0[:,3] *= Zsim

    TXYZ_eqn1 = np.random.rand(num_train_samples,4)
    TXYZ_eqn1[:,0] *= maxT
    TXYZ_eqn1[:,1] = TXYZ_eqn1[:,1] * (Xsim-Xe) + Xe 
    TXYZ_eqn1[:,2] *= Ysim  
    TXYZ_eqn1[:,3] *= Zsim

    TXYZ_eqn2 = np.random.rand(num_train_samples,4)
    TXYZ_eqn2[:,0] *= maxT
    TXYZ_eqn2[:,1] = TXYZ_eqn2[:,1]*(Xsim-Xe) + Xe 
    TXYZ_eqn2[:,2] *= Ysim 
    TXYZ_eqn2[:,3] *= Zsim

    TXYZ_eqn3 = np.random.rand(num_train_samples,4)
    TXYZ_eqn3[:,0] *= maxT
    TXYZ_eqn3[:,1] *= Xsim 
    TXYZ_eqn3[:,2] *= Ysim  
    TXYZ_eqn3[:,3] = TXYZ_eqn3[:,3]*(Zsim-Ze) + Ze

    TXYZ_eqn4 = np.random.rand(num_train_samples,4)
    TXYZ_eqn4[:,0] *= maxT
    TXYZ_eqn4[:,1] *= Xsim 
    TXYZ_eqn4[:,2] = TXYZ_eqn4[:,2] * (Ysim -Ye) + Ye
    TXYZ_eqn4[:,3] *= Zsim

    TXYZ_eqn5 = np.random.rand(num_train_samples,4)
    TXYZ_eqn5[:,0] *= maxT
    TXYZ_eqn5[:,1] *= Xsim 
    TXYZ_eqn5[:,2] *= Ysim  
    TXYZ_eqn5[:,3] = TXYZ_eqn5[:,3] * (Zsim-Ze) + Ze


    # Simulation Surface
    # Front surface
    TXYZ_sim_surf_0 = np.random.rand(num_train_samples,4)
    TXYZ_sim_surf_0[:,0] *= maxT
    TXYZ_sim_surf_0[:,1] = TXYZ_sim_surf_0[:,1] * (Xsim-Xe) + Xe
    TXYZ_sim_surf_0[:,2] = 0.0
    TXYZ_sim_surf_0[:,3] *= Zsim

    TXYZ_sim_surf_1 = np.random.rand(num_train_samples,4)
    TXYZ_sim_surf_1[:,0] *= maxT
    TXYZ_sim_surf_1[:,1] *= Xsim 
    TXYZ_sim_surf_1[:,2] = 0.0 
    TXYZ_sim_surf_1[:,3] = TXYZ_sim_surf_1[:,3] * (Zsim-Ze) + Ze

    # Left surface
    TXYZ_sim_surf_2 = np.random.rand(num_train_samples,4)
    TXYZ_sim_surf_2[:,0] *= maxT 
    TXYZ_sim_surf_2[:,1] = 0.0
    TXYZ_sim_surf_2[:,2] = TXYZ_sim_surf_2[:,2] * (Ysim-Ye) + Ye
    TXYZ_sim_surf_2[:,3] *= Zsim 

    TXYZ_sim_surf_3 = np.random.rand(num_train_samples,4)
    TXYZ_sim_surf_3[:,0] *= maxT 
    TXYZ_sim_surf_3[:,1] = 0.0
    TXYZ_sim_surf_3[:,2] *= Ysim 
    TXYZ_sim_surf_3[:,3] = TXYZ_sim_surf_3[:,3] * (Zsim-Ze) + Ze

    # Bottom surface
    TXYZ_sim_surf_4 = np.random.rand(num_train_samples,4)
    TXYZ_sim_surf_4[:,0] *= maxT
    TXYZ_sim_surf_4[:,1] = TXYZ_sim_surf_4[:,1] * (Xsim-Xe) + Xe
    TXYZ_sim_surf_4[:,2] *= Ysim 
    TXYZ_sim_surf_4[:,3] = 0.0

    TXYZ_sim_surf_5 = np.random.rand(num_train_samples,4)
    TXYZ_sim_surf_5[:,0] *= maxT
    TXYZ_sim_surf_5[:,1] *= Xsim 
    TXYZ_sim_surf_5[:,2] = TXYZ_sim_surf_5[:,2] * (Ysim-Ye) + Ye
    TXYZ_sim_surf_5[:,3] = 0.0


    # Back surface
    TXYZ_sim_surf_6 = np.random.rand(num_train_samples,4)
    TXYZ_sim_surf_6[:,0] *= maxT
    TXYZ_sim_surf_6[:,1] *= Xsim 
    TXYZ_sim_surf_6[:,2] = Ysim
    TXYZ_sim_surf_6[:,3] *= Zsim

    # Right surface # Fix concentration 
    TXYZ_sim_surf_7 = np.random.rand(num_train_samples,4)
    TXYZ_sim_surf_7[:,0] *= maxT
    TXYZ_sim_surf_7[:,1] = Xsim 
    TXYZ_sim_surf_7[:,2] *= Ysim 
    TXYZ_sim_surf_7[:,3] *= Zsim

    # Top surface  # Fix concentration 
    TXYZ_sim_surf_8 = np.random.rand(num_train_samples,4)
    TXYZ_sim_surf_8[:,0] *= maxT
    TXYZ_sim_surf_8[:,1] *= Xsim 
    TXYZ_sim_surf_8[:,2] *= Ysim
    TXYZ_sim_surf_8[:,3] = Zsim

    # Initial condition  # Fixed concentration
    TXYZ_ini = np.random.rand(num_train_samples,4)
    TXYZ_ini[:,0] = 0.0
    TXYZ_ini[:,1] *= Xsim
    TXYZ_ini[:,2] *= Ysim
    TXYZ_ini[:,3] *= Zsim



    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x, y, z = np.indices((1, 1, 1))
    cube1 = (x < 1) & (y < 1) & (z < 1)
    ax.voxels(cube1, edgecolor='k')
    ax.scatter(TXYZ_eqn0[:,1],TXYZ_eqn0[:,2],TXYZ_eqn0[:,3])
    ax.scatter(TXYZ_eqn1[:,1],TXYZ_eqn1[:,2],TXYZ_eqn1[:,3])
    ax.scatter(TXYZ_eqn2[:,1],TXYZ_eqn2[:,2],TXYZ_eqn2[:,3])
    ax.scatter(TXYZ_eqn3[:,1],TXYZ_eqn3[:,2],TXYZ_eqn3[:,3])
    ax.scatter(TXYZ_eqn4[:,1],TXYZ_eqn4[:,2],TXYZ_eqn4[:,3])
    ax.scatter(TXYZ_eqn5[:,1],TXYZ_eqn5[:,2],TXYZ_eqn5[:,3])
    """
    """
    ax.scatter(TXYZ_sim_surf_0[:,1],TXYZ_sim_surf_0[:,2],TXYZ_sim_surf_0[:,3],label='Front Surface 1')
    ax.scatter(TXYZ_sim_surf_1[:,1],TXYZ_sim_surf_1[:,2],TXYZ_sim_surf_1[:,3],label='Front Surface 2')
    ax.scatter(TXYZ_sim_surf_2[:,1],TXYZ_sim_surf_2[:,2],TXYZ_sim_surf_2[:,3],label='Left Surface 1')
    ax.scatter(TXYZ_sim_surf_3[:,1],TXYZ_sim_surf_3[:,2],TXYZ_sim_surf_3[:,3],label='Left Surface 2')
    ax.scatter(TXYZ_sim_surf_4[:,1],TXYZ_sim_surf_4[:,2],TXYZ_sim_surf_4[:,3],label='Bottom Surface 1')
    ax.scatter(TXYZ_sim_surf_5[:,1],TXYZ_sim_surf_5[:,2],TXYZ_sim_surf_5[:,3],label='Bottom Surface 2')
    ax.scatter(TXYZ_sim_surf_6[:,1],TXYZ_sim_surf_6[:,2],TXYZ_sim_surf_6[:,3],label='Back Surface')
    ax.scatter(TXYZ_sim_surf_7[:,1],TXYZ_sim_surf_7[:,2],TXYZ_sim_surf_7[:,3],label='Right Surface')
    ax.scatter(TXYZ_sim_surf_8[:,1],TXYZ_sim_surf_8[:,2],TXYZ_sim_surf_8[:,3],label='Top Surface')
    
    ax.scatter(TXYZ_ini[:,1],TXYZ_ini[:,2],TXYZ_ini[:,3])

    ax.legend()


    plt.show()
    """
    C_e_surf_0 = np.ones((num_train_samples,1))
    C_e_surf_1 = np.ones((num_train_samples,1))
    C_e_surf_2 = np.ones((num_train_samples,1))
    C_e = np.ones((num_train_samples,1))

    for i in range(num_train_samples):
        if TXYZ_e_surf_0[i,0]>stall_T:
            C_e_surf_0[i] = 0.0
        if TXYZ_e_surf_1[i,0]>stall_T:
            C_e_surf_1[i] = 0.0
        if TXYZ_e_surf_2[i,0]>stall_T:
            C_e_surf_2[i] = 0.0
        if TXYZ_e[i,0]>stall_T:
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

    C_ini = np.ones((num_train_samples,1))

    x_train = [TXYZ_ini,TXYZ_e,TXYZ_e_surf_0,TXYZ_e_surf_1,TXYZ_e_surf_2,TXYZ_eqn0,TXYZ_eqn1,TXYZ_eqn2,TXYZ_eqn3,TXYZ_eqn4,TXYZ_eqn5,TXYZ_sim_surf_0,TXYZ_sim_surf_1,TXYZ_sim_surf_2,TXYZ_sim_surf_3,TXYZ_sim_surf_4,TXYZ_sim_surf_5,TXYZ_sim_surf_6,TXYZ_sim_surf_7,TXYZ_sim_surf_8,]
    y_train = [C_ini,C_e,C_e_surf_0,C_e_surf_1,C_e_surf_2,C_eqn0,C_eqn1,C_eqn2,C_eqn3,C_eqn4,C_eqn5,C_sim_surf_0,C_sim_surf_1,C_sim_surf_2,C_sim_surf_3,C_sim_surf_4,C_sim_surf_5,C_sim_surf_6,C_sim_surf_7,C_sim_surf_8]



    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
    pinn.load_weights(default_weight_name)
    # If train = True, fit the NN, else, use saved weights 
    if train:
        if initial_weights:
            pinn.load_weights(initial_weights)
        pinn.fit(x=x_train,y=y_train,epochs=epochs,batch_size =batch_size,verbose=2,callbacks=[lr_scheduler_callback])
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

    
    time_sects = [0.0,0.2,0.3,0.5]
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
    

    # Estimate the flux
    J_0 = list()
    J_1 = list()
    J_2 = list()
    Time = list()

    delta = 1e-4 
    for time_sect in np.linspace(0,maxT,num=int(num_test_samples*10)):

        # surf 1 Z = 1, XY
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
        J_surf_0 = sum(sum((C_surf_0_delta - C_surf_0)/delta * X_i) * Y_i)

        # surf 1 Y=1
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
        J_surf_1 = sum(sum((C_surf_1_delta - C_surf_1)/delta * X_i) * Z_i)

        # surf 2 X=1
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
        J_surf_2 = sum(sum((C_surf_2_delta - C_surf_2)/delta * Y_i) * Z_i)


        Time.append(time_sect)
        J_0.append(J_surf_0)
        J_1.append(J_surf_1)
        J_2.append(J_surf_2)

        

    df = pd.DataFrame({'Time':Time,'J_0':J_0,'J_1':J_1,'J_2':J_2})

    df.to_csv(f'{file_name}.csv')



    tf.keras.backend.clear_session()
    plt.close('all')

    return f'./weights/weights {file_name}.h5'





if __name__ =='__main__':
    
    prediction(epochs = 300,num_train_samples=1e7,train=False)