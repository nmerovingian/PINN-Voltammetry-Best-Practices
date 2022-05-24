import numpy as np 
import os
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle,Wedge
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
fontsize = 20

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : fontsize }
plt.rc('font', **font)  # pass in the font dict as kwargs



"""
# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    network = Network.build()
    network.summary()
    pinn = PINN(network).build()
    pinn.compile(optimizer='Adam',loss='mse')
"""
network = Network.build()
network.summary()
pinn = PINN(network).build()
pinn.compile(optimizer='Adam',loss='mse')
default_weight_name = "./weights/default.h5"
pinn.save_weights(default_weight_name)


"""
Training with a data generator by generating training data on the fly can be a major computer memory saving practice. 

"""



class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,num_train_samples,batch_size,maxT,stall_T,Re,Xsim,Ysim,Zsim):
        self.num_train_samples = num_train_samples
        self.batch_size = batch_size
        self.maxT = maxT
        self.stall_T = stall_T
        self.Re = Re
        self.Xsim = Xsim
        self.Ysim = Ysim 
        self.Zsim = Zsim


    
    def __len__(self):
        return int(np.floor(self.num_train_samples / self.batch_size))

    def __getitem__(self, index):



        TXYZ_ini = np.random.rand(self.batch_size,4)
        TXYZ_ini[:,0] = 0.0
        TXYZ_ini[:,1] *= self.Xsim
        TXYZ_ini[:,2] *= self.Ysim
        TXYZ_ini[:,3] *= self.Zsim


        TXYZ_dmn0 = np.random.rand(self.batch_size,4)
        TXYZ_dmn0[:,0] *= self.maxT
        TXYZ_dmn0[:,1] *= self.Xsim
        TXYZ_dmn0[:,2] *= self.Ysim
        TXYZ_dmn0[:,3] *= self.Zsim

        TXYZ_dmn1 = np.random.rand(self.batch_size,4)
        TXYZ_dmn1[:,0] *= self.maxT
        TXYZ_dmn1[:,1] *= self.Xsim
        TXYZ_dmn1[:,2] *= self.Ysim
        TXYZ_dmn1[:,3] *= 1.0


        R_e = np.random.rand(self.batch_size)*self.Re
        theta_e = np.random.rand(self.batch_size)*np.pi/2
        TXYZ_e = np.random.rand(self.batch_size,4)
        TXYZ_e[:,0] *= self.maxT
        TXYZ_e[:,2] = np.sqrt(R_e)*np.cos(theta_e) 
        TXYZ_e[:,1] = np.sqrt(R_e)*np.sin(theta_e) 
        TXYZ_e[:,3] = 0.0

        R_e_1 = np.random.rand(self.batch_size)*0.4 + 0.6
        theta_e_1 = np.random.rand(self.batch_size)*np.pi/2
        TXYZ_e_1 = np.random.rand(self.batch_size,4)
        TXYZ_e_1[:,0] *= self.maxT
        TXYZ_e_1[:,2] = R_e_1*np.cos(theta_e_1) 
        TXYZ_e_1[:,1] = R_e_1*np.sin(theta_e_1) 
        TXYZ_e_1[:,3] = 0.0


        TXYZ_bnd_0 = np.random.rand(self.batch_size,4)
        TXYZ_bnd_0[:,0] *= self.maxT
        TXYZ_bnd_0[:,1] *= self.Xsim
        TXYZ_bnd_0[:,2] *= self.Ysim
        TXYZ_bnd_0[:,3] = 0.0
        TXYZ_bnd_0 = TXYZ_bnd_0[TXYZ_bnd_0[:,1]**2+TXYZ_bnd_0[:,2]**2>1.0]
        while len(TXYZ_bnd_0) < self.batch_size:
            TXYZ_bnd_temp = np.random.rand(int(self.batch_size/5),4)
            TXYZ_bnd_temp[:,0] *= self.maxT
            TXYZ_bnd_temp[:,1] *= self.Xsim
            TXYZ_bnd_temp[:,2] *= self.Ysim
            TXYZ_bnd_temp[:,3] = 0.0
            TXYZ_bnd_temp = TXYZ_bnd_temp[TXYZ_bnd_temp[:,1]**2+TXYZ_bnd_temp[:,2]**2>1.0]
            TXYZ_bnd_0  = np.concatenate((TXYZ_bnd_0,TXYZ_bnd_temp),axis=0)

        TXYZ_bnd_0 = TXYZ_bnd_0[:self.batch_size]

        # left surface
        TXYZ_bnd_1 = np.random.rand(self.batch_size,4)
        TXYZ_bnd_1[:,0] *= self.maxT
        TXYZ_bnd_1[:,1]  = 0.0
        TXYZ_bnd_1[:,2] *= self.Ysim
        TXYZ_bnd_1[:,3] *= self.Zsim

        # right 
        TXYZ_bnd_2 = np.random.rand(self.batch_size,4)
        TXYZ_bnd_2[:,0] *= self.maxT
        TXYZ_bnd_2[:,1] = self.Xsim
        TXYZ_bnd_2[:,2] *= self.Ysim
        TXYZ_bnd_2[:,3] *= self.Zsim

        # top 
        TXYZ_bnd_3 = np.random.rand(self.batch_size,4)
        TXYZ_bnd_3[:,0] *= self.maxT
        TXYZ_bnd_3[:,1] *= self.Xsim
        TXYZ_bnd_3[:,2] *= self.Ysim
        TXYZ_bnd_3[:,3] = self.Zsim

        # front 
        TXYZ_bnd_4 = np.random.rand(self.batch_size,4)
        TXYZ_bnd_4[:,0] *= self.maxT
        TXYZ_bnd_4[:,1] *= self.Xsim
        TXYZ_bnd_4[:,2] = 0.0
        TXYZ_bnd_4[:,3] *= self.Zsim

        # back
        TXYZ_bnd_5 = np.random.rand(self.batch_size,4)
        TXYZ_bnd_5[:,0] *= self.maxT
        TXYZ_bnd_5[:,1] *= self.Xsim
        TXYZ_bnd_5[:,2] = self.Ysim
        TXYZ_bnd_5[:,3] *= self.Zsim


        C_ini = np.ones((self.batch_size,1))
        C_dmn0 = np.zeros((self.batch_size,1))
        C_dmn1 = np.zeros((self.batch_size,1))

        C_e = np.zeros((self.batch_size,1))
        for i in range(self.batch_size):
            if TXYZ_e[i,0]<self.stall_T:
                C_e[i] = 1.0

        C_e_1 = np.zeros((self.batch_size,1))
        for i in range(self.batch_size):
            if TXYZ_e_1[i,0]<self.stall_T:
                C_e_1[i] = 1.0

        C_bnd0 = np.zeros((self.batch_size,1))
        C_bnd1 = np.zeros((self.batch_size,1))
        C_bnd2 = np.ones((self.batch_size,1))
        C_bnd3 = np.ones((self.batch_size,1))
        C_bnd4 = np.zeros((self.batch_size,1))
        C_bnd5 = np.ones((self.batch_size,1))

        x_train = [TXYZ_ini,TXYZ_dmn0,TXYZ_dmn1,TXYZ_e,TXYZ_e_1,TXYZ_bnd_0,TXYZ_bnd_1,TXYZ_bnd_2,TXYZ_bnd_3,TXYZ_bnd_4,TXYZ_bnd_5]
        y_train = [C_ini,C_dmn0,C_dmn1,C_e,C_e_1,C_bnd0,C_bnd1,C_bnd2,C_bnd3,C_bnd4,C_bnd5]


        return x_train,y_train


def prediction(epochs=10,maxT=1.0,Re=1.0,num_train_samples=1e6,batch_size = 1e4, alpha=0.99,initial_weights = None,train=True,saving_directory="./Data",):
    # prefix or suffix of files 
    file_name = f'maxT={maxT:.2E} epochs={epochs:.2E} n_train={num_train_samples:.2E} batch_size = {batch_size:.2E}'

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

    """
    theta_i = 10.0 # start/end potential of scan 
    theta_v = -10.0 # reverse potential of scan
    maxT = 2.0*abs(theta_v-theta_i)/sigma  # total time of voltammetric scan 
    """
    maxT = maxT 
    stall_T = 0.1
    # number of training samples
    num_train_samples =  int(num_train_samples)
    # number of test samples
    num_test_samples = 100

    # number of degrees
    num_theta_test_samples = 31 
    # batch size
    batch_size = int(batch_size)

    Xsim = 3.0*np.sqrt(maxT) + Re
    Ysim = 3.0*np.sqrt(maxT) + Re
    Zsim = 3.0*np.sqrt(maxT) + Re


    training_generator = DataGenerator(num_train_samples=num_train_samples,batch_size=batch_size,maxT=maxT,stall_T=stall_T,Re=Re,Xsim=Xsim,Ysim=Ysim,Zsim=Zsim)




    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
    pinn.load_weights(default_weight_name)
    # If train = True, fit the NN, else, use saved weights 
    if train:
        if initial_weights:
            pinn.load_weights(initial_weights)
        pinn.fit(training_generator,epochs=epochs,verbose=2,callbacks=[lr_scheduler_callback])
        pinn.save_weights(f'./weights/weights {file_name}.h5')
    else:
        try:
            pinn.load_weights(f'./weights/weights {file_name}.h5')
        except:
            print('Weights does not exist\nStart training')
            if initial_weights:
                pinn.load_weights(initial_weights)
            pinn.fit(training_generator,epochs=epochs,verbose = 2,callbacks=[lr_scheduler_callback])
            pinn.save_weights(f'./weights/weights {file_name}.h5')

    


    
    time_steps = np.linspace(0.0,maxT,num=int(num_test_samples*10))
    theta_steps = np.linspace(0.0,np.pi/2,num=num_theta_test_samples)
    R_steps = np.linspace(0.0,Re,num=num_test_samples)

    R_i = R_steps[1] - R_steps[0]

    deltaZ = 1e-3
    J_disc = list()
    for time_sect in time_steps:
        
        theta,R= np.meshgrid(theta_steps,R_steps)
        X = R*np.cos(theta)
        Y = R*np.sin(theta)
        
        TXYZ_test = np.zeros((int(num_test_samples*num_theta_test_samples),4))
        TXYZ_test[:,0] = time_sect
        TXYZ_test[:,1] = X.flatten()
        TXYZ_test[:,2] = Y.flatten()
        TXYZ_test[:,3] = 0.0
        
        

        

        C_surf_0 = network.predict(TXYZ_test)
        C_surf_0 = C_surf_0.reshape(X.shape)

        TXYZ_test[:,3] = deltaZ

        C_surf_0_delta = network.predict(TXYZ_test)
        C_surf_0_delta = C_surf_0_delta.reshape(X.shape)

        flux_surf = -(C_surf_0_delta - C_surf_0)/deltaZ
        
        

        J = np.average(sum(flux_surf*R_i))
        J_disc.append(J)
        plt.close('all')
    df = pd.DataFrame({'Time':time_steps,'J':J_disc})


    df.to_csv(f'{file_name}.csv',index=False)
    

    # plot the chrono
    df = pd.read_csv(f'{file_name}.csv')

    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot(df['Time'],df['J'],label='PINN',color='k',lw=3,alpha=0.7)
    Time  = np.linspace(0.007,maxT-stall_T,num=200)
    Analytical_J = 0.7854+0.4431/np.sqrt(Time)+0.2146*np.exp(0.39115/np.sqrt(Time))
    Analytical_J = -Analytical_J
    ax.set_xlabel('Time, T')
    ax.set_ylabel('Flux, J')
    ax.plot(Time+stall_T,Analytical_J,label='Shoup-Szabo Equation',color='r',ls='--',lw=3,alpha=0.7)
    ax.legend()
    fig.savefig(f'{file_name} analytical.png')


    tf.keras.backend.clear_session()
    plt.close('all')


    time_sects = [0.5]
    Z_sects = [0.0]

    
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

            fig,axes = plt.subplots(figsize=(8,9),nrows=2)
            ax = axes[0]
            mesh = ax.pcolormesh(X,Y,C,shading='auto')
            cbar = plt.colorbar(mesh,pad=0.05, aspect=10,ax=ax)
            cbar.set_label('$C_R(X,Y)$')
            ax.contour(X,Y,C,10,colors='white',linewidths=2,alpha=0.5)
            cbar.mappable.set_clim(0, 1.0)
            if Z_sect == 0.0:
                ax.add_patch(Wedge(center=(0,0),r=1,theta1=0,theta2=90,facecolor='k',edgecolor='r',lw=2,zorder=2))

            #X_coordinates = np.linspace(0,Re)
            #Y_coordinates = np.sqrt(Re**2-X_coordinates**2)
            #ax.plot(X_coordinates,Y_coordinates,lw=3,alpha=0.7,color='r')


            ax.set_aspect('equal')

            ax.set_xlabel(r'X',fontsize='large',fontweight='bold')
            ax.set_ylabel(r'Y',fontsize='large',fontweight='bold')

            ax.set_xticks([0,1,2,3,4])


            ax = axes[1]

            df = pd.read_csv(f'{file_name}.csv')

            ax.plot(df['Time'],df['J'],label='PINN',color='b',lw=3,alpha=0.7)
            Time  = np.linspace(0.007,maxT-stall_T,num=200)
            Analytical_J = 0.7854+0.4431/np.sqrt(Time)+0.2146*np.exp(0.39115/np.sqrt(Time))
            Analytical_J = -Analytical_J
            ax.set_xlabel('Time, T',fontsize='large',fontweight='bold')
            ax.set_ylabel('Flux, J',fontsize='large',fontweight='bold')
            ax.plot(Time+stall_T,Analytical_J,label='Analytical Equation',color='r',ls='--',lw=3,alpha=0.7)

            ax.legend()

            fig.tight_layout()
            fig.text(0.05,0.92,'A)',fontsize=30)
            fig.text(0.05,0.45,'B)',fontsize=30)


            fig.savefig(f'{file_name}.png',dpi=250,bbox_inches='tight')

            plt.close('all')
    










    return f'./weights/weights {file_name}.h5'





if __name__ =='__main__':

    prediction(epochs = 800,num_train_samples=1e8,batch_size=1e5,train=False,saving_directory='./Data')

