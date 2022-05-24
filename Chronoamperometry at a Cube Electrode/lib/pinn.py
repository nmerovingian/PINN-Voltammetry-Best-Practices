import tensorflow as tf

from .layer import GradientLayer,BoundaryGradientLayer

class PINN:


    def __init__(self, network):


        self.network = network


        self.grads = GradientLayer(self.network)
        self.boundaryGrad = BoundaryGradientLayer(self.network)






    def build(self):

        TXYZ_ini = tf.keras.layers.Input(shape=(4,))
        TXYZ_e_surf_0 = tf.keras.layers.Input(shape=(4,))
        TXYZ_e_surf_1 = tf.keras.layers.Input(shape=(4,))
        TXYZ_e_surf_2 = tf.keras.layers.Input(shape=(4,))
        TXYZ_e = tf.keras.layers.Input(shape=(4,))
        TXYZ_eqn0 = tf.keras.layers.Input(shape=(4,))
        TXYZ_eqn1 = tf.keras.layers.Input(shape=(4,))
        TXYZ_eqn2 = tf.keras.layers.Input(shape=(4,))
        TXYZ_eqn3 = tf.keras.layers.Input(shape=(4,))
        TXYZ_eqn4 = tf.keras.layers.Input(shape=(4,))
        TXYZ_eqn5 = tf.keras.layers.Input(shape=(4,))
        TXYZ_sim_surf_0 = tf.keras.layers.Input(shape=(4,))
        TXYZ_sim_surf_1 = tf.keras.layers.Input(shape=(4,))
        TXYZ_sim_surf_2 = tf.keras.layers.Input(shape=(4,))
        TXYZ_sim_surf_3 = tf.keras.layers.Input(shape=(4,))
        TXYZ_sim_surf_4 = tf.keras.layers.Input(shape=(4,))
        TXYZ_sim_surf_5 = tf.keras.layers.Input(shape=(4,))
        TXYZ_sim_surf_6 = tf.keras.layers.Input(shape=(4,))
        TXYZ_sim_surf_7 = tf.keras.layers.Input(shape=(4,))
        TXYZ_sim_surf_8 = tf.keras.layers.Input(shape=(4,))


        C_ini = self.network(TXYZ_ini)
        C_e = self.network(TXYZ_e)
        C_e_surf_0 = self.network(TXYZ_e_surf_0)
        C_e_surf_1 = self.network(TXYZ_e_surf_1)
        C_e_surf_2 = self.network(TXYZ_e_surf_2)
        Ceqn0,dC_dT_eqn0,dC_dX_eqn0,dC_dY_eqn0,dC_dZ_eqn0,d2C_dX2_eqn0,d2C_dY2_eqn0,d2C_dZ2_eqn0 = self.grads(TXYZ_eqn0)
        Ceqn1,dC_dT_eqn1,dC_dX_eqn1,dC_dY_eqn1,dC_dZ_eqn1,d2C_dX2_eqn1,d2C_dY2_eqn1,d2C_dZ2_eqn1 = self.grads(TXYZ_eqn1)
        Ceqn2,dC_dT_eqn2,dC_dX_eqn2,dC_dY_eqn2,dC_dZ_eqn2,d2C_dX2_eqn2,d2C_dY2_eqn2,d2C_dZ2_eqn2 = self.grads(TXYZ_eqn2)
        Ceqn3,dC_dT_eqn3,dC_dX_eqn3,dC_dY_eqn3,dC_dZ_eqn3,d2C_dX2_eqn3,d2C_dY2_eqn3,d2C_dZ2_eqn3 = self.grads(TXYZ_eqn3)
        Ceqn4,dC_dT_eqn4,dC_dX_eqn4,dC_dY_eqn4,dC_dZ_eqn4,d2C_dX2_eqn4,d2C_dY2_eqn4,d2C_dZ2_eqn4 = self.grads(TXYZ_eqn4)
        Ceqn5,dC_dT_eqn5,dC_dX_eqn5,dC_dY_eqn5,dC_dZ_eqn5,d2C_dX2_eqn5,d2C_dY2_eqn5,d2C_dZ2_eqn5 = self.grads(TXYZ_eqn5)

        C_eqn0 = dC_dT_eqn0 - d2C_dX2_eqn0 - d2C_dY2_eqn0 - d2C_dZ2_eqn0
        C_eqn1 = dC_dT_eqn1 - d2C_dX2_eqn1 - d2C_dY2_eqn1 - d2C_dZ2_eqn1
        C_eqn2 = dC_dT_eqn2 - d2C_dX2_eqn2 - d2C_dY2_eqn2 - d2C_dZ2_eqn2
        C_eqn3 = dC_dT_eqn3 - d2C_dX2_eqn3 - d2C_dY2_eqn3 - d2C_dZ2_eqn3
        C_eqn4 = dC_dT_eqn4 - d2C_dX2_eqn4 - d2C_dY2_eqn4 - d2C_dZ2_eqn4
        C_eqn5 = dC_dT_eqn5 - d2C_dX2_eqn5 - d2C_dY2_eqn5 - d2C_dZ2_eqn5

        Csim_surf_0, dC_dT_sim_surf_0, dC_dX_sim_surf_0,dC_dY_sim_surf_0,dC_dZ_sim_surf_0 = self.boundaryGrad(TXYZ_sim_surf_0)
        Csim_surf_1, dC_dT_sim_surf_1, dC_dX_sim_surf_1,dC_dY_sim_surf_1,dC_dZ_sim_surf_1 = self.boundaryGrad(TXYZ_sim_surf_1)
        Csim_surf_2, dC_dT_sim_surf_2, dC_dX_sim_surf_2,dC_dY_sim_surf_2,dC_dZ_sim_surf_2 = self.boundaryGrad(TXYZ_sim_surf_2)
        Csim_surf_3, dC_dT_sim_surf_3, dC_dX_sim_surf_3,dC_dY_sim_surf_3,dC_dZ_sim_surf_3 = self.boundaryGrad(TXYZ_sim_surf_3)
        Csim_surf_4, dC_dT_sim_surf_4, dC_dX_sim_surf_4,dC_dY_sim_surf_4,dC_dZ_sim_surf_4 = self.boundaryGrad(TXYZ_sim_surf_4)
        Csim_surf_5, dC_dT_sim_surf_5, dC_dX_sim_surf_5,dC_dY_sim_surf_5,dC_dZ_sim_surf_5 = self.boundaryGrad(TXYZ_sim_surf_5)

        C_sim_surf_0 = dC_dY_sim_surf_0
        C_sim_surf_1 = dC_dY_sim_surf_1
        C_sim_surf_2 = dC_dX_sim_surf_2
        C_sim_surf_3 = dC_dX_sim_surf_3
        C_sim_surf_4 = dC_dZ_sim_surf_4
        C_sim_surf_5 = dC_dZ_sim_surf_5

        C_sim_surf_6 = self.network(TXYZ_sim_surf_6)
        C_sim_surf_7 = self.network(TXYZ_sim_surf_7)
        C_sim_surf_8 = self.network(TXYZ_sim_surf_8)





        return tf.keras.models.Model(
            inputs=[TXYZ_ini,TXYZ_e,TXYZ_e_surf_0,TXYZ_e_surf_1,TXYZ_e_surf_2,TXYZ_eqn0,TXYZ_eqn1,TXYZ_eqn2,TXYZ_eqn3,TXYZ_eqn4,TXYZ_eqn5,TXYZ_sim_surf_0,TXYZ_sim_surf_1,TXYZ_sim_surf_2,TXYZ_sim_surf_3,TXYZ_sim_surf_4,TXYZ_sim_surf_5,TXYZ_sim_surf_6,TXYZ_sim_surf_7,TXYZ_sim_surf_8,], outputs=[C_ini,C_e,C_e_surf_0,C_e_surf_1,C_e_surf_2,C_eqn0,C_eqn1,C_eqn2,C_eqn3,C_eqn4,C_eqn5,C_sim_surf_0,C_sim_surf_1,C_sim_surf_2,C_sim_surf_3,C_sim_surf_4,C_sim_surf_5,C_sim_surf_6,C_sim_surf_7,C_sim_surf_8])


            