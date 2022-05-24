import tensorflow as tf

from .layer import GradientLayer,BoundaryGradientLayer

class PINN:


    def __init__(self, network):


        self.network = network


        self.grads = GradientLayer(self.network)
        self.boundaryGrad = BoundaryGradientLayer(self.network)






    def build(self):


        TXYZ_ini = tf.keras.layers.Input(shape=(4,))
        TXYZ_dmn0 = tf.keras.layers.Input(shape=(4,))
        TXYZ_dmn1 = tf.keras.layers.Input(shape=(4,))
        TXYZ_e = tf.keras.layers.Input(shape=(4,))
        TXYZ_e_1 = tf.keras.layers.Input(shape=(4,))
        TXYZ_bnd_0 = tf.keras.layers.Input(shape=(4,))
        TXYZ_bnd_1 = tf.keras.layers.Input(shape=(4,))
        TXYZ_bnd_2 = tf.keras.layers.Input(shape=(4,))
        TXYZ_bnd_3 = tf.keras.layers.Input(shape=(4,))
        TXYZ_bnd_4 = tf.keras.layers.Input(shape=(4,))
        TXYZ_bnd_5 = tf.keras.layers.Input(shape=(4,))




        C_ini = self.network(TXYZ_ini)


        Cdm0,dC_dT_dmn0,dC_dX_dmn0,dC_dY_dmn0,dC_dZ_dmn0,d2C_dX2_dmn0,d2C_dY2_dmn0,d2C_dZ2_dmn0 = self.grads(TXYZ_dmn0)
        Cdmn1,dC_dT_dmn1,dC_dX_dmn1,dC_dY_dmn1,dC_dZ_dmn1,d2C_dX2_dmn1,d2C_dY2_dmn1,d2C_dZ2_dmn1 = self.grads(TXYZ_dmn1)


        C_dmn0 = dC_dT_dmn0 - d2C_dX2_dmn0 - d2C_dY2_dmn0 - d2C_dZ2_dmn0
        C_dmn1 = dC_dT_dmn1 - d2C_dX2_dmn1 - d2C_dY2_dmn1 - d2C_dZ2_dmn1


        C_e = self.network(TXYZ_e)
        C_e_1 = self.network(TXYZ_e_1)


        Cbnd0,dC_dT_bnd0,dC_dX_bnd0,dC_dY_bnd0,dC_dZ_bnd0 = self.boundaryGrad(TXYZ_bnd_0)
        C_bnd0 = dC_dZ_bnd0

        Cbnd1,dC_dT_bnd1,dC_dX_bnd1,dC_dY_bnd1,dC_dZ_bnd1 = self.boundaryGrad(TXYZ_bnd_1)
        C_bnd1 = dC_dX_bnd1


        C_bnd2 = self.network(TXYZ_bnd_2)
        C_bnd3 = self.network(TXYZ_bnd_3)


        Cbnd4,dC_dT_bnd4,dC_dX_bnd4,dC_dY_bnd4,dC_dZ_bnd4 = self.boundaryGrad(TXYZ_bnd_4)
        C_bnd4 = dC_dY_bnd4
        C_bnd5 = self.network(TXYZ_bnd_5)




        return tf.keras.models.Model(
            inputs=[TXYZ_ini,TXYZ_dmn0,TXYZ_dmn1,TXYZ_e,TXYZ_e_1,TXYZ_bnd_0,TXYZ_bnd_1,TXYZ_bnd_2,TXYZ_bnd_3,TXYZ_bnd_4,TXYZ_bnd_5], outputs=[C_ini,C_dmn0,C_dmn1,C_e,C_e_1,C_bnd0,C_bnd1,C_bnd2,C_bnd3,C_bnd4,C_bnd5])


            