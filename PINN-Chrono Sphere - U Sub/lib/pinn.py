import tensorflow as tf
from .layer import GradientLayer

class PINN:


    def __init__(self, network):

        self.network = network
        self.grads = GradientLayer(self.network)

    def build(self):


        # equation input: (t, x)
        tx_eqn = tf.keras.layers.Input(shape=(2,))



        tx_ini = tf.keras.layers.Input(shape=(2,))

        tx_bnd0 = tf.keras.layers.Input(shape=(2,))
        tx_bnd1 = tf.keras.layers.Input(shape=(2,))

        # compute gradients
        c, dc_dt, dc_dx, d2c_dx2 = self.grads(tx_eqn)

        # equation output being zero
        c_eqn = dc_dt - d2c_dx2 
        # initial condition output
        c_ini = self.network(tx_ini)
        # boundary condition output
        c_bnd0 = self.network(tx_bnd0)
        c_bnd1 = self.network(tx_bnd1)

        # build the PINN model for Burgers' equation
        return tf.keras.models.Model(
            inputs=[tx_eqn,tx_ini, tx_bnd0,tx_bnd1], outputs=[c_eqn, c_ini, c_bnd0,c_bnd1])
