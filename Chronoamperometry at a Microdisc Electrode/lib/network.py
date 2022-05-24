import tensorflow as tf


class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self,units,activation='tanh',kernel_initializer='he_normal',**kwagrs):
        super().__init__(**kwagrs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            tf.keras.layers.Dense(units,activation=activation, kernel_initializer=kernel_initializer),
            tf.keras.layers.Dense(units,activation=activation,kernel_initializer=kernel_initializer)
        ]
        self.skip_layers = [
            #tf.keras.layers.Dense(units,activation=activation, kernel_initializer=kernel_initializer),
        ]

    def call(self,inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z+skip_Z)
        



class Network:

    @classmethod
    def build(cls, num_inputs=4, layers=[256,128,64, 64, 64, 128], activation='tanh', num_outputs=1):

        # input layer
        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        # hidden layers
        x = inputs
        
        for layer in layers:
            x = tf.keras.layers.Dense(layer, activation=activation,
                kernel_initializer='he_normal')(x)
        
        """
        x = tf.keras.layers.Dense(32, activation=activation,
            kernel_initializer='he_normal')(x)
        
        for layer in layers:
            x = ResidualUnit(layer, activation=activation,
                kernel_initializer='he_normal')(x)
        """
        # output layer
        outputs = tf.keras.layers.Dense(num_outputs,
            kernel_initializer='he_normal')(x)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs)
