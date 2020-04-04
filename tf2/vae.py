
import tensorflow as tf
from tensorflow.keras import layers
tf.keras.backend.clear_session()

import click
import pickle
import h5py
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon # *1.0e-3 for Full Model


class Encoder(layers.Layer):
    """Map inputs to a triplet (z_mean, z_log_var, z)."""

    def __init__(self,
               latent_dim=10,
               intermediate_dim=64,
               name='encoder',
               **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_1 = layers.Dense(256, activation='relu')
        self.dense_2 = layers.Dense(128, activation='relu')
        self.dense_3 = layers.Dense(64, activation='relu')
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded vector, back into input space."""

    def __init__(self,
               original_dim,
               name='decoder',
               **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_1=layers.Dense(64, activation='relu')
        self.dense_2=layers.Dense(128, activation='relu')
        self.dense_3=layers.Dense(256, activation='relu')
        self.dense_output = layers.Dense(original_dim, activation='linear')

    def call(self, inputs):
        x=self.dense_1(inputs)
        x=self.dense_2(x)
        x=self.dense_3(x)
        return self.dense_output(x)
        
class Encoder_0(layers.Layer):
    """Map inputs to a triplet (z_mean, z_log_var, z)."""

    def __init__(self,
               latent_dim=10,
               name='encoder',
               **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_1 = layers.Dense(256, activation='relu')
        self.dense_2 = layers.Dense(128, activation='relu')
        self.dense_3 = layers.Dense(64, activation='relu')
        self.dense_latent = layers.Dense(latent_dim, activation='relu')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_latent(x)
        return x     
        
class AutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""
    def __init__(self,
               original_dim,
               latent_dim=10,
               name='autoencoder',
               **kwargs):
        super(AutoEncoder, self).__init__(name=name, **kwargs)
        self.encoder = Encoder_0(latent_dim=latent_dim)
        self.decoder = Decoder(original_dim=original_dim)

    def call(self, inputs):
        z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        mse_loss = tf.keras.losses.MeanSquaredError(inputs, reconstructed)
        self.add_loss(mse_loss)
        return reconstructed

class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""
    def __init__(self,
               original_dim,
                 latent_dim=10,
               name='autoencoder',
               **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(original_dim=original_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        ######### ATTENTION: changed to keep batch dimension
        kl_loss = - 0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1, axis=1)
        self.add_loss(kl_loss)
        return reconstructed
    
    