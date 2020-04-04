#!/usr/bin/env python

# Training beta-VAE, Taoli Cheng, 2019-08-13
# Tenforflow v1 and keras

# [2020-04-03] clean-up and restructure for pre-release

# Input: sequences of jet constituents 4-vecs (pt-ordered) [E1, px1, py1, pz1, E2, px2, py2, pz2, ...]
# Model mode: FCN, LSTM
# stacked LSTM layers, with intermediate LSTM units=50

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import pandas as pd
import itertools

from keras.layers import Lambda, Input, Dense, LSTM, RepeatVector, CuDNNLSTM
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import regularizers, optimizers, objectives
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from utils import load_data, jet_pt, jet_mass
from utils import AnnealingCallback

# To solve CudnnRNN error: Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)
############################################
    
    
def jet_vae_fcn(cfg, beta, weight):
    ''' define FCN-VAE model'''
    
    input_vec = Input(shape =(cfg["input_dim"],))

    encoded = Dense(256, activation='relu')(input_vec)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    #encoded = Dense(24, activation='relu', activity_regularizer=regularizers.l2(5e-4))(encoded)

    z_mean = Dense(cfg["latent_dim"])(encoded)
    z_log_sigma = Dense(cfg["latent_dim"])(encoded)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(cfg["batch_size"], cfg["latent_dim"]),
                                  mean=0., stddev=cfg["epsilon_std"])
        return z_mean + K.exp(0.5*z_log_sigma) * epsilon

    z = Lambda(sampling, output_shape=(cfg["latent_dim"],))([z_mean, z_log_sigma])

    # encoder, from inputs to latent space
    encoder = Model(input_vec, [z_mean, z_log_sigma, z], name='encoder')
    encoder.summary()

    # build decoder model
    latent_inputs = Input(shape=(cfg["latent_dim"],), name='z_sampling')
    decoded = Dense(64, activation='relu')(latent_inputs)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(256, activation='relu')(decoded)

    outputs = Dense(cfg["input_dim"], activation='linear')(decoded)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(input_vec)[2])
    vae = Model(input_vec, outputs, name='vae')

    def kl_loss(x, x_decoded_mean):
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        return kl_loss

    # define loss function
    def vae_loss(x, x_decoded_mean):
        mse_loss = objectives.mse(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        return mse_loss + kl_loss
    
    # define beta loss function
    #beta=cfg["beta"]
    def beta_vae_loss(x, x_decoded_mean):
        mse_loss = objectives.mse(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        return mse_loss + beta*kl_loss

    def annealing_vae_loss(weight):
        def loss(x, x_decoded_mean):
            #mse_loss = objectives.mse(x, x_decoded_mean)
            # adapt loss for 2-d input in LSTM case
            mse_loss = K.mean(objectives.mse(x, x_decoded_mean))
            kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
            return mse_loss + weight*beta*kl_loss
        return loss

    if cfg["annealing"]==True:
        #weight = cfg["weight"]
        vae.compile(optimizer="adam", loss=annealing_vae_loss(weight),
                    metrics=[kl_loss])
    else:
        vae.compile(
        optimizer="adam",
        loss=beta_vae_loss,
        #loss=vae_loss,
        #metrics=[kl_loss]
        )
       
    return vae, encoder, decoder
    #Data parallelism working with multi GPUs within keras
    #parallel_autoencoder = multi_gpu_model(autoencoder, gpus=2)

def jet_vae_lstm(cfg, beta, weight):
    '''LSTM-VAE model'''
    
    input_vec_dim=4 # 4-vec for jet constituents
    timesteps=int(cfg["input_dim"]/input_vec_dim)
    
    inputs = Input(shape=(None, input_vec_dim)) # 20*4
    #encoded = LSTM(latent_dim, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(inputs)
    encoded = CuDNNLSTM(50, return_sequences=True)(inputs) # output, state_h, state_c
    encoded = CuDNNLSTM(cfg["latent_dim"])(encoded) # output, state_h, state_c

    z_mean = Dense(cfg["latent_dim"])(encoded)
    z_log_sigma = Dense(cfg["latent_dim"])(encoded)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(cfg["batch_size"], cfg["latent_dim"]),
                                  mean=0., stddev=cfg["epsilon_std"])
        return z_mean + K.exp(z_log_sigma) * epsilon

    z = Lambda(sampling, output_shape=(cfg["latent_dim"],))([z_mean, z_log_sigma])

    # encoder, from inputs to latent space
    encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
    encoder.summary()


    latent_inputs = Input(shape=(cfg['latent_dim'],), name='z_sampling') # 10
    decoded = RepeatVector(timesteps)(latent_inputs) # 20*10
    #decoded = LSTM(input_vec_dim, return_sequences=True)(decoded)  # for CPU users
    #decoded = CuDNNLSTM(input_vec_dim, return_sequences=True)(decoded) # for GPU users, output: 20*4
    decoded = CuDNNLSTM(50,  return_sequences=True)(decoded) # for GPU users, output: 20*4
    decoded = CuDNNLSTM(input_vec_dim,  return_sequences=True)(decoded) # for GPU users, output: 20*4

    
    # instantiate decoder model
    decoder = Model(latent_inputs, decoded, name='decoder')
    decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2]) # encoder(inputs)[2] --> z
    vae = Model(inputs, outputs, name='vae')

    def kl_loss(x, x_decoded_mean):
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        return kl_loss

    # define loss function
    def vae_loss(x, x_decoded_mean):
        mse_loss = K.mean(objectives.mse(x, x_decoded_mean))
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        return mse_loss + kl_loss
    
    # define beta loss function
    beta=cfg["beta"]
    def beta_vae_loss(x, x_decoded_mean):
        mse_loss = K.mean(objectives.mse(x, x_decoded_mean))
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        return mse_loss + beta*kl_loss

    def annealing_vae_loss(weight):
        def loss(x, x_decoded_mean):
            #xent_loss = objectives.mse(x, x_decoded_mean)
            # adapt loss for 2-d input in LSTM case
            mse_loss = K.mean(objectives.mse(x, x_decoded_mean))
            kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
            return mse_loss + weight*beta*kl_loss
        return loss

    
    if cfg["annealing"]==True:
        print("Compile Annealing Model ---")
        #weight = cfg["weight"]
        vae.compile(optimizer="adam", loss=annealing_vae_loss(weight),
                    metrics=[kl_loss])
    else:
        print("Compile Normal Model ---")
        vae.compile(
        optimizer="adam",
        loss=beta_vae_loss,
        #loss=vae_loss,
        #metrics=[kl_loss]
        )
    
    return vae, encoder, decoder


if __name__ == "__main__":
    print(K.tensorflow_backend._get_available_gpus())

    parser = argparse.ArgumentParser() 
    
    parser.add_argument('--train', default='data/AE_training_qcd_preprocessed_realign.h5',help='path of training set')
    parser.add_argument('--model', default='models/test.h5', help='path of model to be saved')
    parser.add_argument('--input_dim', default=80, type=int, help='length of input jet vectors (# of constituents*4)')
    parser.add_argument('--train_number', default=20000, type=int,  help='size of training set')
    parser.add_argument('--batch_size', default=100)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--vae', default='fcn')
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--epsilon', default=1.0, type=float)
    parser.add_argument('--pt_scaling', default=False, action='store_true')
    parser.add_argument('--annealing', default=False, action='store_true')
    
    args = parser.parse_args()
    
    print(args.annealing)
    print(args.epsilon)
    # define configuration parameters
    cfg={}
    #cfg["LEN_INPUT"]=80
    cfg["input_dim"]=args.input_dim
    cfg["latent_dim"]=10
    cfg["batch_size"]=args.batch_size
    cfg["epsilon_std"]=args.epsilon #1.0e-3 was tested to be a better value than 1
    cfg["intermediate_dim"]=512
    cfg["epochs"]=args.epochs
    cfg["beta"]=args.beta
    cfg["annealing"]=args.annealing
    #cfg["weight"]=K.variable(0.)

    print("traingin beta VAE with beta = %.3f"%args.beta)
    # load training set
    x_train=load_data(args.train, cfg['input_dim'], pt_scaling=args.pt_scaling)
    
    # create model: 
    #    FCN: jet_vae_fcn
    #    LSTM: jet_vae_lstm
    beta=K.variable(cfg["beta"])
    weight=K.variable(0.) # parameter for annealing training

    if args.vae == 'fcn':
        vae, encoder, _ =jet_vae_fcn(cfg, beta, weight)
    elif args.vae == 'lstm':
        vae, encoder, _=jet_vae_lstm(cfg, beta, weight)
        x_train=np.reshape(x_train,(len(x_train), -1, 4)) # reshape 1d array into 2d array with rows are 4-vecs


    if args.annealing == True:
        # training
        print("Start Annealing Training ---")
        #weight=cfg["weight"]
        history=vae.fit(x_train[:args.train_number], x_train[:args.train_number], epochs=cfg['epochs'],
                       batch_size=cfg["batch_size"],
                       shuffle='batch',
                       validation_split=0.2,
                       callbacks=[AnnealingCallback(weight)]
                       )
    else:
        print("Start Normal Training with EarlyStopping ---")
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=8)

        # training
        history=vae.fit(x_train[:args.train_number], x_train[:args.train_number], epochs=cfg['epochs'],
                       batch_size=cfg["batch_size"],
                       shuffle='batch',
                       validation_split=0.2,
                       callbacks=[early_stopping]
                       #callbacks=[early_stopping, check_point]
                       )

    # output results and save the model
    with open(args.model+'.trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    #encoder.save(args.model+'.encoder')
    vae.save(args.model)
    
    
    
