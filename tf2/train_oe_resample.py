#!/usr/bin/env python

# Author: Taoli Cheng

import tensorflow as tf
#tf.keras.backend.set_floatx('float64')
from tensorflow.keras import layers
tf.keras.backend.clear_session()

import click
import pickle
import h5py
import numpy as np
from hep_ml import reweight  

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from utils import jet_pt, jet_mass
from vae import VariationalAutoEncoder

DATA_DIR="/network/tmp1/taoliche/data/VAE_Final/"


def load_data():
    DATA_DIR="/network/tmp1/taoliche/data/VAE_Final/"
    pt_scaling=False
    f=h5py.File(DATA_DIR+"qcd_preprocessed.h5", "r")

    #f=h5py.File("./data/AE_training_qcd_preprocessed_realignphi.h5", "r")

    # mass labels for mass-decorrelation
    #mass_labels=f['mass_labels']
    #one_hot_mlabels = tf.keras.utils.to_categorical(np.array(mass_labels)-1, num_classes=10)

    qcd_train=f["constituents" if "constituents" in f.keys() else "table"]
    qcd_train=qcd_train[:,:80]    

    # pt-rescale [! Temporary solution. Performance should be improved!] 
    if pt_scaling:
         for i in range(len(x_train)):
            pt=jet_pt(x_train[i])
            x_train[i]=x_train[i]/pt

    # Robust Scaling
    scaler=RobustScaler().fit(qcd_train)
    qcd_train=scaler.transform(qcd_train)
    f.close()
    return qcd_train, scaler
    
# Load Test data
def load_test(scaler, fn, pt_scaling=False, pt_refine=True):
    from utils import jet_pt
    #DATA_DIR="/network/tmp1/taoliche/data/VAE_Final/"
    #pt_refine=False # restrict jet pt to [550, 650]

    f=h5py.File(fn, 'r')

    ########w_test=f[key for key in ['table', 'constituents', 'jet1'] if key in f.keys()]
    for key in ['table', 'constituents', 'jet1']:
        if key in f.keys():
            w_test=f[key]
            if key == "jet1":
                labels=f["labels"]
                labels=np.array(labels)
                
    ######## Select pt range [550, 650] #########
    if pt_refine:
        from utils import jet_pt, jet_mass
        pts=[]
        for j in w_test:
            pts.append(jet_pt(j))
        w_test=np.array(w_test)
        pts=np.array(pts)
        w_test=w_test[(pts>550)&(pts<=650)]
    ############################################

    w_test=w_test[:,:80]    
    # pt-rescale [! Temporary solution. Performance should be improved!] 

    if pt_scaling:
         for i in range(len(w_test)):
            pt=jet_pt(w_test[i])
            w_test[i]=w_test[i]/pt

    w_test=scaler.transform(w_test)

    f.close()
    return w_test

############ Load OE data ####################
def load_oe_data_resample(scaler):
    def load_w_oe(name, input_dim=80):
        DATA_DIR="/network/tmp1/taoliche/data/VAE_Final/"
        fn=DATA_DIR+name
        f=h5py.File(fn, 'r')
        ########w_test=f[key for key in ['table', 'constituents', 'jet1'] if key in f.keys()]
        for key in ['table', 'constituents', 'jet1']:
            if key in f.keys():
                w_test=f[key]
                if key == "jet1":
                    labels=f["labels"]
                    labels=np.array(labels) 
        w_test=w_test[:,:input_dim]    
        w_test=scaler.transform(w_test)
        f.close()
        return w_test
    
    w_test=load_w_oe("resamples_oe_w.h5")
    from sklearn.utils import shuffle
    w_test=shuffle(w_test)
    print("-----------Loaded %i samples for OE Training----------"%w_test.shape[0])
    return w_test

# Training utility functions    
def annealing_fn(epoch, weight):
    klstart=0
    kl_annealtime=5
    if epoch > klstart and epoch < klstart+kl_annealtime :
        weight=min(weight + (1.0/ (kl_annealtime-1)), 1.0)

    if epoch == klstart+2*kl_annealtime:
        weight=0

    if epoch > klstart+2*kl_annealtime and epoch < klstart+3*kl_annealtime:
        weight=min(weight + (1.0/ (kl_annealtime-1)), 1.0)

    if epoch == klstart+4*kl_annealtime:
        weight=0

    if epoch > klstart+4*kl_annealtime and epoch < klstart+5*kl_annealtime:
        weight=min(weight + (1.0/ (kl_annealtime-1)), 1.0)

    if epoch >=klstart+5*kl_annealtime:
        weight=1.0

    return weight
    
    
@click.command()
@click.argument("model_path")
@click.argument("oe_path")
@click.option("--n_train", default=-1)
@click.option("--beta", default=1.0)
@click.option("--lam", default=2.0)
@click.option("--input_dim", default=80)
@click.option("--hidden_dim", default=10)
@click.option("--epochs", default=50)
@click.option("--annealing", is_flag=True, default=False)            
@click.option("--oe_scenario", default=1)
@click.option("--oe_type", default=1) # 1: MSE; 2:KL
@click.option("--mse_oe_type", default=1) # 1: sigmoid; 2: margin
@click.option("--margin", default=2.0)
#@click.option("--vae_trained") to be implemented later
def train_oe(model_path, oe_path, n_train=-1, beta=1.0, lam=2.0, input_dim=80, hidden_dim=10, epochs=50, annealing=False, oe_scenario=1, oe_type=1, mse_oe_type=1, margin=2.0):
    import tensorflow_addons as tfa
    
    # Prepare training and OE datasets
    
    # load training data
    train_data, scaler=load_data()
    train_data=train_data[:n_train]
    x_train_in, x_val_in = train_test_split(train_data,test_size=0.2)
    
    oe_data=load_oe_data_resample(scaler)    
    
    ######################################################
    #n_train=200000
    #n_oe=40000
    n_oe=int(len(x_train_in)/4)
    x_train_oe = oe_data[:n_oe]
    
    train_in_dataset = tf.data.Dataset.from_tensor_slices(x_train_in)
    train_in_dataset = train_in_dataset.shuffle(buffer_size=1024).batch(100)
    
    weights=np.ones(n_oe)

    train_oe_dataset=tf.data.Dataset.from_tensor_slices(x_train_oe)
    weights_oe_dataset=tf.data.Dataset.from_tensor_slices(weights)
    train_oe_dataset=tf.data.Dataset.zip((train_oe_dataset, weights_oe_dataset))
    train_oe_dataset=train_oe_dataset.shuffle(buffer_size=1024).repeat(4).batch(100) 
    
    ####### Validation Dataset ##################
    val_dataset=tf.data.Dataset.from_tensor_slices(x_val_in)
    val_dataset=val_dataset.shuffle(buffer_size=1024).batch(100)
    
    #scenario=1 # 1: from scratch; 2: fine-tuning

    ############# Define the Model: with Outlier Exposure ###############
    original_dim = 80
    ############# Scenario I: from scratch
    if oe_scenario == 1:
        vae_oe = VariationalAutoEncoder(original_dim, hidden_dim)
    ############# Scenario II: fine-tune
    elif oe_scenario == 2:
        vae_oe=vae_trained
    
    optimizer = tf.keras.optimizers.Adam(lr=1e-3)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    
    # define outlier loss term for MSE
    def loss_mse_oe_fn(x_in, x_oe, xhat_in, xhat_oe, weights_oe, mse_oe_type, margin):
        loss_recon_in = tf.keras.losses.MSE(x_in, xhat_in)
        loss_recon_oe = tf.keras.losses.MSE(x_oe, xhat_oe)
        
        # sigmoid
        if mse_oe_type == 1:
            loss=tf.keras.activations.sigmoid(loss_recon_oe-loss_recon_in) # without reweighting
        # margin loss, margin=1, 2
        elif mse_oe_type ==2:
            loss=-tf.keras.activations.relu(loss_recon_in-loss_recon_oe+margin) # unweighted
        return tf.reduce_mean(loss)

    # OE loss in latent space    
    def loss_kl_oe_fn(z_mean_0, z_log_var_0, z_mean_1, z_log_var_1, weights_oe, margin):
        ############### To be implemented ################
        # margin loss: kl_oe - kl_in > margin
        def kl_loss_fn(z_mean, z_log_var):
                kl_loss = - 0.5 * (
                    z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
                return kl_loss

        loss=-tf.keras.activations.relu(kl_loss_fn(z_mean_0, z_log_var_0) - 
                                        kl_loss_fn(z_mean_1, z_log_var_1)+margin)

        return tf.reduce_mean(loss)

    loss_metric = tf.keras.metrics.Mean()
    loss_oe_metric=tf.keras.metrics.Mean()
    
    ### Logging training history    
    loss_metric = tf.keras.metrics.Mean()
    loss_in_metric = tf.keras.metrics.Mean()
    loss_oe_metric=tf.keras.metrics.Mean()
    val_loss_metric=tf.keras.metrics.Mean()
    
    train_loss_results=[]
    oe_loss_results=[]
    in_loss_results=[]
    val_loss_results=[]

    ############ Parameters
    weight_lam=0. # initial OE loss weight
    #beta=0.1 # D_KL loss weight    
    #epochs=30
    
    ############ Start training loops
    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,), "beta = %.3f"%beta, 'lambda=%f'%(weight_lam*lam))

        loss_metric.reset_states()
        loss_oe_metric.reset_states()
        loss_in_metric.reset_states()

        ##### set OE training schedule (annealing)
        #weight_lam+=0.25
        it = iter(train_oe_dataset)
        #if epoch % 5 ==0:
        #    weight_lam=0.0
        
        weight_lam=annealing_fn(epoch, weight_lam) 
        
        # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(train_in_dataset):
            x_batch_test, weights_batch = it.next()
            weights_batch = tf.ones_like(weights_batch)
            # To avoid training instability
            # weights_batch=tf.clip_by_value(weights_batch, clip_value_min=1e-1, clip_value_max=10) # only for DNN reweighter

            with tf.GradientTape() as tape:
                reconstructed = vae_oe(x_batch_train)
                # Compute reconstruction loss
                loss_in = mse_loss_fn(x_batch_train, reconstructed)
                loss_in += beta*sum(vae_oe.losses)  # Add KLD regularization loss

                # OE losses 
                if oe_type == 1:
                # input space
                    reconstructed_oe=vae_oe(x_batch_test)
                    loss_oe=loss_mse_oe_fn(x_batch_train, x_batch_test, reconstructed, reconstructed_oe, weights_oe=weights_batch,  mse_oe_type =  mse_oe_type, margin=margin)
                    #loss = loss_in - lam*loss_oe
                    loss = loss_in - weight_lam*lam*loss_oe
                    
                # latent space KL
                elif oe_type == 2:
                    z_mean_1, z_log_var_1, _ = vae_oe.encoder(x_batch_test)
                    z_mean_0, z_log_var_0, _ = vae_oe.encoder(x_batch_train)
                    loss_oe=loss_kl_oe_fn(z_mean_0, z_log_var_0, z_mean_1, z_log_var_1, weights_oe=weights_batch, margin=margin)
                    loss = loss_in - weight_lam*lam*loss_oe

            grads = tape.gradient(loss, vae_oe.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae_oe.trainable_weights))

            loss_metric(loss)
            loss_in_metric(loss_in)
            loss_oe_metric(loss_oe)

            if step % 200 == 0:
                  print('step %s: mean train loss = %s, in loss %s, oe loss %s' % 
                        (step, loss_metric.result(), loss_in_metric.result(), loss_oe_metric.result()))
        train_loss_results.append(loss_metric.result())
        in_loss_results.append(loss_in_metric.result())
        oe_loss_results.append(loss_oe_metric.result())
        
        # validate
        for x_batch_val in val_dataset:
            reconstructed = vae_oe(x_batch_val)
            val_loss = mse_loss_fn(x_batch_val, reconstructed)
            val_loss += beta*sum(vae_oe.losses)  

            val_loss_metric(val_loss)
        print('Validation loss: %s' % (val_loss_metric.result()))
        
        # logging validation loss for each epoch
        val_loss_results.append(val_loss_metric.result())
        val_loss_metric.reset_states() 
        

    # save model
    print("Saving model to %s"%model_path)
    vae_oe.save_weights(model_path, save_format='tf')

    # save training history
    print("Saving training history to %s"%(model_path+'.trainHistoryDict'))
    history={"train_loss": train_loss_results,
            "val_loss": val_loss_results,
            "in_loss": in_loss_results,
            "oe_loss": oe_loss_results}
    with open(model_path+'.trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history, file_pi)    
    
    
if __name__ == "__main__":
    #train_vae()
    train_oe()
