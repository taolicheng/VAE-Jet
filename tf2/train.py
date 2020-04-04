#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers
tf.keras.backend.clear_session()

import click
import pickle
import h5py
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

from vae import VariationalAutoEncoder

def load_data():
    pt_scaling=False

    # training data path
    f=h5py.File("./data/AE_training_qcd_preprocessed_realignphi.h5", "r")

    # mass labels for mass-decorrelation
    #mass_labels=f['mass_labels']
    #one_hot_mlabels = tf.keras.utils.to_categorical(np.array(mass_labels)-1, num_classes=10)

    #qcd_train=f["constituents" if "constituents" in f.keys() else "table"]

    for key in ['table', 'constituents', 'jet1']:
        if key in f.keys():
            qcd_train=f[key]
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
    #pt_scaling=False
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

    if epoch >=5*kl_annealtime:
        weight=1.0

    return weight    


@click.command()
@click.argument("model_path")
@click.option("--n_train", default=-1)
@click.option("--beta", default=1.0)
@click.option("--input_dim", default=80)
@click.option("--epochs", default=50)
@click.option("--annealing", is_flag=True, default=False)
def train_vae(model_path, n_train=-1, beta=1.0, input_dim=80, epochs=50, annealing=False):

    # load training data
    train_data, _=load_data()
    train_data=train_data[:n_train]
    x_train_in, x_val_in = train_test_split(train_data,test_size=0.2)
    
    train_in_dataset = tf.data.Dataset.from_tensor_slices(x_train_in)
    train_in_dataset = train_in_dataset.shuffle(buffer_size=1024).batch(100)
    
    val_dataset=tf.data.Dataset.from_tensor_slices(x_val_in)
    val_dataset=val_dataset.shuffle(buffer_size=1024).batch(100)
    
    # Initialize the model
    vae = VariationalAutoEncoder(input_dim, 10)
    
    optimizer = tf.keras.optimizers.Adam()
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    # metrics for plotting learning curves
    loss_metric = tf.keras.metrics.Mean()
    kl_loss_metric=tf.keras.metrics.Mean()
    mse_loss_metric=tf.keras.metrics.Mean()
    val_loss_metric=tf.keras.metrics.Mean()

    train_loss_results=[]
    mse_loss_results=[]
    kl_loss_results=[]
    val_loss_results=[]
    
    # weight initialization for annealing training
    if annealing:
        weight=0.0
    else:
        weight=1.0
        
    for epoch in range(epochs):

        if annealing:
            weight=annealing_fn(epoch, weight)

        print('Start of epoch %d, with beta weight = %.3f' % (epoch,weight))

        # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(train_in_dataset):
            with tf.GradientTape() as tape:
                reconstructed = vae(x_batch_train)
                # Compute reconstruction loss
                loss_mse = mse_loss_fn(x_batch_train, reconstructed)
                loss_kl = sum(vae.losses) 
                loss = loss_mse + weight*beta*loss_kl # Add KLD regularization loss

            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))

            loss_metric(loss)
            mse_loss_metric(loss_mse)
            kl_loss_metric(loss_kl)

            # logging mse and kl loss for each iteration
            mse_loss_results.append(mse_loss_metric.result())
            kl_loss_results.append(kl_loss_metric.result())

            if step % 200 == 0:
                  print('step %s: mean mse loss = %s, kl mean loss %s' % (step, mse_loss_metric.result(), kl_loss_metric.result()))

        # logging training loss for each epoch
        train_loss_results.append(loss_metric.result())

        loss_metric.reset_states()
        mse_loss_metric.reset_states()
        kl_loss_metric.reset_states()

        # validate
        for x_batch_val in val_dataset:
            reconstructed = vae(x_batch_val)
            val_loss = mse_loss_fn(x_batch_val, reconstructed)
            val_loss += beta*sum(vae.losses)  

            val_loss_metric(val_loss)
        print('Validation loss: %s' % (val_loss_metric.result()))
        
        # logging validation loss for each epoch
        val_loss_results.append(val_loss_metric.result())
        val_loss_metric.reset_states() 
    
    # save model
    print("Saving model to %s"%model_path)
    vae.save_weights(model_path, save_format='tf')

    # save training history
    print("Saving training history to %s"%(model_path+'.trainHistoryDict'))
    history={"train_loss": train_loss_results,
            "val_loss": val_loss_results,
            "mse_loss": mse_loss_results,
            "kl_loss": kl_loss_results}
    with open(model_path+'.trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history, file_pi)
    
if __name__ == "__main__":
    train_vae()
