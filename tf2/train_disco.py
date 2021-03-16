#!/usr/bin/env python

# Taoli Cheng

import h5py
import pickle
import click
import tensorflow as tf
#tf.keras.backend.set_floatx('float64') # to solve data type conflict between float32 and float64
from vae import VariationalAutoEncoder
from disco import distance_corr
from utils import load_data

from sklearn.model_selection import train_test_split


# Training utility functions    
def annealing_fn(epoch, weight):
    start=10
    annealtime=10
    if epoch > start and epoch < start+annealtime :
        weight=min(weight + (1.0/ (annealtime-1)), 1.0)

    if epoch == start+2*annealtime:
        weight=0

    if epoch > start+2*annealtime and epoch < start+3*annealtime:
        weight=min(weight + (1.0/ (annealtime-1)), 1.0)

    if epoch == start+4*annealtime:
        weight=0

    if epoch > start+4*annealtime and epoch < start+5*annealtime:
        weight=min(weight + (1.0/ (annealtime-1)), 1.0)

    if epoch >=5*annealtime:
        weight=1.0

    return weight   

def annealing_fn_2steps(epoch, weight):
    start=10
    annealtime=10
    if epoch > start and epoch < start+annealtime :
        weight=min(weight + (1.0/ (annealtime-1)), 1.0)

    if epoch == start+2*annealtime:
        weight=0

    if epoch > start+2*annealtime and epoch < start+3*annealtime:
        weight=min(weight + (1.0/ (annealtime-1)), 1.0)

    if epoch >= start+4*annealtime:
        weight = 1.0

    return weight  

@click.command()
@click.argument("model_path")
@click.option("--n_train", default=-1)
@click.option("--beta", default=0.1)
@click.option("--lam", default=100.0)
#@click.option("--input_dim", default=80)
@click.option("--epochs", default=50)
@click.option("--annealing", is_flag=True, default=False)
def train(model_path,n_train=-1, beta=0.1, lam=100.0, epochs=100, annealing=False):
    
    qcd_train, scaler=load_data()

    # read in mass
    DATA_DIR="/network/tmp1/taoliche/data/VAE_Final/"
    f=h5py.File(DATA_DIR+"qcd_preprocessed.h5", "r")
    m=f["obs"][:,3]

    # Prepare training and OE datasets
    #n_train=200000
    train_data = qcd_train[:n_train]
    m_train=m[:n_train]

    x_train_in, x_val_in, m_train_in, m_val_in = train_test_split(train_data, m_train, test_size=0.2)

    train_in_dataset = tf.data.Dataset.from_tensor_slices(x_train_in)
    m_train_in_dataset=tf.data.Dataset.from_tensor_slices(m_train_in)
    train_in_dataset=tf.data.Dataset.zip((train_in_dataset, m_train_in_dataset))
    train_in_dataset = train_in_dataset.shuffle(buffer_size=1024).batch(100)
    
    val_in_dataset=tf.data.Dataset.from_tensor_slices(x_val_in)
    m_val_in_dataset=tf.data.Dataset.from_tensor_slices(m_val_in)
    val_in_dataset=tf.data.Dataset.zip((val_in_dataset, m_val_in_dataset))
    val_in_dataset = val_in_dataset.shuffle(buffer_size=1024).batch(100)

    f.close()
    
    # Quick implementing Mass-Deco (var_1 = m, var_2=vae_recon)
    #vae = VariationalAutoEncoder(80, 64, 10)
    vae = VariationalAutoEncoder(80, 10)
    mse_loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    #mse_loss_fn = tf.keras.losses.MeanSquaredError()
    #optimizer = tf.keras.optimizers.Adam()
    
    num_steps = int(n_train*0.8/100)
    lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay([30*num_steps],
        [1e-3, 1e-4])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_fn)

    train_loss_metric=tf.keras.metrics.Mean()
    val_loss_metric=tf.keras.metrics.Mean()
    dc_metric=tf.keras.metrics.Mean()
    dc_val_metric=tf.keras.metrics.Mean()
    
    train_loss_results=[]
    val_loss_results=[]
    dc_results=[]
    dc_val_results=[]

    #epochs = 50
    #beta=0.1
    #lam=1000.0
    
    if annealing:
        lam_weight=0.0
    else:
        lam_weight=1.0   

    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,))
        train_loss_metric.reset_states()
        val_loss_metric.reset_states()
        dc_metric.reset_states()
        dc_val_metric.reset_states()
            
        if annealing:
            lam_weight=annealing_fn_2steps(epoch, lam_weight)    
            
        for step, (train_batch, m_batch) in enumerate(train_in_dataset):
            with tf.GradientTape() as tape:
                reconstructed = vae(train_batch)
                mse_loss=mse_loss_fn(reconstructed, train_batch)
                loss = mse_loss + beta*sum(vae.losses)

                # add distance correlation as regularizer
                if lam_weight == 0 :
                    pass
                else:
                    normed_weight=tf.ones_like(m_batch)
                    #dc=distance_corr(mse_loss, m_batch, normedweight=normed_weight) # decorrelate MSE reconstruction error
                    dc=distance_corr(loss, m_batch, normedweight=normed_weight) # decorrelate beta-loss
                    loss += lam_weight*lam*dc

            # apply gradients
            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))

            train_loss_metric(loss)
            if lam_weight == 0:
                pass
            else:
                dc_metric(dc)


            if step % 200 == 0:
                print("step %s; mean loss %s; DC_loss: %s"%(step, train_loss_metric.result(), dc_metric.result()))
        train_loss_results.append(train_loss_metric.result())
        dc_results.append(dc_metric.result())
        
        # validate
        for (x_batch_val, m_batch) in val_in_dataset:
            reconstructed = vae(x_batch_val)
            val_loss = mse_loss_fn(x_batch_val, reconstructed)
            val_loss += beta*sum(vae.losses)  
            normed_weight=tf.ones_like(m_batch)
            #dc=distance_corr(mse_loss, m_batch, normedweight=normed_weight) # decorrelate MSE reconstruction error
            dc=distance_corr(val_loss, m_batch, normedweight=normed_weight) # decorrelate beta-loss
            val_loss += lam_weight*lam*dc
            
            val_loss_metric(val_loss)
            if lam_weight == 0:
                pass
            else:
                dc_val_metric(dc)

        print('Validation loss: %s' % (val_loss_metric.result()))
        
        # logging validation loss for each epoch
        val_loss_results.append(val_loss_metric.result())
        dc_val_results.append(dc_val_metric.result())
        val_loss_metric.reset_states() 
        dc_val_metric.reset_states()
    
    # save model
    #model_path="models/mass_deco/elbo_lam1000_anneal"
    print("Saving model to %s"%model_path)
    vae.save_weights(model_path, save_format='tf')

    # save training history
    print("Saving training history to %s"%(model_path+'.trainHistoryDict'))
    history={"train_loss": train_loss_results,
             "val_loss": val_loss_results,
            "train_dc": dc_results,
            "val_dc": dc_val_results}
    with open(model_path+'.trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history, file_pi)
    
    
if __name__ == "__main__":
    train()
