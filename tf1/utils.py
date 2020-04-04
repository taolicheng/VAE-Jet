import h5py
import numpy as np
from keras import backend as K
from keras.callbacks import Callback


from sklearn.preprocessing import RobustScaler
    
import ot
from energyflow import emd
from preprocessing import jet_4v_translate_py3
import itertools
    

# Simple Jet Observables
def jet_mass(jet):
    E_j=0
    Px_j=0
    Py_j=0
    Pz_j=0

    jet=np.reshape(jet, (-1, 4))
    #n_consti=len(jet)
    
    E_j, Px_j, Py_j, Pz_j = np.sum(jet, axis=0)
    
    if E_j**2 > (Px_j**2 + Py_j**2 + Pz_j**2):
            m=np.sqrt(E_j**2 - (Px_j**2 + Py_j**2 + Pz_j**2))
    else:
            m=0

    return m

def jet_pt(jet):
    Px_j=0
    Py_j=0

    jet=np.reshape(jet, (-1, 4))
    n_consti=len(jet)

    for i in range(n_consti):
            Px_j+=jet[i, 1]
            Py_j+=jet[i ,2]
            
    pt=np.sqrt(Px_j**2 + Py_j**2)
    return pt

def load_scaler(filename, key="table", len_input=80, pt_scaling=False):
    '''
    load scaler (training data) for model evaluation or inference.
    
    len_input:
       length of input jet vectors
    pt_scaling:
        0: no pt_scaling
        1: pt-scaled input: (E, Px, Py, Pz) / Jet_Pt
    '''
    # load hdf5 dataset: data/AE_training_qcd.h5
    f_train=h5py.File(filename,'r')
    for key in ['table', 'constituents', 'jet1']:
        if key in f_train.keys():
            x_train=f_train[key]
    #x_train=f_train['table' if 'table' in f_train.keys() else 'constituents']
    #if key not in f_train.keys():
    #    print("input correct key for dataset")
    #    return
    #x_train=f_train[key]

    x_train=x_train[:,:len_input]

    if pt_scaling:
         for i in range(len(x_train)):
            pt=jet_pt(x_train[i])
            x_train[i]=x_train[i]/pt
    
    # Robust Scaling
    scaler=RobustScaler().fit(x_train)
    return scaler

def load_data(filename, len_input=80, pt_scaling=False):
    '''
    len_input
       length of input jet vectors
    pt_scaling
        0: no pt_scaling
        1: pt-scaled input: (E, Px, Py, Pz) / Jet_Pt
    '''
    # set input length
    #len_input=80

    # load hdf5 dataset: data/AE_training_qcd.h5
    f_train=h5py.File(filename,'r')
    for key in ['table', 'constituents', 'jet1']:
        if key in f_train.keys():
            x_train=f_train[key]

    #x_train=f_train['table' if 'table' in f_train.keys() else 'constituents']
    #x_train=f_train['constituents']
    #x_validation=f_validation['table']
    
    # zero-padding added to accommodate variable length input 
    #x_train=np.array(list(itertools.zip_longest(*x_train, fillvalue=0))).T

    x_train=x_train[:,:len_input]
    
    # pt-rescale [! Temporary solution. Performance should be improved!] 
    if pt_scaling:
         for i in range(len(x_train)):
            pt=jet_pt(x_train[i])
            x_train[i]=x_train[i]/pt
    
    # Robust Scaling
    scaler=RobustScaler().fit(x_train)
    x_train=scaler.transform(x_train)
    
    f_train.close()
    return x_train

def load_test(filename, scaler, file_type=0, len_input=80):
    '''
    filename: input h5 filename
    file_type: in order to cope with different data sources. Right now, we have
        0: Vanilla data formatting (Top Reference Dataset). 
             * jets=f["table"]
             * zero-padded
             * 200 constituents
        1: Pre-recnn data formatting (Gilles and Taoli). 
             * jets=f["constituents"]
             * variable length
        2: Pre-LPS data formatting (Taoli and Julien).
             * jets=f["objs"]["jets"]["constituents"][:,0] # [:, 0] leading-pt jet; [:, 1] sub-leading-pt jet
             * variable length
             * event-level storage
    '''
    f_test=h5py.File(filename, 'r')
    
    if file_type == 0:
        x_test = f_test['table']
    elif file_type == 1:
        x_test = f_test['constituents']
    elif file_type == 2:
        x_test = f_test["objs"]["jets"]["constituents"][:,0]
        
    x_test=scaler.transform(x_test)
    
    return x_test


def calc_emd(j1,j2):
    ''' 
    calculating emds for input jets arrray with 4-vec formatting
    len(j1) should be equal to len(j2)
    
    '''
    # translate 4-vec [E, px, py, pz] into [pt, eta, phi, m]
    j1_trans=[]
    j2_trans=[]
    for i in range(len(j1)):
        j1_trans.append(jet_4v_translate_py3(j1[i]))
    for i in range(len(j2)):
        j2_trans.append(jet_4v_translate_py3(j2[i]))
        
    # reshape and extract [pt, eta, phi] 
    j1_emd=[]
    j2_emd=[]
    for i in range(len(j1)):
        j1_emd.append(j1_trans[i].reshape(-1,4)[:,0:3])
    for i in range(len(j2)):
        j2_emd.append(j2_trans[i].reshape(-1,4)[:,0:3])
    
    emds_array=[]
    for i in range(len(j1)):
        emds_array.append(emd.emd(j1_emd[i], j2_emd[i], R=1.0, norm=True, measure='euclidean', coords='hadronic',
                      return_flow=False, gdim=None, mask=False, n_iter_max=100000,
                      periodic_phi=False, phi_col=2, empty_policy='error'))
        
    emds_array=np.array(emds_array)
    
    return emds_array


class AnnealingCallback(Callback):
    def __init__(self, weight):
        self.weight = weight
        self.klstart=0
        self.kl_annealtime=30
    def on_epoch_end (self, epoch, logs={}):
        klstart=self.klstart
        kl_annealtime=self.kl_annealtime
        '''cycle-1'''
        if epoch > klstart and epoch < klstart+kl_annealtime :
            new_weight = min(K.get_value(self.weight) + (1./ (kl_annealtime-1)), 1.)
            K.set_value(self.weight, new_weight)

        '''cycle-2'''
        if epoch == klstart+2*kl_annealtime:
            K.set_value(self.weight, 0.)
            
        if epoch > klstart+2*kl_annealtime and epoch < klstart+3*kl_annealtime:
            new_weight = min(K.get_value(self.weight) + (1./ (kl_annealtime-1)), 1.)
            K.set_value(self.weight, new_weight)
            
        '''cycle-3'''
        if epoch == klstart+4*kl_annealtime:
            K.set_value(self.weight, 0.)
            
        if epoch > klstart+4*kl_annealtime and epoch < klstart+5*kl_annealtime:
            new_weight = min(K.get_value(self.weight) + (1./ (kl_annealtime-1)), 1.)
            K.set_value(self.weight, new_weight) 
            
        '''cycle-4'''
        #if epoch == klstart+6*kl_annealtime:
        #    K.set_value(self.weight, 0.)
            
        #if epoch > klstart+6*kl_annealtime and epoch < klstart+7*kl_annealtime:
        #    new_weight = min(K.get_value(self.weight) + (1./ (kl_annealtime-1)), 1.)
        #    K.set_value(self.weight, new_weight) 
            
        print ("Current KL Weight is " + str(K.get_value(self.weight)))
