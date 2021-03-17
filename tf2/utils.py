
import numpy as np
import h5py
from sklearn.preprocessing import RobustScaler

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

def load_data():
    pt_scaling=False

    # training data path
    DATA_DIR="/network/tmp1/taoliche/data/VAE_Final/"
    f=h5py.File(DATA_DIR+"qcd_preprocessed.h5", "r")

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
