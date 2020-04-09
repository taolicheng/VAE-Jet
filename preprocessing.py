# ### Notes
# * need to install ROOT with pyROOT enabled. Right now this only works in python2.

import numpy as np
from rootpy.vector import LorentzVector
import h5py
import copy

def preprocessing(jet):
    jet = copy.deepcopy(jet)

    jet=jet.reshape(-1,4)
    n_consti=len(jet)

    # find the jet (eta, phi)
    center=jet.sum(axis=0)

    v_jet=LorentzVector(center[1], center[2], center[3], center[0])

    # centering parameters
    phi=v_jet.phi()
    bv = v_jet.boost_vector()
    bv.set_perp(0)    

    for i in range(n_consti):
        v = LorentzVector(jet[i,1], jet[i,2], jet[i,3], jet[i,0])
        v.rotate_z(-phi)
        v.boost(-bv)  
        jet[i, 0]=v[3] #e
        jet[i, 1]=v[0] #px
        jet[i, 2]=v[1] #py
        jet[i, 3]=v[2] #pz

    # rotating parameters
    weighted_phi=0
    weighted_eta=0
    for i in range(n_consti):
        if jet[i,0]<1e-10: # pass zero paddings
            continue
        v = LorentzVector(jet[i,1], jet[i,2], jet[i,3], jet[i,0])
        r=np.sqrt(v.phi()**2 + v.eta()**2)
        if r == 0: # in case there is only one component. In fact these data points should generally be invalid.
            continue
        weighted_phi += v.phi() * v.E()/r
        weighted_eta += v.eta() * v.E()/r
    #alpha = np.arctan2(weighted_phi, weighted_eta) # approximately align at eta
    alpha = np.arctan2(weighted_eta, weighted_phi) # approximately align at phi

    for i in range(n_consti):
        v = LorentzVector(jet[i,1], jet[i,2], jet[i,3], jet[i,0])
        #v.rotate_x(alpha) # approximately align at eta
        v.rotate_x(-alpha) # approximately align at phi

        jet[i, 0]=v[3]
        jet[i, 1]=v[0]
        jet[i, 2]=v[1]
        jet[i, 3]=v[2]

    #jet=jet.reshape(1,-1)
    jet=jet.ravel()
    return jet
