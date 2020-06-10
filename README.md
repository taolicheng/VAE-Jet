# VAE-Jet

Variational Autoencoder for jet physics at the Large Hadron Collider. Low-level information of jet constituents are taken as input. 

### Dependencies

* tensorflow
* (ROOT): only for specific parts of the code.

### Training
* To train the model:
`./train_betaVAE.py --train [path of training dataset] --model [path to save the model] --train_number [sample number of training set] --epochs 100 --vae 'fcn' --beta 0.1`
* Options
  * pt-scaling: `--pt_scaling`
  * annealing training: `--annealing`
