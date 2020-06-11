# VAE-Jet

Variational Autoencoder for jet physics at the Large Hadron Collider. Low-level information of jet constituents are taken as input. We employ simple fully connected networks for embedding architecture. (There is also a LSTM alternative in tensorflow v1). Inputs should be pt-ordered four-vector momenta of jet constituents.

### Dependencies

* tensorflow
* (ROOT): only for specific parts of the code.

### Training
* To train the model:
`./train_betaVAE.py --train [path of training dataset] --model [path to save the model] --train_number [sample number of training set] --epochs 100 --vae 'fcn' --beta 0.1`
* Options
  * pt-scaling: `--pt_scaling`
  * annealing training: `--annealing`

### Testsets

* We released a series of test sets for general performance examination of anomalous jet tagging: https://zenodo.org/record/3774560#.XuG1fs9KgkJ, in which boosted W jets (2-prong), Top jets (3-prong), and Higgs jets (THDM Heavy Higgs, 4-prong decayed) are included.
