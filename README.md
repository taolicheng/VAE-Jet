# VAE-Jet

<img src="https://github.com/taolicheng/VAE-Jet/figs/VAE.jpg" width="600" height="400">

Variational Autoencoder for jet physics at the Large Hadron Collider. Low-level information of jet constituents are taken as input. We employ simple fully connected networks for embedding architecture. (There is also a LSTM alternative in tensorflow v1). Inputs should be pt-ordered four-vector momenta of jet constituents.

Due to historical reasons, versions for tensorflow v1 and v2 are both presented. In tf2, all training components including basic VAE, DisCo-VAE and OE-VAE are facilitated.

### Dependencies

* tensorflow: `tensorflow-gpu 1.13.1` or `tensorflow-gpu 2.1.0`
* (ROOT): only for specific parts of the code.

### Training

#### tf1
* To train the model:
`./train_betaVAE.py --train [path of training dataset] --model [path to save the model] --train_number [sample number of training set] --epochs 100 --vae 'fcn' --beta 0.1`
* Options
  * pt-scaling: `--pt_scaling`
  * annealing training: `--annealing`

### Testsets

* We released a series of test sets for general performance examination of anomalous jet tagging: https://zenodo.org/record/3774560#.XuG1fs9KgkJ, in which boosted W jets (2-prong), Top jets (3-prong), and Higgs jets (THDM Heavy Higgs, 4-prong decayed) are included.
