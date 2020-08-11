# VAE-Jet

Code repository for [Variational Autoencoders for Anomalous Jet Tagging](https://arxiv.org/abs/2007.01850).
Talk links: [part1](https://docs.google.com/presentation/d/1t_W5YVQ3GBD0LLw3-wJ96WhtlOJ9sTh9jNY4pGUwl6Q/edit?usp=sharing), [part2](https://docs.google.com/presentation/d/1a8Ej-D2EGTyBdP1xLXqyW4Fz9MLNB3qLd65VW0yAGkc/edit?usp=sharing)

<img src="https://github.com/taolicheng/VAE-Jet/blob/master/figs/VAE.jpg" width="700" height="360">

Variational Autoencoder for jet physics at the Large Hadron Collider. Low-level information of jet constituents are taken as input. We employ simple fully connected networks for embedding architecture. (There is also a LSTM alternative in tensorflow v1). Inputs should be pt-ordered four-vector momenta of jet constituents. VAE is trained on background QCD jets. Then the trained VAE is used to tag anomalous jets as an anomaly detector.

Due to historical reasons, versions for tensorflow v1 and v2 are both presented. In tf2, all training components including basic VAE, DisCo-VAE and OE-VAE are facilitated.

```
└── tf2
    ├── disco.py
    ├── train_disco.py # DisCo-VAE
    ├── train_oe_resample.py # OE-VAE
    ├── train.py # beta-VAE
    └── vae.py
```

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

#### tf2
* OE-VAE:
`./train_oe_resample.py [path to save the model] [pre-trained model, set to null for scratch training]  --n_train [sample number of training set] --beta 0.1 --lam 2.0 --epochs 100 --oe_type 2 --margin 1`
  * options:
     * lam: lambda for OE loss
     * oe_type: 1 MSE-OE; 2 KL-OE
     * mse_oe_type: 1： sigmoid 2: margin
     * margin: `margin` for KL-OE or type-2 MSE-OE

### Testsets

* We released a series of test sets for general performance examination of anomalous jet tagging: https://zenodo.org/record/3774560#.XuG1fs9KgkJ, in which boosted W jets (2-prong), Top jets (3-prong), and Higgs jets (THDM Heavy Higgs, 4-prong decayed) are included.

### Citation

```
@article{Cheng:2020dal,
    author = "Cheng, Taoli and Arguin, Jean-François and Leissner-Martin, Julien and Pilette, Jacinthe and Golling, Tobias",
    title = "{Variational Autoencoders for Anomalous Jet Tagging}",
    eprint = "2007.01850",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "7",
    year = "2020"
}
```
