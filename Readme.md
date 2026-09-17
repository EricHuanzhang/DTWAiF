# WAVELET-GUIDED ASYMMETRIC ATTENTION FOR MULTIVARIATE TIME-SERIES ANOMALY DETECTION

## Introduction

**DTWAiF** is a **d**ual-**t**rack **w**avelet **a**symmetric **i**nverted **T**ransformer for multivariate time-series anomaly detection. It uses wavelet approximation coefficients as coarse context to generate queries, keys, and a learned channel mask for cross-channel attention, while a separate content track preserves all wavelet bands for value projection and reconstruction. Heterogeneous normalization handles continuous measurements and discrete control signals differently to reduce distortion associated with uniform variance-based scaling. Anomaly scores combine time- and wavelet-domain residuals with drift correction. 

<img width="2638" height="997" alt="architecture" src="https://github.com/user-attachments/assets/d3c36761-cc95-40ff-9842-697d1638c980" />


## Method

### Architecture

```
x [B, L, C]
  └─ heterogeneous per-channel scaling
  └─ DWT ─────────────────────────────────────────────┐
       ├─ content track : all bands  ──► values       │
       └─ context track : cA only    ──► queries/keys │
                                        └─ channel mask
  └─ masked cross-channel attention  (context fixed across layers)
  └─ per-band decoding
  └─ inverse DWT ──► reconstruction
```



### Repository layout

```
ts_benchmark/baselines/DTWAiF/
├── DTWAiF.py                                  detector: training, scoring, thresholding
├── models/DTWAiF_model.py                     the dual-track wavelet reconstructor
├── layers/
│   ├── cross_channel_bandwise_Transformer.py  band-wise two-track attention
│   └── channel_mask.py                        channel mask
└── utils/
    ├── loss.py                                time, wavelet losses
    ├── ch_discover_loss.py                    contrastive channel regulariser
    └── tools.py                               POT threshold, early stopping
```

## Quickstart

### Installation

Given a python environment (**note**: this project is fully tested under python 3.8), install the dependencies with the following command:

```
pip install -r requirements.txt
```

### Data preparation

You can obtained the datasets from [GoogleDrive](https://drive.google.com/file/d/1N_SGBo7ZVCFoHEWsdAkfsHyv4A77clLU/view?usp=sharing)  directly. (This may take some time, please wait patiently.) Then place the downloaded data under the folder `./dataset`.

### Train and evaluate model

We provide the experiment scripts for DTWAiF under the folder `./scripts/multivariate_detection`. For example you can reproduce a experiment result as the following:

```
sh ./scripts/multivariate_detection/detect_label/Genesis_script/DTWAiF.sh

sh ./scripts/multivariate_detection/detect_score/Genesis_script/DTWAiF.sh
```



## Contact

If you have any questions or suggestions, feel free to contact:

- Eric Cao caoj20@fudan.edu.cn



Or describe it in Issues.
