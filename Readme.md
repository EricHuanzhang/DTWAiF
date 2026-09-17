# WAVELET-GUIDED ASYMMETRIC ATTENTION FOR MULTIVARIATE TIME-SERIES ANOMALY DETECTION

## Introduction

**DTWAiF** is a **d**ual-**t**rack **w**avelet **a**symmetric **i**nverted **T**ransformer for multivariate time-series anomaly detection. It uses wavelet approximation coefficients as coarse context to generate queries, keys, and a learned channel mask for cross-channel attention, while a separate content track preserves all wavelet bands for value projection and reconstruction. Heterogeneous normalization handles continuous measurements and discrete control signals differently, and anomaly scores combine time- and wavelet-domain residuals with distibution drift correction. 

![architecture](D:/Research/论文/DTWAiF/论文正文图/architechture/正式版/architecture.png)

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
│   └── channel_mask.py                        Gumbel-softmax channel mask
└── utils/
    ├── loss.py                                Sobolev, wavelet and projection losses
    ├── ch_discover_loss.py                    contrastive channel regulariser
    └── tools.py                               POT threshold, early stopping, LR schedule

tools/                                         equivalence-verification scripts
```

## Quickstart

### Installation

Given a python environment (**note**: this project is fully tested under python 3.8), install the dependencies with the following command:

```
pip install -r requirements.txt
```

### Data preparation

Datasets follow TFB's layout: one CSV per series under `dataset/`. The nine datasets used in the paper — Genesis, PSM, SWaT, GECCO, CalIt2, NYC, MSL, SMAP, SMD — come from the TFB and CATCH distributions; see their repositories for download instructions and terms. You can also obtained the datasets from [GoogleDrive](https://drive.google.com/file/d/1N_SGBo7ZVCFoHEWsdAkfsHyv4A77clLU/view?usp=sharing)  directly. (This may take some time, please wait patiently.) Then place the downloaded data under the folder `./dataset`.

### Train and evaluate model

We provide the experiment scripts for DTWAiF under the folder `./scripts/multivariate_detection`. For example you can reproduce a experiment result as the following:

```
sh ./scripts/multivariate_detection/detect_label/MSL_script/DTWAiF.sh

sh ./scripts/multivariate_detection/detect_score/MSL_script/DTWAiF.sh
```



## Contact

If you have any questions or suggestions, feel free to contact:

- Eric Cao caoj20@fudan.edu.cn



Or describe it in Issues.