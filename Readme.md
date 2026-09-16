# WAVELET-GUIDED ASYMMETRIC ATTENTION FOR MULTIVARIATE TIME-SERIES ANOMALY DETECTION

## Introduction

**DTWAiF** is a **d**ual-**t**rack **w**avelet **a**symmetric **i**nverted **T**ransformer for multivariate time-series anomaly detection. It uses wavelet approximation coefficients as coarse context to generate queries, keys, and a learned channel mask for cross-channel attention, while a separate content track preserves all wavelet bands for value projection and reconstruction. Heterogeneous normalization handles continuous measurements and discrete control signals differently, and anomaly scores combine time- and wavelet-domain residuals with distibution drift correction. 

![architecture](D:/Research/论文/DTWAiF/论文正文图/architechture/正式版/architecture.png)

## Quickstart

### Installation

Given a python environment (**note**: this project is fully tested under python 3.8), install the dependencies with the following command:

```
pip install -r requirements.txt
```

### Data preparation

You can obtained the datasets from [GoogleDrive](https://drive.google.com/file/d/1N_SGBo7ZVCFoHEWsdAkfsHyv4A77clLU/view?usp=sharing) . (This may take some time, please wait patiently.) 

Then place the downloaded data under the folder `./dataset`.

### Train and evaluate model

- To see the model structure of DTWAiF.
- We provide the experiment scripts for DTWAiF under the folder `./scripts/multivariate_detection`. For example you can reproduce a experiment result as the following:

```
sh ./scripts/multivariate_detection/detect_label/MSL_script/DTWAiF.sh

sh ./scripts/multivariate_detection/detect_score/MSL_script/DTWAiF.sh
```



## Contact

If you have any questions or suggestions, feel free to contact:

- Jian Cao caoj20@fudan.edu.cn



Or describe it in Issues.