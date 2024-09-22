# STFlow: Spatio-Temporal Fusion of Events and Frames for Robust Optical Flow Estimation


![](./img/STF-framework.png)
This is the official codebase for the paper: STFlow: Spatio-Temporal Fusion of Events and Frames for Robust Optical Flow Estimation.

## Installation
The code has been tested with Python3.8 and PyTorch 1.11, some packages are needed:
```
pip install opencv-python-headless
pip install llvmlite==0.36.0
pip install numba==0.53.1
pip install gitpython==3.1.14
pip install hdf5plugin~=2.0
pip install tensorboardX 
pip install tables
pip install wandb
```

# Datasets
## DSEC-Flow
The DSEC-Flow dataset can be downloaded [here](https://dsec.ifi.uzh.ch/dsec-datasets/download/)


# Experiments

## Voxel Generation
Some preprocess is helpful to save training time. We use pre-generated event volumes saved in `.npz` files and flows in `.npy` files. Basically, we follow the data preprocess in [E-RAFT](https://github.com/uzh-rpg/E-RAFT).

The files used in the preprocessing process are placed in the `dataprocess` directory.

We put data in DSEC-Flow folder, and the structure should be like this:
```
|-dsec
    |-voxelandflow
        |-train
            |-thun_00_a
                |-000000.npz
                |-flow_000000.npy
                ...
        |-test
            |-interlaken_00_b
                |-xxxxxx.npz
                ...
    |-images
        |-train
            |-thun_00_a
                |-000000.png
                ...
        |-test
            |-interlaken_00_b
                |-xxxxxx.png
                ...
```

For train data, each `.npz` file contains two consecutive event streams named `events_prev` and `events_curr`, each `flow_xxxxxx.npy` file contains corresponding 16-bit optical flow.

For test data, the `.npz` file is indexed by test timestamp, which is useful for generating predictions for online benchmark.

The image data needs to be calibrated, it is recommended to use the data provided by [bflow](https://github.com/uzh-rpg/bflow).

## For train
```
bash scripts/train.sh
```

## For test
```
bash scripts/gen_upload.sh
```
