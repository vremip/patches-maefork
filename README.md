## This repository contains a fork of MAE

Original link: https://github.com/facebookresearch/mae

`pip install torch==1.7.1 torchvision==0.8.2 timm==0.3.2`

# MNIST

``python main_pretrain.py --num-workers 1 --batch-size 4 --device cpu --dataset mnist --data-path data --model mae_vit_base --patch-size 4``

# IMAGENET

``python main_pretrain.py --num-workers 8 --batch-size 128 --device cuda --data-path /datadrive/retachet/imagenet2012 --model mae_vit_tiny --patch-size 16``

# IMAGENET DISTRIBUTED

By default, runs on one node, uses all gpus present.

``MASTER_ADDR="localhost" MASTER_PORT="29500" python distributed_pretrain.py --num-workers 8 --batch-size 128 --device cuda --data-path /datadrive/retachet/imagenet2012 --model mae_vit_tiny --patch-size 16``

Possible add NUMBA_THREADING_LAYER='omp' if Numba complains about TBB.

# IMAGENET FFCV

Install instructions:

`conda install cupy pkg-config compilers libjpeg-turbo cudatoolkit=11.3 numba tbb -c conda-forge`

<!-- `conda install cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba tbb -c pytorch -c conda-forge` -->
`pip install opencv-python ffcv`

Expects `train_500_0.50_90.ffcv` to be in your datapath. See LINK on instructions to generate that file.

``NUMBA_THREADING_LAYER='omp' MASTER_ADDR="localhost" MASTER_PORT="29500" python distributed_pretrain.py --num-workers 8 --batch-size 128 --device cuda --data-path /datadrive/retachet/imagenet2012 --model mae_vit_tiny --patch-size 16 --ffcv-loader``
