Learning to rank by diffusion model.

## Usage

1.To run this project, you should install conda environment from 'denoiseRank_env.yml' ( run : `conda env create -f environment.yml`)

2.Datasets should put into folder MSLR-WEB30K, YAHOO and ISTELLLA to train or test DenoiseRank on the datasets.

3.Start training by torchrun command : `CUDA_VISIBLE_DEVICES=xx NCCL_DEBUG=INFO torchrun --nproc_per_node=xx --master_port=xxxx main.py`, you can set `CUDA_VISIBLE_DEVICES=xx` and `--nproc_per_node=xx` to train on GPU parallelly. 

