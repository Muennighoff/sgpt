#!/bin/bash
#SBATCH --job-name=muennighoffs
#SBATCH --partition=gpu_p5
#SBATCH --constraint=a100
#SBATCH --reservation=hug
#SBATCH --qos=qos_gpu-gc             # up to 100h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 100:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@a100

set -x -e

source $six_ALL_CCFRWORK/start-tr13f-6B3-ml-t0
conda activate muennighoffs
echo "START TIME: $(date)"

cd /gpfsscratch/rech/six/commun/experiments/muennighoff/sgpt/biencoder/nli_msmarco/sentence-transformers

WANDB_MODE="dryrun" WANDB_BASE_URL=https://api.wandb.ai WANDB_API_KEY=YOUR_KEY WANDB_ENTITY=muennighoff WANDB_PROJECT="sgpt" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file /gpfsscratch/rech/six/commun/experiments/muennighoff/sgpt/accelerate_config_fp32 examples/training/ms_marco/train_bi-encoder_mnrl.py --model_name /gpfsscratch/rech/six/commun/commun/experiments/muennighoff/bloomckpt/6b3/bloom-7b1 --train_batch_size 32 --eval_batch_size 16 --freezenonbias --specb --lr 4e-4 --wandb --wandbwatchlog gradients --pooling weightedmean --gradcache --chunksize 8

echo "DONE"
