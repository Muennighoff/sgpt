#!/bin/bash
#SBATCH --job-name=run-array-a100    # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time 100:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@a100
#SBATCH --reservation=hug
#SBATCH --constraint=a100
#SBATCH --partition=gpu_p5
#SBATCH --qos=qos_gpu-gc             # up to 100h

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate muennighoffmtb

echo "START TIME: $(date)"

cd /gpfswork/rech/six/commun/code/tr13f-6B3-ml-t0/sgptce/

#"fever"
#"climate-fever"
#"nq"
#"hotpotqa"
DATASETS=(
"trec-covid"
"webis-touche2020"
"nfcorpus"
"scifact"
"fiqa"
"dbpedia-entity"
"quora"
"arguana"
"scidocs"
)

dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}

python sgptce.py \
	--batchsize 128 \
	--dataset $dataset \
	--modelpath /gpfsscratch/rech/six/commun/experiments/muennighoff/bloomckpt/6b3/bloom-6b3
