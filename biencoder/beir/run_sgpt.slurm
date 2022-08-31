#!/bin/bash
#SBATCH --job-name=run-array-a100    # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=20:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=six@a100
#SBATCH --reservation=hug
#SBATCH --constraint=a100
#SBATCH --partition=gpu_p5

set -x -e

source $six_ALL_CCFRWORK/start-prod
conda activate muennighoffs

# Modelname to run
modelname=sgpt-bloom-7b1-msmarco
device=cuda:0

declare -a arr=("msmarco" "nfcorpus" "bioasq" "nq" "hotpotqa" "fiqa"
                "signal1m" "trec-news" "arguana" "webis-touche2020" "quora" "dbpedia-entity"
                "scidocs" "fever" "climate-fever" "scifact" "robust04" "cqadupstack/android" "cqadupstack/english" "cqadupstack/gaming"
               "cqadupstack/gis" "cqadupstack/mathematica" "cqadupstack/physics" "cqadupstack/programmers"
               "cqadupstack/stats" "cqadupstack/wordpress" "cqadupstack/webmasters" "cqadupstack/unix"
               "cqadupstack/tex" "trec-covid")


python beir_dense_retriever.py --device $device --specb --modelname $modelname --usest --dataset ${arr[$SLURM_ARRAY_TASK_ID]} --batchsize 16 --method weightedmean
