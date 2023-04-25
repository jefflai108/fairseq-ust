#!/bin/bash 
#SBATCH -J info-parse
#SBATCH -o /data/sls/scratch/clai24/lexicon/exp/slurm_dump/info-parse_%j.out   
#SBATCH -e /data/sls/scratch/clai24/lexicon/exp/slurm_dump/info-parse_%j.err   
#SBATCH --qos=regular 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 
#SBATCH --partition=a5,a6,2080
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00 
#SBATCH --mem=50G

## Set the python environment you want to use for your code 
PYTHON_VIRTUAL_ENVIRONMENT=yung-sung-a5
CONDA_ROOT=/data/sls/scratch/clai24/anaconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh 
conda activate $PYTHON_VIRTUAL_ENVIRONMENT 

python decode_scripts/hubert_info_align_phn_decode.py \
    --min-pmi 0 \
    --hubert-info-align-ckpt /data/sls/scratch/clai24/lexicon/exp/hubert_pretraining/s2u_en.v03.pretrainedmHubert.6LuDecoder.100k.lr5e-4/checkpoints/checkpoint_best.pt \
    --split en-valid_vp \
    --model-version v03 \
    --parse-alg top_down \
    --njobs 10 \
    --job-id $1

#python decode_scripts/hubert_info_align_phn_decode.py \
#    --min-pmi $1 \
#    --hubert-info-align-ckpt /data/sls/scratch/clai24/lexicon/exp/hubert_pretraining/s2u_en.v02.pretrainedmHubert.6LuDecoder.200k.lr5e-4/checkpoints/checkpoint_best.pt \
#    --split en-valid_vp-subset100 \
#    --model-version v02 \
#    --parse-alg $2

