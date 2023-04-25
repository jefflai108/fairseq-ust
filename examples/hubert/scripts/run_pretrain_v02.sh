#!/bin/bash
#SBATCH -J InfoA.v2
#SBATCH -o /data/sls/scratch/clai24/lexicon/exp/slurm_dump/s2u_en.v02.pretrainedmHubert.6LuDecoder.200k.lr5e-4_%j.out   
#SBATCH -e /data/sls/scratch/clai24/lexicon/exp/slurm_dump/s2u_en.v02.pretrainedmHubert.6LuDecoder.200k.lr5e-4_%j.err   
#SBATCH --qos=regular 
#SBATCH --gres=gpu:4
#SBATCH --nodes=1 
#SBATCH --partition=a5,a6
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00 
#SBATCH --exclusive 
#SBATCH --mem=400G

#srun -p a6 --qos=regular --gres=gpu:4 --mem=420G --cpus-per-task=6 --pty bash

## Set the python environment you want to use for your code 
PYTHON_VIRTUAL_ENVIRONMENT=yung-sung-a5
CONDA_ROOT=/data/sls/scratch/clai24/anaconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh 
conda activate $PYTHON_VIRTUAL_ENVIRONMENT 

# your script/code to run below
expname=s2u_en.v02.pretrainedmHubert.6LuDecoder.200k.lr5e-4
expdir=/data/sls/scratch/clai24/lexicon/exp/hubert_pretraining/${expname}
mkdir -p $expdir
LAB_DIR=/data/sls/scratch/clai24/lexicon/exp/hubert_kmeans/s2u_en-es
TRAIN_SET=en-train_mined_t1.09
VAL_SET=en-valid_vp

############ DEBUG ##########
#expdir=/data/sls/scratch/clai24/lexicon/exp/hubert_pretraining/dummy
#LAB_DIR=/data/sls/scratch/clai24/lexicon/exp/hubert_kmeans/dummy
#TRAIN_SET=es_dummy
#VAL_SET=es_dummy
#############################

# set up environment variables for Torch DistributedDataParallel
#WORLD_SIZE_JOB=\$SLURM_NTASKS
#RANK_NODE=\$SLURM_NODEID
#PROC_PER_NODE=4
#MASTER_ADDR_JOB=\$SLURM_SUBMIT_HOST
#MASTER_PORT_JOB="12234"
#DDP_BACKEND=c10d

# reduce optimization.max_update=100k to 60k for faster model dev 
HYDRA_FULL_ERROR=1 python -u /data/sls/scratch/clai24/lexicon/fairseq/fairseq_cli/hydra_train.py \
    --config-dir /data/sls/scratch/clai24/lexicon/fairseq/examples/hubert/config/pretrain \
    --config-name hubert_base_info_align_v02 \
    hydra.run.dir=${expdir} \
    common.log_file=train.log \
    task.data=${LAB_DIR} \
    task.label_dir=${LAB_DIR} \
    task.labels=["km"] \
    dataset.train_subset=${TRAIN_SET} \
    dataset.valid_subset=${VAL_SET} \
    dataset.num_workers=8 \
    checkpoint.keep_best_checkpoints=5 \
    model.pretrained_hubert_ckpt=/data/sls/temp/clai24/pretrained-models/mHuBERT/mhubert_base_vp_en_es_fr_it3.pt \
    model.label_rate=50 \
    optimization.update_freq=[8] \
    optimization.max_update=200000 \
    lr_scheduler.warmup_updates=8000 \
    distributed_training.distributed_world_size=4 \
    distributed_training.nprocs_per_node=4 \
    distributed_training.distributed_port=0 \
    2>&1 | tee ${expdir}/train.log

    # modify:
    #   distributed_training.distributed_world_size=4
    #   optimization.update_freq=[8] 
    #   distributed_training.nprocs_per_node=4
    # to stimulate 32 GPUs 

