#!/bin/bash

########### DEBUG ##########
expdir=/data/sls/scratch/clai24/lexicon/exp/hubert_pretraining/dummy
LAB_DIR=/data/sls/scratch/clai24/lexicon/exp/hubert_kmeans/dummy
TRAIN_SET=es_dummy
VAL_SET=es_dummy
mkdir -p $expdir
############################

# set up environment variables for Torch DistributedDataParallel
#WORLD_SIZE_JOB=\$SLURM_NTASKS
#RANK_NODE=\$SLURM_NODEID
#PROC_PER_NODE=4
#MASTER_ADDR_JOB=\$SLURM_SUBMIT_HOST
#MASTER_PORT_JOB="12234"
#DDP_BACKEND=c10d

HYDRA_FULL_ERROR=1 python -u /data/sls/scratch/clai24/lexicon/fairseq/fairseq_cli/hydra_train.py \
    --config-dir /data/sls/scratch/clai24/lexicon/fairseq/examples/hubert/config/pretrain \
    --config-name hubert_base_info_align_v01 \
    hydra.run.dir=${expdir} \
    common.log_file=train.log \
    task.data=${LAB_DIR} \
    task.label_dir=${LAB_DIR} \
    task.labels=["km"] \
    dataset.train_subset=${TRAIN_SET} \
    dataset.valid_subset=${VAL_SET} \
    dataset.num_workers=8 \
    checkpoint.keep_best_checkpoints=5 \
    model.pretrained_hubert_ckpt="" \
    model.label_rate=50 \
    optimization.update_freq=[1] \
    optimization.max_update=100000 \
    lr_scheduler.warmup_updates=8000 \
    distributed_training.distributed_world_size=1 \
    distributed_training.nprocs_per_node=1 \
    distributed_training.distributed_port=0 \
    2>&1 | tee ${expdir}/train.log

    # modify:
    #   distributed_training.distributed_world_size=4
    #   optimization.update_freq=[8] 
    #   distributed_training.nprocs_per_node=4
    # to stimulate 32 GPUs 

