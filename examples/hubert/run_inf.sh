#!/bin/bash 

#PROC_FLEURS_DIR=/data/sls/temp/clai24/data/speech_matrix/eval_data/fairseq_processed_fleurs
#SAVE_ROOT=/data/sls/temp/clai24/data/speech_matrix/speech_to_unit
#HUBERT_CKPT=/data/sls/scratch/clai24/lexicon/exp/hubert_pretraining/dummy/checkpoints/checkpoint_best.pt
#
#export PYTHONPATH="${PYTHONPATH}:${PWD}/../../:${PWD}/simple_kmeans"
#
##CUDA_VISIBLE_DEVICES=0 
#python3 hubert_info_align_decode.py --proc-fleurs-dir ${PROC_FLEURS_DIR} --save-root ${SAVE_ROOT} --hubert-ckpt ${HUBERT_CKPT}



expname=s2u_en.v00.pretrained.lmL12C1k.60k.lr5e-4
expdir=/data/sls/scratch/clai24/lexicon/exp/hubert_pretraining/${expname}
mkdir -p $expdir
LAB_DIR=/data/sls/scratch/clai24/lexicon/exp/hubert_kmeans/s2u_en-es
TRAIN_SET=en_train
VAL_SET=en_val

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
    --config-dir /data/sls/scratch/clai24/lexicon/fairseq/examples/hubert/config/finetune \
    --config-name hubert_base_info_align_v00 \
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
    model.w2v_path=/data/sls/scratch/clai24/lexicon/exp/hubert_pretraining/dummy/checkpoints/checkpoint_best.p \
    model.label_rate=50 \
    optimization.update_freq=[8] \
    optimization.max_update=60000 \
    lr_scheduler.warmup_updates=8000 \
    distributed_training.distributed_world_size=4 \
    distributed_training.nprocs_per_node=4 \
    distributed_training.distributed_port=0 \
    2>&1 | tee ${expdir}/train.log


