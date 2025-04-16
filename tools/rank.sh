#!/bin/bash
module purge
module load anaconda/2021.11
module load compilers/cuda/11.8
module load compilers/gcc/12.2.0
module load cudnn/8.4.0.27_cuda11.x
source activate swinTF

export PYTHONUNBUFFERED=1
export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

### nodes gpus rank master_addr job_id
# nodes
NODES=$1
# gpus
NPROC_PER_NODE=$2

# rank
NODE_RANK=$3

# master
MASTER_ADDR=$4
MASTER_PORT="29501"

#JOB ID
BATCH_JOB_ID=$5

# logs
echo "$NODE_RANK,$NODES,$NPROC_PER_NODE,$MASTER_ADDR,$BATCH_JOB_ID"
OUTPUT_LOG="/home/bingxing2/ailab/scxlab0061/Astro_SR/slurm_output/train_rank${NODE_RANK}_${BATCH_JOB_ID}.log"

CONFIG="configs/models/restormer.py"

torchrun \
     --nnodes="${NODES}" \
     --node_rank="${NODE_RANK}" \
     --nproc_per_node="${NPROC_PER_NODE}" \
     --master_addr="${MASTER_ADDR}" \
     --master_port="${MASTER_PORT}" \
     tools/trainval.py ${CONFIG} --launcher="slurm" --log_dir='log/restormer_l2' >> "${OUTPUT_LOG}" 2>&1
