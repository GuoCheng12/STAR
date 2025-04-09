#!/bin/bash
#SBATCH --qos gpugpu
#SBATCH -A ai4astro
#SBATCH -p vip_gpu_ailab

### 用户指定参数 ###
TOTAL_GPUS=15          # 总共使用的GPU数量
GPUS_PER_NODE=4        # 每个节点可用的GPU数量

### 自动计算需要的节点数 ###
NODES=$(( (TOTAL_GPUS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))  # 向上取整

### SBATCH参数动态设置 ###
#SBATCH -N ${NODES}
#SBATCH --gres=gpu:${GPUS_PER_NODE}

### 脚本名称
RANK_SCRIPT="tools/rank.sh"

### Job Path
JOB_PATH=`pwd`

### Job ID
JOB_ID="${SLURM_JOB_ID}"

### 获取节点主机名
for i in `scontrol show hostnames`
do
  let k=k+1
  host[$k]=$i
  rank[$k]=$(($k-1))
  echo ${host[$k]}
done

### 设置主节点
MASTER_ADDR=${host[1]}

### 启动作业
for((i=1;i<=${NODES};i++));
do
  node_host=${host[$i]}
  node_rank=${rank[$i]}
  # 计算每个节点的nproc_per_node
  if [ $i -lt ${NODES} ]; then
    nproc_per_node=${GPUS_PER_NODE}
  else
    # 最后一个节点使用的GPU数量
    remaining_gpus=$(( TOTAL_GPUS - (NODES - 1) * GPUS_PER_NODE ))
    nproc_per_node=${remaining_gpus}
  fi
  echo "nodes:${NODES}, host:${node_host}, node_rank:${node_rank}, master_addr:${MASTER_ADDR}, nproc_per_node:${nproc_per_node}"
  if [ $i -eq 1 ]; then
    # 主节点直接运行
    bash ${RANK_SCRIPT} ${NODES} ${nproc_per_node} 0 ${MASTER_ADDR} ${JOB_ID} &
  else
    # 其他节点使用srun运行
    srun -N 1 -w $node_host bash ${RANK_SCRIPT} ${NODES} ${nproc_per_node} $node_rank ${MASTER_ADDR} ${JOB_ID} &
  fi
done
wait