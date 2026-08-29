#!/bin/bash

# Example usage:
# ./run_train.sh <nnodes> <nproc_per_node> <config_path> <output_dir>

if [ $# -ne 4 ]; then
  echo "Usage: $0 <nnodes> <nproc_per_node> <config_path> <output_dir>"
  exit 1
fi

NNODES=$1
NPROC_PER_NODE=$2
CONFIG_PATH=$3
OUTPUT_DIR=$4

# Launch training with torch.distributed
nohup venv/bin/python -m torch.distributed.run \
  --nnodes "$NNODES" \
  --nproc_per_node "$NPROC_PER_NODE" \
  --node_rank 0 \
  main.py \
  --config "$CONFIG_PATH" \
  --output_dir "$OUTPUT_DIR" \
  > log.log 2>&1 &
