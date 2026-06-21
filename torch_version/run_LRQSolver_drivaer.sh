#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Train
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# uv run torchrun --nproc_per_node=4 "${SCRIPT_DIR}/main_drivaer.py" \
#    --config-name lrqsolver_drivaerpp.yaml \
#    mode=train \
#    enable_dp=true \
#    data_module.num_points=100000 \
#    data_module.data_dir=./data/drivaerpp/p/new_sample \
#    data_module.csv_file=./data/drivaerpp/DrivAerNetPlusPlus_Cd_8k_Updated.csv \
#    data_module.para_file=./data/drivaerpp/DrivAerNet_ParametricData.csv \
#    data_module.area_file=./data/drivaerpp/DrivAerNetPlusPlus_CarDesign_Areas.csv \
#    data_module.subset_dir=./data/drivaerpp/subset_dir \
#    data_module.n_test_num=1

# Test
export CUDA_VISIBLE_DEVICES=0
uv run python "${SCRIPT_DIR}/main_drivaer.py" \
   --config-name lrqsolver_drivaerpp.yaml \
   mode=test \
   test_batch_size=1 \
   data_module.num_points=100000 \
   data_module.data_dir=./data/drivaerpp/p/new_sample \
   data_module.para_file=./data/drivaerpp/DrivAerNet_ParametricData.csv \
   data_module.subset_dir=./data/drivaerpp/subset_dir \
   data_module.area_file=./data/drivaerpp/DrivAerNetPlusPlus_CarDesign_Areas.csv \
   data_module.csv_file=./data/drivaerpp/DrivAerNetPlusPlus_Cd_8k_Updated.csv \
   data_module.n_train_num=1 \
   data_module.n_test_num=1154 \
   data_module.n_val_num=1 \
   checkpoint=./output/lrqsolver_drivaerpp/best_LRQSolver.pt
