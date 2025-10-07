# Train
export CUDA_VISIBLE_DEVICES=0,1,2
python -m paddle.distributed.launch --gpus=0,1,2 main_drivaer.py \
   --config-name lrqsolver_drivaerpp.yaml \
   mode=train \
   data_module.num_points=8192 \
   data_module.data_dir=./data/drivaerpp/sample \
   data_module.csv_file=./data/drivaerpp/DrivAerNetPlusPlus_Cd_8k_Updated.csv \
   data_module.para_file=./data/drivaerpp/DrivAerNet_ParametricData.csv \
   data_module.area_file=./data/drivaerpp/DrivAerNetPlusPlus_CarDesign_Areas.csv \
   data_module.subset_dir=./data/drivaerpp/subset_dir \
   data_module.n_test_num=1

# Test
# Cd summary over 1154 cases | MSE=5.8725e-05  MAE=5.9726e-03  MaxAE=3.0549e-02  MRE=2.28%
python main_drivaer.py \
   --config-name lrqsolver_drivaerpp.yaml \
   mode=test \
   test_batch_size=128 \
   data_module.num_points=8192 \
   data_module.data_dir=./data/drivaerpp/sample \
   data_module.para_file=./data/drivaerpp/DrivAerNet_ParametricData.csv \
   data_module.subset_dir=./data/drivaerpp/subset_dir \
   data_module.area_file=./data/drivaerpp/DrivAerNetPlusPlus_CarDesign_Areas.csv \
   data_module.csv_file=./data/drivaerpp/DrivAerNetPlusPlus_Cd_8k_Updated.csv \
   data_module.n_train_num=1 \
   data_module.n_test_num=1154 \
   data_module.n_val_num=1 \
   checkpoint=./checkpoint/LRQSolver_DrivAerpp_epoch_56.pdparams
