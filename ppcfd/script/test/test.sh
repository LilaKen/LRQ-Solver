# Preparation : prepare env and data
if ! command -v nvidia-smi &> /dev/null
then
    echo "CUDA 未安装或 NVIDIA 驱动未正确安装。"
    exit 1
fi

DIRECTORY="data_drivaer"
if [ -d "$DIRECTORY" ]; then
    echo "文件 '$DIRECTORY' 存在于当前目录。"
else
    echo "文件 '$DIRECTORY' 不存在于当前目录。"
    if ls *"_data_checkpoints.tar" 1> /dev/null 2>&1; then
        tar -xvf 0318_data_checkpoints.tar
    else
        wget https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/docker_image/0318_data_checkpoints.tar
        tar -xvf 0318_data_checkpoints.tar
    fi
fi

# Step 2:测试模型

# 测试单卡
# export CUDA_VISIBLE_DEVICES=7
# python main_v2.py --config-path ./configs --config-name transolver_dp.yaml enable_dp=false num_epochs=10 batch_size=1 train_sample_number=1000 n_train_num=100 n_test_num=10

# # dp 4卡 报错
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# python -m paddle.distributed.launch --gpus=3,4,5,6 main_v2.py --config-path ./configs --config-name transolver_dp.yaml enable_dp=true num_epochs=10 batch_size=8 train_sample_number=1000 n_train_num=100 n_test_num=10

# dp 只支持双卡
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# python -m paddle.distributed.launch --gpus=5,6 main_v2.py --config-path ./configs --config-name transolver_dp.yaml enable_dp=true num_epochs=10 batch_size=4 train_sample_number=1000 n_train_num=100 n_test_num=10

# python main_v2.py --config-path ./configs --config-name transolver.yaml checkpoint=./outputs_gino/2025-05-07-03-47-51/Transolver_39
# python main_v2.py --config-path ./configs --config-name transolver.yaml
# export CUDA_VISIBLE_DEVICES=0

# Test : transolver + pressure wss
# MRE Error Mean: [Cp], 2.51%, [Cf], 0.00%, [Cd], 0.00%, [Cl], 0.00%  
# Relative L2 Error Mean : [P], 0.1583, [WSS], 0.0000, [VEL], 0.0000
# R2 Score: [Cd], 1.00
# python main_v2.py \
#     --config-name transolver.yaml  \
#     mode=test \
#     model.out_dim=1 \
#     data_module.n_test_num=10000 \
#     data_module.data_dir="/workspace/gino_data/6_DrivAir_p/" \
#     out_keys=[pressure] \
#     checkpoint="./checkpoints/Transolver_299"

python main_v2.py \
    --config-name transolver.yaml  \
    mode=test \
    model.out_dim=1 \
    data_module.n_test_num=10 \
    data_module.data_dir="/home/wangguan12/car/data/drivaer/test/data_drivaer" \
    out_keys=[pressure] \
    checkpoint="/home/wangguan12/car/checkpoint/drivaer/Transolver_299"

# Test : paddledict + wss
python main_v2.py \
    --config-name transolver_starccm+.yaml \
    mode=test \
    model.out_dim=3 \
    data_module.data_dir="./data_fake_starccm+/" \
    out_keys=["wss"] \
    checkpoint=./checkpoints/Transolver_starccm+_299_wss

# Train : paddledict + pressure wss
python main_v2.py \
    --config-name transolver_starccm+.yaml \
    mode=train \
    num_epochs=10 \
    data_module.data_dir="./data_fake_mp/"

# Test : transolver + velocity
python main_v2.py \
    --config-name transolver.yaml \
    data_module._target_=ppcfd.data.PointCloud_Vel_DataModule \
    data_module.data_dir="./data_drivaer_ml/" \
    mode=test \
    out_keys=["vel"] \
    out_channels=[1] \
    model.out_dim=1 \
    checkpoint="./checkpoints/Transolver_299_velocity_slice"

# Test : gino + presssure wss
python main_v2.py \
    --config-name gino.yaml \
    mode=test \
    data_module.n_test_num=10 \
    data_module.data_path="./data_drivaer/" \
    model.flash_neighbours=v2 \
    checkpoint="./checkpoints/p_wss_1_3_DrivAer_299_0527_new"

# Test : regpointnet + cd
# python main_v2.py \
#     --config-name regpointnet_test.yaml  \
#     mode=test \
#     data_module.num_points=100000 \
#     checkpoint="./checkpoints/pointnet_cd_200"


# Toolbox : case to paddledict
python -m ppcfd.script.starccm_plus.read_all

# Toolbox : frontal area caculation
python -m ppcfd.script.starccm_plus.pyfrontal ./data_fake_starccm+/case/test.case
python -m ppcfd.script.starccm_plus.pyfrontal ./data_fake_starccm+/test.stl


# Step 4:模型并行切分
## 获取可用 GPU 的数量
gpu_count=$(nvidia-smi --list-gpus | grep -c "GPU")

## 判断是否有多于四张的 GPU
if [ "$gpu_count" -le 4 ]; then
    echo "只检测到 $gpu_count / 4 张 GPU，无法进行多卡测试。"
else
    echo "检测到 $gpu_count / 8 张 GPU，可以继续进行多卡测试。"
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

    # 模型(enable_mp)并行2卡切分
    python -m paddle.distributed.launch --gpus=1,5 main_v2.py \
        --config-name transolver.yaml \
        mode=test \
        checkpoint="./checkpoints/Transolver_299" \
        enable_mp=true \
        data_module.data_dir="./data_drivaer/" \
        out_keys=[pressure] \
        out_channels=[1] \
        model.out_dim=1

    # 模型(enable_mp)并行4卡切分
    python -m paddle.distributed.launch --gpus=1,2,3,4 main_v2.py \
        --config-name transolver.yaml \
        mode=test \
        checkpoint="./checkpoints/Transolver_299" \
        enable_mp=true \
        data_module.data_dir="./data_drivaer/" \
        out_keys=[pressure] \
        out_channels=[1] \
        model.out_dim=1

    # 800W points
    python -m paddle.distributed.launch --gpus=0,1,2,3 main_v2.py \
        --config-name transolver.yaml \
        mode=test \
        enable_mp=true \
        data_module._target_=ppcfd.data.StarCCMDataModule \
        data_module.data_dir=/home/wangguan12/car/data/test/data_fake_mp \
        checkpoint=/home/wangguan12/car/checkpoint/drivaer/Transolver_299 \
        out_keys=[pressure] \
        out_channels=[1] \
        model.out_dim=1
    
    python main_v2.py \
        --config-name transolver.yaml \
        mode=test \
        enable_mp=true \
        data_module._target_=ppcfd.data.StarCCMDataModule \
        data_module.data_dir=/home/wangguan12/car/data/test/data_fake_mp \
        checkpoint=/home/wangguan12/car/checkpoint/drivaer/Transolver_299 \
        out_keys=[pressure] \
        out_channels=[1] \
        model.out_dim=1

    python -m paddle.distributed.launch --gpus=0,1,2,3 main_v2.py \
        --config-name gino.yaml \
        mode=test \
        enable_mp=true \
        data_module.n_test_num=10 \
        model.flash_neighbours=False \
        data_module.data_path="/workspace/DNNFluid_Car/DNNFluid-Car/data_drivaer/" \
        checkpoint="/workspace/DNNFluid_Car/DNNFluid-Car/checkpoints/p_wss_1_3_DrivAer_299_0527_new"

    # 流水(enable_pp)并行切分
    python -m paddle.distributed.launch --gpus=0,1,2,3 main_v2.py --config-path ./ppcfd/script/test/ --config-name transolver.yaml enable_pp=true
    python -m paddle.distributed.launch --gpus=0,1,2,3 main_v2.py --config-path ./ppcfd/script/test/ --config-name transolver_starccm+.yaml enable_pp=true data_path=./data_fake_mp/

    # 数据(enable_dp)并行切分
    python -m paddle.distributed.launch --gpus=4,5,6,7 main_v2.py --config-path ./ppcfd/script/test/ --config-name transolver.yaml enable_dp=true
fi

# test in 01
# python ppcfd/web/predict_case.py --config-path ../../configs/ --config-name=test.yaml output_filename=0222_test.vtk

# python -m streamlit run ./ppcfd/web/viewer.py  --server.maxUploadSize 2000

# web端可视化的工具，进行真值推理
python ppcfd/web/predict.py \
    --config-name=transolver.yaml \
    mode=test \
    data_module._target_=ppcfd.data.PointCloudDataModule \
    checkpoint=["./checkpoints/Transolver_299","./checkpoints/Transolver_starccm+_299_wss"]

# ppcfd/script/starccm+/paraview/bin/pvpython ./ppcfd/script/starccm+/read_by_paraview.py
# python /workspace/docker_image/BosClient.py PaddleCFD_0616.tar dataset/PaddleScience/DNNFluid-Car/PaddleCFD