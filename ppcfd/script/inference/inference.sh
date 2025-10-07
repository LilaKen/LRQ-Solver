python infer.py \
    --config-name transolver.yaml \
    mode=export \
    model.out_dim=1 \
    checkpoint=/home/wangguan12/car/checkpoint/drivaer/Transolver_299 \
    data_module.data_dir=/home/wangguan12/car/data/test/data_drivaer \
    out_channels=[1] \
    out_keys=["pressure"]

python infer.py \
    --config-name transolver.yaml \
    mode=inference \
    model.out_dim=1 \
    checkpoint=output/2025-06-23-02-58-47/Transolver_299 \
    data_module.data_dir=/home/wangguan12/car/data/test/data_drivaer \
    out_channels=[1] \
    out_keys=["pressure"]


python infer.py \
    --config-name transolver.yaml \
    mode=export \
    model.out_dim=1 \
    data_module._target_=ppcfd.data.StarCCMDataModule \
    data_module.data_dir=/home/wangguan12/car/data/test/data_fake_mp \
    out_keys=["pressure"] \
    out_channels=[1] \
    checkpoint=/home/wangguan12/car/checkpoint/drivaer/Transolver_299


python infer.py \
    --config-name transolver.yaml \
    mode=inference \
    model.out_dim=1 \
    data_module._target_=ppcfd.data.StarCCMDataModule \
    data_module.data_dir=/home/wangguan12/car/data/test/data_fake_mp \
    out_keys=["pressure"] \
    out_channels=[1] \
    checkpoint=./output/2025-06-23-05-56-25/Transolver_299


python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 infer.py \
    --config-name transolver.yaml \
    mode=export \
    model.out_dim=1 \
    enable_mp=true \
    data_module._target_=ppcfd.data.StarCCMDataModule \
    data_module.data_dir=/home/wangguan12/car/data/test/data_fake_mp \
    out_keys=["pressure"] \
    out_channels=[1] \
    checkpoint=/home/wangguan12/car/checkpoint/drivaer/Transolver_299

python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 infer.py \
    --config-name transolver.yaml \
        mode=inference \
        model.out_dim=1 \
        enable_mp=true \
        data_module._target_=ppcfd.data.StarCCMDataModule \
        data_module.data_dir=/home/wangguan12/car/data/test/data_fake_mp \
        out_keys=["pressure"] \
        out_channels=[1] \
        checkpoint=./output/2025-06-23-07-30-34/Transolver_299

