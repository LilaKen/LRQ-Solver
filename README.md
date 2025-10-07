# News:

[飞桨采用NVIDIA Modulus打造汽车风阻预测模型DNNFluid-Car](https://mp.weixin.qq.com/s/pxmOpfwe0DXCon4uGG93uQ)

## 常见报错汇总
https://github.com/wangguan1995/DNNFluid-Car/issues/71

# Step 1 : 快速安装

## 显卡驱动要求cuda 12.3

方法一 Linux离线docker安装

linux端文件夹没有权限（报错：Permission Denied）, 需要chmod 777 -R 文件名
```shell
wget https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/docker_image/dnnfluid-car_v1.0.tar
docker load -i dnnfluid-car_v1.0.tar
```

方法二 Linux联网安装
```shell
pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu123/
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/open3d-0.18.0%2Bd31268ae-cp310-cp310-manylinux_2_31_x86_64.whl
apt-get update
apt-get install xvfb
```

# Step 2 : 下载代码、测试集，下载Checkpoints，以及验证安装
```
# wget https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/docker_image/data_checkpoint_0519.tar

# 执行自测（每次提交代码必做）
./ppcfd/script/test/test.sh
```

# Step 3 : 调整和训练模型

模型目录为src/networks
| 模型 | DrivAer L2 error | Ahmed L2 error | 已接入测试 | 模型论文 |合入PR|
|:---------------------:|:--------:|:-:|:------------:|:------------:|:------------:|
|  GINO                |     0.156  ||      ✅       ||
|  Transolver          |     0.14   ||      ✅       ||
|  UNet3D              |     0.23   ||      ✅       ||
|  FigConvnet          |     0.16   ||      🚧       |[PR 55](https://github.com/wangguan1995/DNNFluid-Car/pull/55)|
|  LNO                 |     🚧     ||      🚧       ||
|  XAeronet            |     🚧     ||      🚧       ||
|  Domino              |     🚧     ||      🚧       ||

# Step 4 : 兼容工业数据集

数据集代码目录为src/data

数据集下载脚本目录为src/script/download

| 工业数据集 | 开源 | dataset可用 | 数据下载地址 |
|:---------------------:|:--------:|:------------:|:------------:|
|  建筑风场数据                |    🚧     |      🚧       ||
|  3D飞行器数据集              |    ✅     |      🚧       ||
|  Arteon_2021               |    ✅     |      🚧       ||
|  DrivAerML                 |    ✅     |      ✅       |[stl_part1](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAerML/part1_1-50.tar) 、[cd](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAerML/drivaerml_csv.tar)|
|  DrivAerNet                |    ✅     |      ✅       ||
|  DrivAerNet++              |    ✅     |      ✅       |[points](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/DrivAer%2B%2B_Points.tar)、[cd](https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B/DrivAerNetPlusPlus_Drag_8k.csv)|
|  Ahmed                     |    ✅     |      ✅       ||
|  ShapeNet-Car(未简化)       |    ✅     |      ✅       ||
|  ShapeNet-Car(简化)         |    ✅     |      ✅       |[飞桨云](https://dataset.bj.bcebos.com/PaddleScience/2024%20Transolver/Car-Design-ShapeNetCar.tar)|

# Step 5 : 可视化
执行命令
```shell
python -m streamlit run ./ppcfd/web/viewer.py
```
![image](https://github.com/user-attachments/assets/d5c042c6-3925-4508-8836-24f4efed4cb3)

