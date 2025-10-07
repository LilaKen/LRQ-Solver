import os
import numpy as np
import paddle
from ppcfd.data.base_datamodule import BaseDataModule


# Airfoil This benchmark is generated from simulations of 
# transonic flow over an airfoil, governed by 
# Euler’s equations (Li et al., 2022). The whole field 
# is discretized to unstructured meshes in the shape of 
# 221 × 51 as the input and the output is the corresponding 
# Mach number on these meshes. The dataset includes 
# 1000 training samples and 200 test samples that are based on 
# the initial NACA-0012 shape.


class PointDataset(paddle.io.Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.inputs_key = ["centroids"]
        self.targets_key = ["vel", "idx"]

    def __getitem__(self, index):
        return {
            "inputs": [self.x[index], self.x[index]],
            "targets": [self.y[index], index]
        }

    def __len__(self):
        return len(self.x)

    def data_to_dict(self, data):
        """
        将数据转换为字典格式
        Args:
            data: 输入数据，包含inputs和targets
        Returns:
            dict: 包含inputs、targets和其他信息的字典
        """
        inputs = {k: data["inputs"][i] for i, k in enumerate(self.inputs_key)}
        targets = {k: data["targets"][i] for i, k in enumerate(self.targets_key)}
        others = {}
        return inputs, targets, others


class PDE_Cylinder_DataModule(BaseDataModule):
    def __init__(
        self,
        data_dir,
        n_train_num,
        n_test_num,
        downsamplex=1,
        downsampley=1,
        train_sample_number=None
    ):
        BaseDataModule.__init__(self)
        n_train_num = max(n_train_num, 1000)
        n_test_num = max(n_train_num, 200)
        INPUT_X = data_dir + "/NACA_Cylinder_X.npy"
        INPUT_Y = data_dir + "/NACA_Cylinder_Y.npy"
        OUTPUT_Sigma = data_dir + "/NACA_Cylinder_Q.npy"
        r1 = downsamplex  # x downsample resolution
        r2 = downsampley  # y downsample resolution
        s1 = int((221 - 1) / r1 + 1)
        s2 = int((51 - 1) / r2 + 1)
        inputX = np.load(INPUT_X).astype("float32")
        inputY = np.load(INPUT_Y).astype("float32")
        input_tensor = np.stack([inputX, inputY], axis=-1)

        # Convert data into tensor
        output_data = np.load(OUTPUT_Sigma)
        output_tensor = output_data[:, 4].astype("float32")

        # Train
        x_train = input_tensor[:n_train_num, ::r1, ::r2][:, :s1, :s2]
        y_train = output_tensor[:n_train_num, ::r1, ::r2][:, :s1, :s2]
        x_train = x_train.reshape(n_train_num, -1, 2)
        y_train = y_train.reshape(n_train_num, -1)
        self.train_data = PointDataset(x_train, y_train)
        # Test
        x_test = input_tensor[n_train_num : n_train_num + n_test_num, ::r1, ::r2][:, :s1, :s2]
        y_test = output_tensor[n_train_num : n_train_num + n_test_num, ::r1, ::r2][:, :s1, :s2]
        x_test = x_test.reshape(n_test_num, -1, 2)
        y_test = y_test.reshape(n_test_num, -1)
        self.test_data = PointDataset(x_test, y_test)

