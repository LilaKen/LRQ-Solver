# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppcfd.networks.net_utils import PositionalEmbedding


class MLP_with_boundary_condition(nn.Layer):
    def __init__(self, input_size=7, out_dim=10, embed_dim=3):
        super(MLP_with_boundary_condition, self).__init__()
        self.pos_embed = PositionalEmbedding(embed_dim)
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_dim)

    def embed_boundary_condition_wind(self, x):
        centroids = x["centroids"]
        speed = x["speed"]
        direction = x["direction"]
        n = centroids.shape[1]
        speed = self.pos_embed(speed)
        direction = self.pos_embed(direction)
        speed = paddle.tile(speed, [n, 1]).astype("float32").reshape([1, -1, 2])
        direction = paddle.tile(direction, [n, 1]).astype("float32").reshape([1, -1, 2])
        x = paddle.concat([centroids, speed, direction], axis=2)
        return x

    def embed_boundary_condition(self, x):
        centroids = x["centroids"]
        x = paddle.concat([centroids, speed, direction], axis=2)
        return x

    def forward(self, x):
        x = self.embed_boundary_condition(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP(nn.Layer):
    def __init__(self, input_size=8, out_dim=10, reshape=False):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_dim)
        self.reshape = reshape
        if reshape:
            self.reshape_output_layer = paddle.nn.Linear(out_dim, 10)
            self.ln = paddle.nn.LayerNorm(normalized_shape=[10, 1])

    def data_dict_to_input(self, inputs):
        if isinstance(inputs, list):
            # Handle list input from dataloader
            features = paddle.concat(x=inputs, axis=-1)
        else:
            # Handle dict input for backward compatibility
            x_centroid = inputs["centroids"]
            x_local_centroid = inputs["local_centroid"]
            features = paddle.concat(x=[x_centroid, x_local_centroid], axis=-1)
        return features

    def reshape_output(self, x):
        x = self.reshape_output_layer(x)
        x = paddle.sum(x, axis=1, keepdim=True).transpose([0, 2, 1])
        x = self.ln(x)
        return x

    def forward(self, data):
        x = self.data_dict_to_input(data)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.reshape:
            x = self.reshape_output(x)
        return x
