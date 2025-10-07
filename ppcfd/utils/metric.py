# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

import paddle


class R2Score:
    @paddle.no_grad()
    def __call__(self, output_dict, label_dict) -> Dict[str, "paddle.Tensor"]:
        r2score_dict = {}

        for key in label_dict:
            output = output_dict[key]
            target = label_dict[key]
            if output.shape != target.shape:
                raise ValueError(
                    f"Output and target shapes do not match for key '{key}'. Output shape: {output.shape}, Target shape: {target.shape}"
                )

            output = output.flatten()
            target = target.flatten()
            target_mean = target.mean(axis=-1, keepdim=True)
            ss_tot = paddle.sum(x=(target - target_mean) ** 2)
            ss_res = paddle.sum(x=(target - output) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            # import pdb; pdb.set_trace()
            r2score_dict[key] = r2.unsqueeze(axis=0).item()
        return r2score_dict
