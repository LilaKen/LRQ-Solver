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


import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.initializer as init
from paddle.incubate.nn.functional import fused_rotary_position_embedding


class LRQSolver(nn.Layer):
    def __init__(
        self,
        in_dim=3,
        out_dim=1,
        hidden_channel=256,
        n_heads=2,
        head_dim=16,
        seqlen=16,
        pseudo_integration=False,
        out_num=10,
        q_input_dim=23,
        q_num_query=10,
        q_num_heads=2,
        q_head_dim=8,
        q_ff_dim=16,
        remove_channel_dim=False,
        **kwargs
    ):
        super(LRQSolver, self).__init__()
        self.out_dim = out_dim
        self.preprocess = MLP(
            in_dim, hidden_channel * 2, hidden_channel, n_layers=2, res=False
        )

        self.qformer = QFormer(
            query_dim=hidden_channel // 4,
            input_dim=q_input_dim,
            num_query=q_num_query,
            num_heads=q_num_heads,
            head_dim=q_head_dim,
            ff_dim=q_ff_dim,
        )

        self.sa1 = SA_Layer(hidden_channel, n_heads, head_dim, seqlen)
        self.sa2 = SA_Layer(hidden_channel, n_heads, head_dim, seqlen)
        self.sa3 = SA_Layer(hidden_channel, n_heads, head_dim, seqlen)
        self.sa4 = SA_Layer(hidden_channel, n_heads, head_dim, seqlen)
        self.post_process = paddle.nn.Linear(
            in_features=hidden_channel, out_features=hidden_channel
        )

        self.pseudo_integration = pseudo_integration
        self.remove_channel_dim = remove_channel_dim

        if pseudo_integration:
            self.ln1 = paddle.nn.LayerNorm(normalized_shape=hidden_channel)
            self.ln2 = paddle.nn.LayerNorm(normalized_shape=hidden_channel)
            self.ln3 = paddle.nn.LayerNorm(normalized_shape=hidden_channel // 2)
            self.ln4 = paddle.nn.LayerNorm(normalized_shape=hidden_channel // 4)
            self.pooling1 = paddle.nn.MaxPool1D(kernel_size=64)
            self.pooling2 = paddle.nn.MaxPool1D(kernel_size=64)
            self.pooling3 = paddle.nn.MaxPool1D(kernel_size=64)
            self.mlp1 = paddle.nn.Linear(hidden_channel, hidden_channel // 2)
            self.mlp2 = paddle.nn.Linear(hidden_channel // 2, hidden_channel // 4)
            self.mlp3 = paddle.nn.Linear(hidden_channel // 4, out_dim)

    def reshape_output(self, x):
        x = self.ln1(x)
        x = x.transpose(perm=[0, 2, 1])
        x = self.pooling1(x)
        x = x.transpose([0, 2, 1])
        x = self.mlp1(self.ln2(x))
        x = x.transpose([0, 2, 1])  # [B, 128, 1562]
        x = self.pooling2(x)
        x = x.transpose([0, 2, 1])
        x = self.mlp2(self.ln3(x))
        x = x.transpose([0, 2, 1])  # [B, 64, 24]
        x = self.pooling3(x)
        x = x.transpose([0, 2, 1])  # [B, out_dim, 64]
        x = self.mlp3(self.ln4(x))
        x = x.squeeze(-1)  # [B, out_dim]
        return x

    def add_gaussian_noise(self, input_tensor, sigma=0.1):
        noise = paddle.randn(shape=input_tensor.shape, dtype=input_tensor.dtype) * sigma
        return input_tensor + noise

    def forward(self, x):
        point_cloud, param = x
        # qformer
        enhanced_design = self.qformer(param)  # [B, num_query, hidden_channel]
        enhanced_design = paddle.mean(enhanced_design, axis=1)  # [B, hidden_channel]
        enhanced_design = enhanced_design.unsqueeze(1).tile(
            [1, point_cloud.shape[1], 1]
        )  # [B, N, hidden_channel]
        x = paddle.concat(
            [enhanced_design, point_cloud], axis=-1
        )  # [B, N, hidden_channel + D_point]
        x = x.transpose(perm=[0, 2, 1])  # [bs, dim, np]

        batch_size, _, N = tuple(x.shape)
        x = self.preprocess(x)
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = self.post_process(x4.transpose(perm=[0, 2, 1]))
        if self.pseudo_integration:
            return self.reshape_output(x)
        return x


class ROPE(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # [dim / 2]
        self.inv_freq = 1.0 / (
            self.base
            ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32") / self.dim)
        )
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        # [seq_len, dim/2]
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation
        # in order to obtain the same calculation
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, dim]
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]

    def forward(self, x, seq_len=None):
        cos = self.cos_cached[:, :seq_len, :, :]
        sin = self.sin_cached[:, :seq_len, :, :]
        return (
            cos.cast(x.dtype) if cos.dtype != x.dtype else cos,
            sin.cast(x.dtype) if sin.dtype != x.dtype else sin,
        )


class SA_Layer(nn.Layer):
    def __init__(self, channels, n_heads=8, head_dim=8, seqlen=32):
        super(SA_Layer, self).__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.seqlen = seqlen
        self.wq = paddle.nn.Linear(
            in_features=channels,
            out_features=n_heads * head_dim * seqlen,
            bias_attr=False,
        )
        self.wk = paddle.nn.Linear(
            in_features=channels,
            out_features=n_heads * head_dim * seqlen,
            bias_attr=False,
        )
        self.wv = paddle.nn.Linear(
            in_features=channels,
            out_features=n_heads * head_dim * seqlen,
            bias_attr=False,
        )
        self.softmax = paddle.nn.Softmax(axis=-1)
        self.channels = channels
        self.mlp = paddle.nn.Linear(
            in_features=n_heads * head_dim * seqlen,
            out_features=channels,
            bias_attr=False,
        )
        self.ln = paddle.nn.LayerNorm(normalized_shape=channels)
        self.rope = ROPE(
            dim=self.head_dim * self.seqlen,
            max_position_embeddings=self.n_heads,
            base=10000,
        )

    def forward(self, x):
        B, D, N = tuple(x.shape)
        x = x.transpose(perm=[0, 2, 1])
        x_q = self.wq(x).reshape([B, self.n_heads, N, self.head_dim * self.seqlen])
        x_k = self.wk(x).reshape([B, self.n_heads, N, self.head_dim * self.seqlen])
        x_v = self.wv(x).reshape([B, self.n_heads, N, self.head_dim * self.seqlen])
        cos, sin = self.rope(x_q)
        position_ids = paddle.arange(self.n_heads).expand((B, self.n_heads))
        x_q, x_k, x_v = fused_rotary_position_embedding(
            x_q,
            x_k,
            x_v,
            sin=sin,
            cos=cos,
            position_ids=position_ids,
            use_neox_rotary_style=False,
        )

        x_k = self.softmax(
            paddle.matmul(x=x_k.transpose(perm=[0, 1, 3, 2]), y=x_k)
        ) / math.sqrt(
            self.head_dim * self.seqlen
        )  # N*C * C*N, C*N * N*C = C*C 256
        x_v = self.softmax(
            paddle.matmul(x=x_v.transpose(perm=[0, 1, 3, 2]), y=x_v)
        ) / math.sqrt(self.head_dim * self.seqlen)

        energy = paddle.matmul(x=x_q, y=x_k)
        attention = self.softmax(energy)
        attention = attention / math.sqrt(self.head_dim * self.seqlen)
        x_r = paddle.matmul(x=attention, y=x_v).contiguous().reshape([B, N, -1])
        x = self.ln(x + self.mlp(x_r))
        x = x.transpose(perm=[0, 2, 1])
        return x


class MLP(nn.Layer):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super(MLP, self).__init__()
        act = paddle.nn.GELU
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=n_input, out_features=n_hidden), act()
        )
        self.linear_post = paddle.nn.Linear(in_features=n_hidden, out_features=n_output)
        self.linears = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.Sequential(
                    paddle.nn.Linear(in_features=n_hidden, out_features=n_hidden), act()
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        x = x.transpose(perm=[0, 2, 1])
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        x = x.transpose(perm=[0, 2, 1])
        return x


class QFormer(nn.Layer):
    def __init__(
        self,
        query_dim=256,
        input_dim=64,
        num_query=10,
        num_heads=4,
        head_dim=64,
        ff_dim=512,
    ):
        super(QFormer, self).__init__()
        self.query_dim = query_dim
        self.input_dim = input_dim

        self.query_tokens = self.create_parameter(
            shape=[1, num_query, query_dim],
            default_initializer=nn.initializer.Normal(std=1e-4),
        )

        self.input_proj = nn.Linear(input_dim, query_dim)

        self.cross_attn = nn.MultiHeadAttention(
            embed_dim=query_dim, num_heads=num_heads, dropout=0.0
        )

        self.ln1 = nn.LayerNorm(query_dim)
        self.ln2 = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, query_dim)
        )

    def forward(self, x):  # x: [B, D_in]
        B = x.shape[0]
        query = self.query_tokens.expand([B, -1, -1])  # [B, num_query, query_dim]

        x = self.input_proj(x)  # [B, D_in] -> [B, query_dim]
        x = x.unsqueeze(1)  # [B, 1, query_dim]

        query = self.cross_attn(query, x, x)  # [B, num_query, query_dim]
        query = self.ln1(query)

        # FFN
        residual = query
        query = self.ffn(query)
        query = query + residual
        query = self.ln2(query)
        return query  # [B, num_query, query_dim]


if __name__ == "__main__":
    k = paddle.randn([3, 2, 100, 256], dtype="float16")
    v = paddle.randn([3, 2, 100, 256], dtype="float16")
    q = paddle.randn([3, 2, 100, 256], dtype="float16")
    batch_size, seq_len, num_heads, head_dim = q.shape
    rotary_emb = ROPE(
        dim=head_dim,
        max_position_embeddings=seq_len,
        base=10000,
    )
    cos, sin = rotary_emb(q)
    print(q.shape)
    print(k.shape)
    print(v.shape)
    print(sin.shape)
    print(cos.shape)
    position_ids = paddle.arange(seq_len, dtype="int64").expand((batch_size, seq_len))
    print(position_ids.shape)
    out_q, out_k, out_v = fused_rotary_position_embedding(
        q,
        k,
        v,
        sin=sin,
        cos=cos,
        position_ids=position_ids,
        use_neox_rotary_style=False,
    )
    print(out_q.shape)
