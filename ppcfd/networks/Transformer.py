import math
from paddle.incubate.nn.functional import fused_rotary_position_embedding


import paddle
import paddle.nn as nn


class ROPE(paddle.nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # [dim / 2]
        self.inv_freq = 1.0 / (self.base ** (paddle.cast(paddle.arange(0, self.dim, 2), dtype="float32") / self.dim))
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        # [seq_len, dim/2]
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
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




class MLP(paddle.nn.Layer):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True, use_kan=False):
        super(MLP, self).__init__()
        act = paddle.nn.GELU
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.use_kan = use_kan # 保存标志

        # 根据 use_kan 标志选择层类型
        linear_layer = paddle.nn.Linear
        activation_layer = paddle.nn.GELU # 使用 paddle.nn.GELU 类

        # 注意：KANLinear 可能需要调整激活函数的使用方式，这里假设可以像 Linear 一样后接激活
        self.linear_pre = paddle.nn.Sequential(
            linear_layer(in_features=n_input, out_features=n_hidden), # 替换
            activation_layer() # 使用 GELU 激活
        )
        self.linear_post = linear_layer(in_features=n_hidden, out_features=n_output) # 替换


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

class LearnablePositionalEmbedding(paddle.nn.Layer):
    def __init__(self, d_model, max_len=4):
        super().__init__()
        self.mlp = paddle.nn.Linear(in_features=d_model, out_features=max_len)
        self.ln = paddle.nn.LayerNorm(normalized_shape=max_len)

    def forward(self, x):
        x = self.mlp(x)
        x = self.ln(x).unsqueeze(1)
        return x


class SA_Layer(paddle.nn.Layer):
    def __init__(self, channels, n_heads=8, head_dim=8, seqlen=1):
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
        x_q, x_k, x_v = fused_rotary_position_embedding(x_q, x_k, x_v, sin=sin, cos=cos, position_ids=position_ids, use_neox_rotary_style=False)

        x_k = self.softmax(
            paddle.matmul(x=x_k.transpose(perm=[0, 1, 3, 2]), y=x_k)
        ) / math.sqrt(self.head_dim * self.seqlen) # N*C * C*N, C*N * N*C = C*C 256
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


class QFormer(nn.Layer):
    def __init__(self, query_dim=256, input_dim=64, num_query=10, num_heads=4, head_dim=64, ff_dim=512):
        super(QFormer, self).__init__()
        self.query_dim = query_dim
        self.input_dim = input_dim

        # 可学习的 query 向量
        self.query_tokens = self.create_parameter(
            shape=[1, num_query, query_dim],
            default_initializer=nn.initializer.Normal(std=1e-4)
        )

        # 输入投影层，将 input_dim 映射到 query_dim
        self.input_proj = nn.Linear(input_dim, query_dim)

        # Cross attention - 注意 embed_dim 要等于 query_dim
        self.cross_attn = nn.MultiHeadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=0.0
        )

        self.ln1 = nn.LayerNorm(query_dim)
        self.ln2 = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, query_dim)
        )

    def forward(self, x):  # x: [B, D_in]
        B = x.shape[0]
        query = self.query_tokens.expand([B, -1, -1])  # [B, num_query, query_dim]

        # 投影输入到正确的维度
        x = self.input_proj(x)  # [B, D_in] -> [B, query_dim]
        x = x.unsqueeze(1)  # [B, 1, query_dim]

        # Cross attention
        query = self.cross_attn(query, x, x)  # [B, num_query, query_dim]
        query = self.ln1(query)

        # FFN
        residual = query
        query = self.ffn(query)
        query = query + residual
        query = self.ln2(query)

        return query  # [B, num_query, query_dim]


class Point_Transformer(paddle.nn.Layer):
    def __init__(self, in_dim=3, q_input_dim=0, out_dim=1, hidden_channel=256, reshape=False, out_num=10, remove_channel_dim=False, **kwargs):
        super(Point_Transformer, self).__init__()
        self.out_dim = out_dim
        self.preprocess = MLP(
            in_dim, hidden_channel * 2, hidden_channel, n_layers=2, res=False
        )

        if q_input_dim > 0:
            self.positional_encoding = LearnablePositionalEmbedding(q_input_dim)
        n_heads = 2
        head_dim = 8
        seqlen = 1

        self.qformer = QFormer(
            query_dim=hidden_channel,
            input_dim=q_input_dim,
            num_query=10,
            num_heads=2,
            head_dim=8,
            ff_dim=16
        )

        self.sa1 = SA_Layer(hidden_channel, n_heads, head_dim, seqlen)
        self.sa2 = SA_Layer(hidden_channel, n_heads, head_dim, seqlen)
        self.sa3 = SA_Layer(hidden_channel, n_heads, head_dim, seqlen)
        self.sa4 = SA_Layer(hidden_channel, n_heads, head_dim, seqlen)


        self.post_process = paddle.nn.Linear(in_features=hidden_channel, out_features=out_dim)

        self.reshape = reshape
        self.remove_channel_dim = remove_channel_dim

        if reshape:
            self.ln1 = paddle.nn.LayerNorm(normalized_shape=hidden_channel)
            self.ln2 = paddle.nn.LayerNorm(normalized_shape=hidden_channel)
            self.ln3 = paddle.nn.LayerNorm(normalized_shape=hidden_channel//2)
            self.ln4 = paddle.nn.LayerNorm(normalized_shape=hidden_channel//4)
            self.pooling1 = paddle.nn.MaxPool1D(kernel_size=64)
            self.pooling2 = paddle.nn.MaxPool1D(kernel_size=64)
            self.pooling3 = paddle.nn.MaxPool1D(kernel_size=64)
            self.mlp1 = paddle.nn.Linear(hidden_channel, hidden_channel//2)
            self.mlp2 = paddle.nn.Linear(hidden_channel//2, hidden_channel//4)
            self.mlp3 = paddle.nn.Linear(hidden_channel//4, out_dim)

    def reshape_output(self, x):
        x = self.ln1(x)
        x = x.transpose(perm=[0, 2, 1])
        x = self.pooling1(x)
        x = x.transpose([0, 2, 1])
        x = self.mlp1(self.ln2(x))
        x = x.transpose([0, 2, 1])       # [B, 128, 1562]
        x = self.pooling2(x)
        x = x.transpose([0, 2, 1])
        x = self.mlp2(self.ln3(x))
        x = x.transpose([0, 2, 1])       # [B, 64, 24]
        x = self.pooling3(x)
        x = x.transpose([0, 2, 1])       # [B, out_dim, 64]
        x = self.mlp3(self.ln4(x))
        x = x.squeeze(-1)                # [B, out_dim]
        return x

    def add_gaussian_noise(self, input_tensor, sigma=0.1):
        noise = paddle.randn(shape=input_tensor.shape, dtype=input_tensor.dtype) * sigma
        return input_tensor + noise

    def forward(self, x):
        if isinstance(x, tuple):
            # design_parameter, point_cloud = x
            point_cloud, design_parameter = x

            # qformer
            enhanced_design = self.qformer(design_parameter)  # [B, num_query, hidden_channel]
            enhanced_design = paddle.mean(enhanced_design, axis=1)  # [B, hidden_channel]
            enhanced_design = enhanced_design.unsqueeze(1).tile([1, point_cloud.shape[1], 1])  # [B, N, hidden_channel]

            x = paddle.concat([enhanced_design, point_cloud], axis=-1)  # [B, N, hidden_channel + D_point]

        x = x.transpose(perm=[0, 2, 1])
        batch_size, _, N = tuple(x.shape)
        x = self.preprocess(x)  # [B, H, N]

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)

        x = self.post_process(x4.transpose(perm=[0, 2, 1]))
        if self.reshape:
            return self.reshape_output(x)
        return x
