import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LRQSolver(nn.Module):
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
        q_query_dim=None,
        q_input_dim=23,
        q_num_query=10,
        q_num_heads=2,
        q_head_dim=8,
        q_ff_dim=16,
        remove_channel_dim=False,
        **kwargs,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.preprocess = MLP(
            in_dim, hidden_channel * 2, hidden_channel, n_layers=2, res=False
        )

        query_dim = q_query_dim if q_query_dim is not None else hidden_channel // 4
        self.qformer = QFormer(
            query_dim=query_dim,
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
        post_out_dim = hidden_channel if pseudo_integration else out_dim
        self.post_process = nn.Linear(hidden_channel, post_out_dim)

        self.pseudo_integration = pseudo_integration
        self.remove_channel_dim = remove_channel_dim

        if pseudo_integration:
            self.ln1 = nn.LayerNorm(hidden_channel)
            self.ln2 = nn.LayerNorm(hidden_channel)
            self.ln3 = nn.LayerNorm(hidden_channel // 2)
            self.ln4 = nn.LayerNorm(hidden_channel // 4)
            self.pooling1 = nn.MaxPool1d(kernel_size=64)
            self.pooling2 = nn.MaxPool1d(kernel_size=64)
            self.pooling3 = nn.AdaptiveMaxPool1d(1)
            self.mlp1 = nn.Linear(hidden_channel, hidden_channel // 2)
            self.mlp2 = nn.Linear(hidden_channel // 2, hidden_channel // 4)
            self.mlp3 = nn.Linear(hidden_channel // 4, out_dim)

    def reshape_output(self, x):
        x = self.ln1(x)
        x = x.transpose(1, 2)
        x = self._pool_reduce(x)
        x = x.transpose(1, 2)
        x = self.mlp1(self.ln2(x))
        x = x.transpose(1, 2)
        x = self._pool_reduce(x)
        x = x.transpose(1, 2)
        x = self.mlp2(self.ln3(x))
        x = x.transpose(1, 2)
        x = self.pooling3(x)
        x = x.transpose(1, 2)
        x = self.mlp3(self.ln4(x))
        return x.squeeze(-1)

    @staticmethod
    def _pool_reduce(x, kernel_size=64):
        if x.shape[-1] >= kernel_size:
            return F.max_pool1d(x, kernel_size=kernel_size)
        return F.adaptive_max_pool1d(x, 1)

    def add_gaussian_noise(self, input_tensor, sigma=0.1):
        return input_tensor + torch.randn_like(input_tensor) * sigma

    def forward(self, x):
        point_cloud, param = x
        enhanced_design = self.qformer(param)
        enhanced_design = enhanced_design.mean(dim=1)
        enhanced_design = enhanced_design.unsqueeze(1).expand(
            -1, point_cloud.shape[1], -1
        )
        x = torch.cat([enhanced_design, point_cloud], dim=-1)
        x = x.transpose(1, 2)

        x = self.preprocess(x)
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = self.post_process(x4.transpose(1, 2))
        if self.pseudo_integration:
            return self.reshape_output(x)
        return x


class ROPE(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, :, None, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, :, None, :], persistent=False
        )

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        cos = self.cos_cached[:, :seq_len].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:, :seq_len].to(dtype=x.dtype, device=x.device)
        return cos, sin


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


class SA_Layer(nn.Module):
    def __init__(self, channels, n_heads=8, head_dim=8, seqlen=32):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.seqlen = seqlen
        out_features = n_heads * head_dim * seqlen
        self.wq = nn.Linear(channels, out_features, bias=False)
        self.wk = nn.Linear(channels, out_features, bias=False)
        self.wv = nn.Linear(channels, out_features, bias=False)
        self.channels = channels
        self.mlp = nn.Linear(out_features, channels, bias=False)
        self.ln = nn.LayerNorm(channels)
        self.rope = ROPE(
            dim=self.head_dim * self.seqlen,
            max_position_embeddings=self.n_heads,
            base=10000,
        )

    def forward(self, x):
        batch_size, _, n_points = x.shape
        x = x.transpose(1, 2)
        hidden = self.head_dim * self.seqlen

        x_q = self.wq(x).reshape(batch_size, self.n_heads, n_points, hidden)
        x_k = self.wk(x).reshape(batch_size, self.n_heads, n_points, hidden)
        x_v = self.wv(x).reshape(batch_size, self.n_heads, n_points, hidden)

        cos, sin = self.rope(x_q)
        x_q = (x_q * cos) + (rotate_half(x_q) * sin)
        x_k = (x_k * cos) + (rotate_half(x_k) * sin)
        x_v = (x_v * cos) + (rotate_half(x_v) * sin)

        scale = math.sqrt(hidden)
        x_k = torch.softmax(x_k.transpose(-1, -2) @ x_k, dim=-1) / scale
        x_v = torch.softmax(x_v.transpose(-1, -2) @ x_v, dim=-1) / scale

        energy = x_q @ x_k
        attention = torch.softmax(energy, dim=-1) / scale
        x_r = (attention @ x_v).contiguous().reshape(batch_size, n_points, -1)
        x = self.ln(x + self.mlp(x_r))
        return x.transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super().__init__()
        activation = nn.GELU
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), activation())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(n_hidden, n_hidden), activation())
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.linear_pre(x)
        for layer in self.linears:
            x = layer(x) + x if self.res else layer(x)
        x = self.linear_post(x)
        return x.transpose(1, 2)


class QFormer(nn.Module):
    def __init__(
        self,
        query_dim=256,
        input_dim=64,
        num_query=10,
        num_heads=4,
        head_dim=64,
        ff_dim=512,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.input_dim = input_dim

        self.query_tokens = nn.Parameter(torch.empty(1, num_query, query_dim))
        nn.init.normal_(self.query_tokens, std=1e-4)

        self.input_proj = nn.Linear(input_dim, query_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim, num_heads=num_heads, dropout=0.0, batch_first=True
        )

        self.ln1 = nn.LayerNorm(query_dim)
        self.ln2 = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, query_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        query = self.query_tokens.expand(batch_size, -1, -1)
        x = self.input_proj(x).unsqueeze(1)
        query, _ = self.cross_attn(query, x, x, need_weights=False)
        query = self.ln1(query)

        residual = query
        query = self.ffn(query)
        query = query + residual
        return self.ln2(query)


if __name__ == "__main__":
    model = LRQSolver(in_dim=67, q_query_dim=64, q_input_dim=23, pseudo_integration=True)
    points = torch.randn(2, 1024, 3)
    params = torch.randn(2, 23)
    print(model((points, params)).shape)
