import paddle


class MLP(paddle.nn.Layer):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(MLP, self).__init__()
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        self.layers = paddle.nn.LayerList()
        for j in range(self.n_layers):
            self.layers.append(
                paddle.nn.Linear(in_features=layers[j], out_features=layers[j + 1])
            )
            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(
                        paddle.nn.BatchNorm1D(num_features=layers[j + 1])
                    )
                self.layers.append(nonlinearity())
        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x


class PositionalEmbedding(paddle.nn.Layer):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = paddle.arange(start=0, end=self.num_channels // 2, dtype="float32")
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        freqs = paddle.cast(freqs, x.dtype)
        x = x.outer(y=freqs)
        x = paddle.concat(x=[x.cos(), x.sin()], axis=1)
        return x
