import paddle


class LpLoss(paddle.nn.Layer):
    def __init__(self, d=2, p=2, size_average=True, reduction=True, enable_dp=False):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.enable_dp = enable_dp

    def abs(self, x, y):
        assert x.shape == y.shape
        num_examples = x.shape[0]

        # Assume uniform mesh
        h = 1.0 / (x.shape[1] - 1.0)
        w = h ** (self.d / self.p)
        diff = x.reshape((num_examples, -1)) - y.reshape((num_examples, -1))
        all_norms = w * paddle.norm(diff, self.p)

        if self.reduction:
            if self.size_average:
                return paddle.mean(all_norms)
            else:
                return paddle.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        assert x.shape == y.shape, f'{x.shape} != {y.shape}'
        if x.ndim == 2:
            axis = None
            start_axis = 0
            keepdim = False
        elif x.ndim == 3:
            axis = 1
            start_axis = 1
            keepdim = False
        else:
            raise ValueError(
                'Input x and y must be [N,C] or [B,N,C], received respectively {} and {}'.format(x.ndim, y.ndim))
        
        diff_norms = paddle.norm((x - y).flatten(start_axis=start_axis), self.p, axis=axis, keepdim=keepdim)
        y_norms = paddle.norm(y.flatten(start_axis=start_axis), self.p, axis=axis, keepdim=keepdim)
        
        if self.reduction:
            if self.size_average:
                loss = paddle.mean(diff_norms / y_norms) # mean over batch
                if self.enable_dp and paddle.distributed.get_world_size() > 1:
                    paddle.distributed.all_reduce(loss, paddle.distributed.ReduceOp.SUM)
                    loss = loss / paddle.distributed.get_world_size()
                return loss
            else:
                return paddle.sum(diff_norms / y_norms)
        else:
            loss = paddle.mean(diff_norms / y_norms, axis=1) # keep batch
            if self.enable_dp and paddle.distributed.get_world_size() > 1:
                paddle.distributed.all_reduce(loss, paddle.distributed.ReduceOp.SUM)
                loss = loss / paddle.distributed.get_world_size()
            return loss
        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
