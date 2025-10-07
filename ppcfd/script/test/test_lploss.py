import pytest
import paddle
import numpy as np
from ppcfd.utils.loss import LpLoss


class TestLpLossAbs:
    @pytest.fixture
    def lp_loss(self):
        """
        计算Lp损失。

        Args:
        无

        Returns:
        LpLoss: Lp损失对象，其中d=2表示维度为2，p=2表示使用L2范数。

        """
        return LpLoss(d=2, p=2)

    def test_abs_same_input(self, lp_loss):
        """
        测试lp_loss.abs函数在输入相同的情况下是否能返回0.0。

        Args:
        self: 测试类实例对象。
        lp_loss: lp_loss对象，包含abs函数。

        Returns:
        无返回值。

        Raises:
        无异常抛出。

        """
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        y = x.clone()
        result = lp_loss.abs(x, y)
        assert result.item() == pytest.approx(0.0)

    def test_abs_different_input(self, lp_loss):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        y = paddle.to_tensor([[0.0, 1.0], [2.0, 3.0]])

        # Compute the expected result dynamically
        diff = x - y
        expected = paddle.norm(diff, p=2).numpy()  # L2 norm (p=2) of the difference
        result = lp_loss.abs(x, y)
        assert result.item() == pytest.approx(expected, rel=1e-6), f"Expected {expected}, got {result.item()}"

    def test_abs_batch_input(self, lp_loss):
        x = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0],
                                                         [7.0, 8.0]]])
        y = paddle.to_tensor([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0],
                                                         [6.0, 7.0]]])
        result = lp_loss.abs(x, y)
        expected = np.sqrt(8) * (1.0 /
                                 (2 - 1))  # 2 examples, each with 4 elements
        assert result.item() == pytest.approx(expected)

    def test_abs_no_reduction(self):
        lp_loss = LpLoss(d=2, p=2, reduction=False)
        x = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0],
                                                         [7.0, 8.0]]])
        y = paddle.to_tensor([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0],
                                                         [6.0, 7.0]]])
        result = lp_loss.abs(x, y)
        expected = paddle.to_tensor([np.sqrt(4), np.sqrt(4)]) * (1.0 / (2 - 1))
        assert paddle.allclose(result, expected)

    def test_abs_sum_reduction(self):
        lp_loss = LpLoss(d=2, p=2, size_average=False)
        x = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0],
                                                         [7.0, 8.0]]])
        y = paddle.to_tensor([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0],
                                                         [6.0, 7.0]]])
        diff = x - y
        expected = paddle.norm(diff, p=2).numpy()  # L2 norm (p=2) of the difference
        result = lp_loss.abs(x, y)
        assert result.item() == pytest.approx(expected)

    def test_abs_different_p_norm(self):
        lp_loss = LpLoss(d=2, p=1)
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        y = paddle.to_tensor([[0.0, 1.0], [2.0, 3.0]])
        result = lp_loss.abs(x, y)
        expected = 4 * (1.0 /
                        (2 - 1))  # L1 norm is sum of absolute differences
        assert result.item() == pytest.approx(expected)

    def test_abs_raises_on_shape_mismatch(self, lp_loss):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        y = paddle.to_tensor([1.0, 2.0, 3.0])
        with pytest.raises(AssertionError):
            lp_loss.abs(x, y)
        
        # Compute the expected result dynamically
        diff = x - y
        expected = paddle.norm(diff, p=2).numpy()  # L2 norm (p=2) of the difference
        
        result = lp_loss.abs(x, y)
        expected = np.sqrt(4) * (1.0 / (2 - 1))  # h=1/(2-1)=1, d=2, p=2
        assert result.item() == pytest.approx(expected)

    def test_abs_batch_input(self, lp_loss):
        x = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0],
                                                         [7.0, 8.0]]])
        y = paddle.to_tensor([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0],
                                                         [6.0, 7.0]]])
        result = lp_loss.abs(x, y)
        expected = np.sqrt(8) * (1.0 /
                                 (2 - 1))  # 2 examples, each with 4 elements
        assert result.item() == pytest.approx(expected)

    def test_abs_no_reduction(self):
        lp_loss = LpLoss(d=2, p=2, reduction=False)
        x = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0],
                                                         [7.0, 8.0]]])
        y = paddle.to_tensor([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0],
                                                         [6.0, 7.0]]])
        result = lp_loss.abs(x, y)
        expected = paddle.norm(x-y, p=2)
        assert paddle.allclose(result, expected)

    def test_abs_sum_reduction(self):
        lp_loss = LpLoss(d=2, p=2, size_average=False)
        x = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0],
                                                         [7.0, 8.0]]])
        y = paddle.to_tensor([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0],
                                                         [6.0, 7.0]]])
        result = lp_loss.abs(x, y)
        diff = x - y
        expected = paddle.norm(diff, p=2).numpy()  # L2 norm (p=2) of the difference
        assert result.item() == pytest.approx(expected)

    def test_abs_different_p_norm(self):
        lp_loss = LpLoss(d=2, p=1)
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        y = paddle.to_tensor([[0.0, 1.0], [2.0, 3.0]])
        result = lp_loss.abs(x, y)
        expected = 4 * (1.0 /
                        (2 - 1))  # L1 norm is sum of absolute differences
        assert result.item() == pytest.approx(expected)

    def test_abs_raises_on_shape_mismatch(self, lp_loss):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        y = paddle.to_tensor([1.0, 2.0, 3.0])
        with pytest.raises(AssertionError):
            lp_loss.abs(x, y)

class TestLpLossRel:
    @pytest.fixture
    def lp_loss(self):
        """返回LpLoss对象用于测试rel方法"""
        return LpLoss(d=2, p=2)

    def test_rel_same_input(self, lp_loss):
        """测试相同输入时相对误差为0"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        y = x.clone()
        result = lp_loss.rel(x, y)
        assert result.item() == pytest.approx(0.0)

    def test_rel_different_input(self, lp_loss):
        """测试不同输入时的相对误差计算"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        # x = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
        y = paddle.to_tensor([[0.0, 1.0], [2.0, 3.0]])

        # 计算预期结果: abs(x-y)/abs(x)
        numerator = paddle.norm(x.flatten() - y.flatten(), p=2)
        denominator = paddle.norm(y.flatten(), p=2)
        expected = (numerator / denominator).numpy()
        result = lp_loss.rel(x, y)
        assert result.item() == pytest.approx(expected, rel=1e-6)

    def test_rel_batch_input(self, lp_loss):
        """
        测试批量输入的相对误差计算。

        Args:
        self (object): 类实例本身。
        lp_loss (object): 损失函数对象，用于计算相对误差。

        Returns:
        None

        """
        x = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]], 
                             [[5.0, 6.0], [7.0, 8.0]]])
        y = paddle.to_tensor([[[0.0, 1.0], [2.0, 3.0]], 
                             [[4.0, 5.0], [6.0, 7.0]]])
        
        # 计算预期结果: 平均相对误差
        diff = paddle.norm(x[0] - y[0], p=2)
        true_norm = paddle.norm(y[0], p=2)
        l2_b1 = (diff / true_norm).mean().numpy()

        diff = paddle.norm(x[1] - y[1], p=2)
        true_norm = paddle.norm(y[1], p=2)
        l2_b2 = (diff / true_norm).mean().numpy()
        expected = (l2_b1 + l2_b2) / 2
        result = lp_loss.rel(x, y)
        assert result.item() == pytest.approx(expected)

    def test_rel_zero_input(self, lp_loss):
        """
        测试输入包含0时的相对误差计算。

        Args:
        self: 类实例对象。
        lp_loss: 一个用于计算相对误差的对象。

        Returns:
        无返回值。

        Raises:
        断言错误: 如果计算得到的相对误差与预期结果不匹配，则抛出断言错误。
        """
        x = paddle.to_tensor([[0.0, 2.0], [3.0, 4.0]])
        y = paddle.to_tensor([[0.0, 1.0], [2.0, 3.0]])

        # 预期结果: 忽略0值计算
        mask = x != 0  # 创建布尔掩码
        x_masked = paddle.masked_select(x, mask).reshape([-1, 1])  # 提取非零元素
        y_masked = paddle.masked_select(y, mask).reshape([-1, 1])  # 提取对应位置的y值

        # 计算分子和分母
        numerator = paddle.norm(x_masked - y_masked, p=2)
        denominator = paddle.norm(y_masked, p=2)

        # 检查分母是否为零，避免除零错误
        if denominator.item() == 0:
            expected = 0.0  # 如果分母为0，则相对误差定义为0
        else:
            expected = (numerator / denominator).item()

        import pdb;pdb.set_trace()
        result = lp_loss.rel(x, y)
        assert result.item() == pytest.approx(expected, rel=1e-6), f"Expected {expected}, got {result.item()}"



    def test_rel_raises_on_shape_mismatch(self, lp_loss):
        """测试形状不匹配时抛出异常"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        y = paddle.to_tensor([1.0, 2.0, 3.0])
        with pytest.raises(AssertionError):
            lp_loss.rel(x, y)
