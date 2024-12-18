from collections import defaultdict
import typing as tp
import torch
from torch import autograd

from distrib import average_metrics


def averager(beta: float = 1):
    """
    Exponential Moving Average callback.
    Returns a single function that can be called to repeatidly update the EMA
    with a dict of metrics. The callback will return
    the new averaged dict of metrics.

    Note that for `beta=1`, this is just plain averaging.
    """
    """
    指数移动平均（Exponential Moving Average）回调函数。
    返回一个可以反复调用的函数，用于更新EMA（指数移动平均）并返回新的平均指标字典。

    注意，当 `beta=1` 时，这实际上就是简单的平均。

    Args:
        beta (float, optional): 指数移动平均的衰减因子，范围通常在0到1之间。默认值为1。

    Returns:
        Callable[[Dict[str, Any], float], Dict[str, float]]: 一个函数，该函数接受一个指标字典和一个权重值，
            并返回更新后的平均指标字典。
    """
    # 使用defaultdict初始化fix和total字典，默认值为0.0
    fix: tp.Dict[str, float] = defaultdict(float)
    total: tp.Dict[str, float] = defaultdict(float)

    def _update(metrics: tp.Dict[str, tp.Any], weight: float = 1) -> tp.Dict[str, float]:
        """
        更新EMA并返回新的平均指标字典。

        Args:
            metrics (Dict[str, Any]): 包含指标名称和数值的字典。
            weight (float, optional): 当前更新的权重。默认值为1。

        Returns:
            Dict[str, float]: 更新后的平均指标字典。
        """
        # 声明使用外部函数的total和fix变量
        nonlocal total, fix
        for key, value in metrics.items():
            # 更新total字典：total = total * beta + weight * value
            total[key] = total[key] * beta + weight * float(value)
            # 更新fix字典：fix = fix * beta + weight
            fix[key] = fix[key] * beta + weight
        # 计算新的平均指标：average = total / fix
        return {key: tot / fix[key] for key, tot in total.items()}
    return _update


class Balancer:
    """Loss balancer.

    The loss balancer combines losses together to compute gradients for the backward.
    A call to the balancer will weight the losses according the specified weight coefficients.
    A call to the backward method of the balancer will compute the gradients, combining all the losses and
    potentially rescaling the gradients, which can help stabilize the training and reasonate
    about multiple losses with varying scales.

    Expected usage:
        weights = {'loss_a': 1, 'loss_b': 4}
        balancer = Balancer(weights, ...)
        losses: dict = {}
        losses['loss_a'] = compute_loss_a(x, y)
        losses['loss_b'] = compute_loss_b(x, y)
        if model.training():
            balancer.backward(losses, x)

    ..Warning:: It is unclear how this will interact with DistributedDataParallel,
        in particular if you have some losses not handled by the balancer. In that case
        you can use `encodec.distrib.sync_grad(model.parameters())` and
        `encodec.distrib.sync_buffwers(model.buffers())` as a safe alternative.

    Args:
        weights (Dict[str, float]): Weight coefficient for each loss. The balancer expect the losses keys
            from the backward method to match the weights keys to assign weight to each of the provided loss.
        rescale_grads (bool): Whether to rescale gradients or not, without. If False, this is just
            a regular weighted sum of losses.
        total_norm (float): Reference norm when rescaling gradients, ignored otherwise.
        emay_decay (float): EMA decay for averaging the norms when `rescale_grads` is True.
        per_batch_item (bool): Whether to compute the averaged norm per batch item or not. This only holds
            when rescaling the gradients.
        epsilon (float): Epsilon value for numerical stability.
        monitor (bool): Whether to store additional ratio for each loss key in metrics.
    """
    """
    损失平衡器。

    损失平衡器将多个损失结合起来计算梯度以进行反向传播。
    调用平衡器会根据指定的权重系数对损失进行加权。
    调用平衡器的backward方法将计算梯度，结合所有损失并可能重新缩放梯度，
    这有助于稳定训练并处理多个不同尺度的损失。

    预期用法：
        weights = {'loss_a': 1, 'loss_b': 4}
        balancer = Balancer(weights, ...)
        losses: dict = {}
        losses['loss_a'] = compute_loss_a(x, y)
        losses['loss_b'] = compute_loss_b(x, y)
        if model.training():
            balancer.backward(losses, x)

    ..警告:: 目前尚不清楚这将如何与DistributedDataParallel交互，
        特别是如果你有一些损失没有通过平衡器处理。在这种情况下，
        你可以使用 `encodec.distrib.sync_grad(model.parameters())` 和
        `encodec.distrib.sync_buffers(model.buffers())` 作为安全的替代方案。

    Args:
        weights (Dict[str, float]): 每个损失的权重系数。平衡器期望backward方法中的损失键与权重键匹配，
            以便为每个提供的损失分配权重。
        rescale_grads (bool): 是否重新缩放梯度。如果为False，则这只是损失的常规加权求和。
        total_norm (float): 在重新缩放梯度时参考的范数，否则忽略。
        ema_decay (float): 当 `rescale_grads` 为True时，用于平均范数的EMA衰减率。
        per_batch_item (bool): 是否按批次项计算平均范数。这仅在重新缩放梯度时有效。
        epsilon (float): 用于数值稳定的epsilon值。
        monitor (bool): 是否在指标中存储每个损失键的额外比率。
    """

    def __init__(self, weights: tp.Dict[str, float], rescale_grads: bool = True, total_norm: float = 1.,
                 ema_decay: float = 0.999, per_batch_item: bool = True, epsilon: float = 1e-12,
                 monitor: bool = False):
        """
        初始化Balancer实例。

        Args:
            weights (Dict[str, float]): 每个损失的权重系数。
            rescale_grads (bool, optional): 是否重新缩放梯度。默认为True。
            total_norm (float, optional): 在重新缩放梯度时参考的范数。默认为1.0。
            ema_decay (float, optional): EMA衰减率，用于平均范数。默认为0.999。
            per_batch_item (bool, optional): 是否按批次项计算平均范数。默认为True。
            epsilon (float, optional): 用于数值稳定的epsilon值。默认为1e-12。
            monitor (bool, optional): 是否在指标中存储每个损失键的额外比率。默认为False。
        """
        self.weights = weights
        self.per_batch_item = per_batch_item
        self.total_norm = total_norm
        self.averager = averager(ema_decay)
        self.epsilon = epsilon
        self.monitor = monitor
        self.rescale_grads = rescale_grads
        self._metrics: tp.Dict[str, tp.Any] = {}

    @property
    def metrics(self):
        """
        获取当前的指标。

        Returns:
            Dict[str, Any]: 当前存储的指标。
        """
        return self._metrics

    def backward(self, losses: tp.Dict[str, torch.Tensor], input: torch.Tensor):
        """
        计算梯度并进行反向传播。

        Args:
            losses (Dict[str, torch.Tensor]): 包含损失名称和对应张量的字典。
            input (torch.Tensor): 输入张量，用于计算梯度。
        """
        # 存储每个损失的范数
        norms = {}
        # 存储每个损失的梯度
        grads = {}
        for name, loss in losses.items():
            grad, = autograd.grad(loss, [input], retain_graph=True)
            if self.per_batch_item:
                dims = tuple(range(1, grad.dim()))
                norm = grad.norm(dim=dims).mean()
            else:
                norm = grad.norm()
            norms[name] = norm
            grads[name] = grad

        count = 1
        if self.per_batch_item:
            count = len(grad)
        # 计算平均范数
        avg_norms = average_metrics(self.averager(norms), count)
        total = sum(avg_norms.values())

        self._metrics = {}
        if self.monitor:
            for k, v in avg_norms.items():
                self._metrics[f'ratio_{k}'] = v / total

        # 计算总权重
        total_weights = sum([self.weights[k] for k in avg_norms])
        # 计算每个损失的比率
        ratios = {k: w / total_weights for k, w in self.weights.items()}

        out_grad: tp.Any = 0
        for name, avg_norm in avg_norms.items():
            if self.rescale_grads:
                scale = ratios[name] * self.total_norm / (self.epsilon + avg_norm)
                # 重新缩放梯度
                grad = grads[name] * scale
            else:
                grad = self.weights[name] * grads[name]
            out_grad += grad
        input.backward(out_grad)


def test():
    """
    测试函数，用于验证 Balancer 类的正确性。

    测试流程：
    1. 创建一个需要梯度的张量 x。
    2. 定义两个损失函数：loss_1 和 loss_2。
    3. 使用 Balancer 对这两个损失进行加权求和，并计算梯度。
    4. 断言梯度的值是否符合预期。
    5. 重复上述步骤，但这次启用梯度重新缩放，并断言梯度是否符合预期。
    """
    # 创建一个需要梯度的张量 x，初始值为0
    from torch.nn import functional as F
    x = torch.zeros(1, requires_grad=True)

    # 创建一个与 x 形状相同的全1张量
    one = torch.ones_like(x)
    # 定义第一个损失函数：L1 损失，期望值为1
    loss_1 = F.l1_loss(x, one)
    # 定义第二个损失函数：L1 损失，期望值为-1，并乘以100
    loss_2 = 100 * F.l1_loss(x, -one)
    # 将两个损失函数放入字典中
    losses = {'1': loss_1, '2': loss_2}

    # 创建一个 Balancer 实例，权重为1:1，不重新缩放梯度
    balancer = Balancer(weights={'1': 1, '2': 1}, rescale_grads=False)
    # 调用 backward 方法，计算梯度
    balancer.backward(losses, x)
    assert torch.allclose(x.grad, torch.tensor(99.)), x.grad

    # 重新定义损失函数
    loss_1 = F.l1_loss(x, one)
    loss_2 = 100 * F.l1_loss(x, -one)
    losses = {'1': loss_1, '2': loss_2}
    x.grad = None
    # 创建一个 Balancer 实例，权重为1:1，并重新缩放梯度
    balancer = Balancer(weights={'1': 1, '2': 1}, rescale_grads=True)
    # 调用 backward 方法，计算梯度
    balancer.backward({'1': loss_1, '2': loss_2}, x)
    assert torch.allclose(x.grad, torch.tensor(0.)), x.grad


if __name__ == '__main__':
    test()
