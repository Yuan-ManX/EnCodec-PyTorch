import typing as tp
import torch


def rank():
    """
    获取当前进程的排名。

    如果分布式环境已经初始化，则返回当前进程的排名。
    否则，返回0，表示单机环境。

    Returns:
        int: 当前进程的排名或0。
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def world_size():
    """
    获取分布式环境中的总进程数。

    如果分布式环境已经初始化，则返回总进程数（world size）。
    否则，返回1，表示单机环境。

    Returns:
        int: 总进程数或1。
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def is_distributed():
    """
    判断当前是否处于分布式训练模式。

    如果总进程数大于1，则表示处于分布式训练模式。
    否则，表示处于单机模式。

    Returns:
        bool: 如果处于分布式训练模式则返回True，否则返回False。
    """
    return world_size() > 1


def all_reduce(tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM):
    """
    对分布式环境中的所有进程执行全规约操作。

    如果当前处于分布式训练模式，则对给定的张量执行全规约操作。
    否则，不进行任何操作。

    Args:
        tensor (torch.Tensor): 需要进行全规约操作的张量。
        op (torch.distributed.ReduceOp, optional): 规约操作的类型，默认为求和。

    Returns:
        torch.Tensor: 全规约后的张量。
    """
    if is_distributed():
        return torch.distributed.all_reduce(tensor, op)


def _is_complex_or_float(tensor):
    """
    判断张量是否为复数或浮点数类型。

    Args:
        tensor (torch.Tensor): 需要判断的张量。

    Returns:
        bool: 如果张量为浮点数或复数类型则返回True，否则返回False。
    """
    return torch.is_floating_point(tensor) or torch.is_complex(tensor)


def _check_number_of_params(params: tp.List[torch.Tensor]):
    # utility function to check that the number of params in all workers is the same,
    # and thus avoid a deadlock with distributed all reduce.
    """
    实用函数，用于检查所有工作节点中的参数数量是否相同，
    以避免分布式全规约操作中出现死锁。

    如果当前不是分布式环境，或者参数列表为空，则不进行任何操作。

    Args:
        params (List[torch.Tensor]): 参数张量列表。

    Raises:
        RuntimeError: 如果至少有一个工作节点的参数数量与其他节点不匹配。
    """
    if not is_distributed() or not params:
        return
    # 创建一个包含参数数量的张量
    tensor = torch.tensor([len(params)], device=params[0].device, dtype=torch.long)
    # 对所有工作节点执行全规约操作，累加参数数量
    all_reduce(tensor)
    # 检查所有工作节点的参数总数是否等于单个节点的参数数量乘以工作节点总数
    if tensor.item() != len(params) * world_size():
        # If not all the workers have the same number, for at least one of them,
        # this inequality will be verified.
        # 如果不匹配，则抛出运行时错误，提示参数数量不一致
        raise RuntimeError(f"Mismatch in number of params: ours is {len(params)}, "
                           "at least one worker has a different one.")


def broadcast_tensors(tensors: tp.Iterable[torch.Tensor], src: int = 0):
    """Broadcast the tensors from the given parameters to all workers.
    This can be used to ensure that all workers have the same model to start with.
    """
    """
    将给定的参数张量从源工作节点广播到所有工作节点。
    这可以用于确保所有工作节点在开始时具有相同的模型。

    Args:
        tensors (Iterable[torch.Tensor]): 需要广播的张量迭代器。
        src (int, optional): 源工作节点的排名，默认为0。

    Raises:
        RuntimeError: 如果当前不是分布式环境，或者张量列表为空。
    """
    if not is_distributed():
        return
    # 过滤出浮点数或复数类型的张量
    tensors = [tensor for tensor in tensors if _is_complex_or_float(tensor)]
    # 检查参数数量是否一致
    _check_number_of_params(tensors)
    handles = []
    for tensor in tensors:
        # 对每个张量启动异步广播操作
        handle = torch.distributed.broadcast(tensor.data, src=src, async_op=True)
        handles.append(handle)
    # 等待所有广播操作完成
    for handle in handles:
        handle.wait()


def sync_buffer(buffers, average=True):
    """
    Sync grad for buffers. If average is False, broadcast instead of averaging.
    """
    """
    同步缓冲区的梯度。如果 `average` 为 False，则进行广播而不是平均。

    Args:
        buffers: 需要同步的缓冲区列表。
        average (bool, optional): 是否对梯度进行平均。默认为 True。

    Raises:
        RuntimeError: 如果当前不是分布式环境。
    """
    if not is_distributed():
        return
    handles = []
    for buffer in buffers:
        if torch.is_floating_point(buffer.data):
            if average:
                # 对缓冲区的数据进行全规约求和（异步操作）
                handle = torch.distributed.all_reduce(
                    buffer.data, op=torch.distributed.ReduceOp.SUM, async_op=True)
            else:
                # 将缓冲区的数据从源节点广播到所有节点（异步操作）
                handle = torch.distributed.broadcast(
                    buffer.data, src=0, async_op=True)
            # 将缓冲区和对应的操作句柄存储起来
            handles.append((buffer, handle))
    for buffer, handle in handles:
        # 等待异步操作完成
        handle.wait()
        if average:
            # 对缓冲区数据进行平均
            buffer.data /= world_size


def sync_grad(params):
    """
    Simpler alternative to DistributedDataParallel, that doesn't rely
    on any black magic. For simple models it can also be as fast.
    Just call this on your model parameters after the call to backward!
    """
    """
    同步模型参数的梯度。这是一个更简单的替代 DistributedDataParallel 的方法，
    不依赖于任何黑魔法。对于简单模型，它也可以同样快速。
    只需在调用 backward 之后，对模型参数调用此函数即可！

    Args:
        params: 需要同步的模型参数列表。

    Raises:
        RuntimeError: 如果当前不是分布式环境。
    """
    if not is_distributed():
        return
    handles = []
    for p in params:
        if p.grad is not None:
            # 对梯度数据进行全规约求和（异步操作）
            handle = torch.distributed.all_reduce(
                p.grad.data, op=torch.distributed.ReduceOp.SUM, async_op=True)
            # 将参数和对应的操作句柄存储起来
            handles.append((p, handle))
    for p, handle in handles:
        # 等待异步操作完成
        handle.wait()
        # 对梯度数据进行平均
        p.grad.data /= world_size()


def average_metrics(metrics: tp.Dict[str, float], count=1.):
    """Average a dictionary of metrics across all workers, using the optional
    `count` as unnormalized weight.
    """
    """
    在所有工作节点之间平均指标字典，使用可选的 `count` 作为未归一化的权重。

    Args:
        metrics (Dict[str, float]): 需要平均的指标字典。
        count (float, optional): 未归一化的权重。默认为1。

    Returns:
        Dict[str, float]: 平均后的指标字典。

    Raises:
        RuntimeError: 如果当前不是分布式环境。
    """
    if not is_distributed():
        return metrics
    keys, values = zip(*metrics.items())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 创建一个张量，包含所有指标值和一个额外的1
    tensor = torch.tensor(list(values) + [1], device=device, dtype=torch.float32)
    # 将张量乘以count
    tensor *= count
    # 对张量进行全规约求和
    all_reduce(tensor)
    # 计算平均值（除以最后一个元素，即所有节点的count总和）
    averaged = (tensor[:-1] / tensor[-1]).cpu().tolist()
    # 返回平均后的指标字典
    return dict(zip(keys, averaged))
