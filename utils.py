from hashlib import sha256
from pathlib import Path
import typing as tp

import torch
import torchaudio


def _linear_overlap_add(frames: tp.List[torch.Tensor], stride: int):
    """
    通用的重叠相加函数，支持线性淡入淡出，适用于复杂场景（例如，每个位置有超过2个帧）。
    核心思想是使用一个三角形权重函数，该函数在段的中点达到最大值。
    在求和帧时使用此权重函数，并在最后对每个位置的权重求和进行归一化。
    因此：
        - 如果一个帧是唯一覆盖某个位置的帧，则权重函数不执行任何操作。
        - 如果两个帧覆盖某个位置：
              ...  ...
             /   \/   \
            /    /\    \
                  S  T         ，即 S 是第二个帧的起始偏移，T 是第一个帧的结束偏移。
    那么每个帧的权重函数为：(t - S), (T - t)，其中 `t` 是给定的偏移量。
    在最终归一化后，第二个帧在位置 `t` 的权重为
    (t - S) / (t - S + (T - t)) = (t - S) / (T - S)，这正是我们想要的。
    
    - 如果超过两个帧在某个点重叠，我们希望通过归纳法得到合理的结果。
    """
    assert len(frames)
    device = frames[0].device
    # 获取数据类型
    dtype = frames[0].dtype
    # 获取帧的形状（不包括最后一维）
    shape = frames[0].shape[:-1]
    # 计算输出总长度
    total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]
    
    # 获取帧的长度
    frame_length = frames[0].shape[-1]
    # 生成归一化时间点
    t = torch.linspace(0, 1, frame_length + 2, device=device, dtype=dtype)[1: -1]
    # 计算线性权重（三角函数）
    weight = 0.5 - (t - 0.5).abs()

    # 初始化权重求和张量
    sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
    # 初始化输出张量
    out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
    # 初始化偏移量
    offset: int = 0

    for frame in frames:
        # 获取当前帧的长度
        frame_length = frame.shape[-1]
        # 应用权重并累加帧
        out[..., offset:offset + frame_length] += weight[:frame_length] * frame
        # 累加权重
        sum_weight[offset:offset + frame_length] += weight[:frame_length]
        # 更新偏移量
        offset += stride
    assert sum_weight.min() > 0
    # 返回归一化后的输出
    return out / sum_weight


def _get_checkpoint_url(root_url: str, checkpoint: str):
    """
    拼接检查点文件的完整URL。

    参数:
    - root_url (str): 根URL地址。
    - checkpoint (str): 检查点文件名。

    返回:
    - str: 拼接后的完整URL。
    """
    # 如果根URL不以斜杠结尾，则添加一个斜杠
    if not root_url.endswith('/'):
        root_url += '/'
    # 返回拼接后的完整URL
    return root_url + checkpoint


def _check_checksum(path: Path, checksum: str):
    """
    验证文件的SHA-256校验和是否与预期的校验和匹配。

    参数:
    - path (Path): 要验证的文件路径。
    - checksum (str): 预期的校验和字符串。

    异常:
    - RuntimeError: 如果实际校验和不匹配预期的校验和。
    """
    # 创建一个SHA-256哈希对象
    sha = sha256()
    # 以二进制读取模式打开文件
    with open(path, 'rb') as file:
        while True:
            # 每次读取1MB的数据块
            buf = file.read(2**20) # 2**20 bytes = 1 MB
            if not buf:
                # 如果没有更多的数据，退出循环
                break
            # 更新哈希对象
            sha.update(buf)
    # 计算文件的实际校验和，并截取与预期校验和相同的长度
    actual_checksum = sha.hexdigest()[:len(checksum)]
    # 比较实际校验和与预期校验和
    if actual_checksum != checksum:
        raise RuntimeError(f'Invalid checksum for file {path}, '
                           f'expected {checksum} but got {actual_checksum}')


def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    """
    转换音频的采样率和通道数。

    参数:
    - wav (torch.Tensor): 输入的音频张量，形状为 [..., C, L]，其中C是通道数，L是样本长度。
    - sr (int): 当前音频的采样率。
    - target_sr (int): 目标采样率。
    - target_channels (int): 目标通道数（1表示单声道，2表示立体声）。

    返回:
    - torch.Tensor: 转换后的音频张量，形状为 [..., target_channels, L']，其中L'是新的样本长度。
    
    异常:
    - AssertionError: 如果音频张量维度不足或通道数不符合要求。
    - RuntimeError: 如果无法从当前通道数转换为目标通道数。
    """
    # 音频张量至少有两个维度
    assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"
    # 音频通道数必须是1（单声道）或2（立体声）
    assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."

    # 解包张量形状
    *shape, channels, length = wav.shape
    # 根据目标通道数调整音频张量
    if target_channels == 1:
        # 如果目标是单声道，则对通道维度取平均
        wav = wav.mean(-2, keepdim=True)
    elif target_channels == 2:
        # 如果目标是立体声，则扩展通道维度
        wav = wav.expand(*shape, target_channels, length)
    elif channels == 1:
        # 如果当前是单声道且目标通道数不是1或2，则尝试扩展到目标通道数
        wav = wav.expand(target_channels, -1)
    else:
        # 如果无法从当前通道数转换为目标通道数，则抛出异常
        raise RuntimeError(f"Impossible to convert from {channels} to {target_channels}")
    # 使用Torchaudio的Resample变换调整采样率
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    # 返回转换后的音频张量
    return wav


def save_audio(wav: torch.Tensor, path: tp.Union[Path, str],
               sample_rate: int, rescale: bool = False):
    """
    保存音频张量为文件。

    参数:
    - wav (torch.Tensor): 要保存的音频张量。
    - path (Path | str): 保存文件的路径。
    - sample_rate (int): 音频的采样率。
    - rescale (bool): 是否重新缩放音频以避免削波。默认为False。

    异常:
    - AssertionError: 如果音频张量维度不足。
    - ValueError: 如果音频张量包含的元素超出预期范围。
    """
    # 定义音频的最大幅度限制
    limit = 0.99

    # 计算音频张量的绝对值最大值
    mx = wav.abs().max()
    if rescale:
        # 如果需要重新缩放，则将音频缩放到不超过limit
        wav = wav * min(limit / mx, 1)
    else:
        # 否则，将音频剪辑到[-limit, limit]范围
        wav = wav.clamp(-limit, limit)
    
    # 使用Torchaudio保存音频文件，编码为PCM_S（带符号的16位PCM）
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)
