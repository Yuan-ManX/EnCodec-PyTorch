import math
from pathlib import Path
import typing as tp

import numpy as np
import torch
from torch import nn

import quantization as qt
import modules as m
from utils import _check_checksum, _linear_overlap_add, _get_checkpoint_url


# 定义根URL（根据实际需要设置）
ROOT_URL = ''

# 定义编码帧的类型别名
EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]


class LMModel(nn.Module):
    """Language Model to estimate probabilities of each codebook entry.
    We predict all codebooks in parallel for a given time step.

    Args:
        n_q (int): number of codebooks.
        card (int): codebook cardinality.
        dim (int): transformer dimension.
        **kwargs: passed to `encodec.modules.transformer.StreamingTransformerEncoder`.
    """
    """
    语言模型，用于估计每个码本条目的概率。
    我们并行预测给定时间步的所有码本。

    参数:
        n_q (int): 码本的数量。默认值为32。
        card (int): 码本的基数。默认值为1024。
        dim (int): Transformer的维度。默认值为200。
        **kwargs: 传递给 `encodec.modules.transformer.StreamingTransformerEncoder` 的参数。
    """
    def __init__(self, n_q: int = 32, card: int = 1024, dim: int = 200, **kwargs):
        super().__init__()
        self.card = card  # 码本的基数
        self.n_q = n_q    # 码本的数量
        self.dim = dim    # Transformer的维度
        # 初始化StreamingTransformerEncoder
        self.transformer = m.StreamingTransformerEncoder(dim=dim, **kwargs)
        # 为每个码本初始化一个嵌入层
        self.emb = nn.ModuleList([nn.Embedding(card + 1, dim) for _ in range(n_q)])
        # 为每个码本初始化一个线性层
        self.linears = nn.ModuleList([nn.Linear(dim, card) for _ in range(n_q)])

    def forward(self, indices: torch.Tensor,
                states: tp.Optional[tp.List[torch.Tensor]] = None, offset: int = 0):
        """
        Args:
            indices (torch.Tensor): indices from the previous time step. Indices
                should be 1 + actual index in the codebook. The value 0 is reserved for
                when the index is missing (i.e. first time step). Shape should be
                `[B, n_q, T]`.
            states: state for the streaming decoding.
            offset: offset of the current time step.

        Returns a 3-tuple `(probabilities, new_states, new_offset)` with probabilities
        with a shape `[B, card, n_q, T]`.

        """
        """
        前向传播方法。
        参数:
            indices (torch.Tensor): 前一时间步的索引。索引应该是1加上码本中的实际索引。
                值0保留用于索引缺失（即第一个时间步）。形状应为 `[B, n_q, T]`。
            states (Optional[List[torch.Tensor]]): 流式解码的状态。
            offset (int): 当前时间步的偏移量。

        返回一个3元组 `(probabilities, new_states, new_offset)`，其中概率的形状为 `[B, card, n_q, T]`。

        """
        # 获取输入张量的形状
        B, K, T = indices.shape
        # 对每个码本的索引进行嵌入，并求和得到输入张量
        input_ = sum([self.emb[k](indices[:, k]) for k in range(K)])
        # 通过Transformer进行编码
        out, states, offset = self.transformer(input_, states, offset)
        # 对每个码本应用线性层，并堆叠结果
        logits = torch.stack([self.linears[k](out) for k in range(K)], dim=1).permute(0, 3, 1, 2)
        # 对logits应用softmax函数，得到概率
        return torch.softmax(logits, dim=1), states, offset


class EncodecModel(nn.Module):
    """EnCodec model operating on the raw waveform.
    Args:
        target_bandwidths (list of float): Target bandwidths.
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        normalize (bool): Whether to apply audio normalization.
        segment (float or None): segment duration in sec. when doing overlap-add.
        overlap (float): overlap between segment, given as a fraction of the segment duration.
        name (str): name of the model, used as metadata when compressing audio.
    """
    """
    EnCodec 模型对原始波形进行操作。

    参数:
        target_bandwidths (list of float): 目标带宽列表。
        encoder (nn.Module): 编码器网络。
        decoder (nn.Module): 解码器网络。
        sample_rate (int): 音频采样率。
        channels (int): 音频通道数。
        normalize (bool): 是否应用音频归一化。
        segment (float 或 None): 进行重叠相加时的片段持续时间，以秒为单位。
        overlap (float): 片段之间的重叠量，表示为片段持续时间的比例。
        name (str): 模型的名称，在压缩音频时用作元数据。
    """
    def __init__(self,
                 encoder: m.SEANetEncoder,
                 decoder: m.SEANetDecoder,
                 quantizer: qt.ResidualVectorQuantizer,
                 target_bandwidths: tp.List[float],
                 sample_rate: int,
                 channels: int,
                 normalize: bool = False,
                 segment: tp.Optional[float] = None,
                 overlap: float = 0.01,
                 name: str = 'unset'):
        super().__init__()
        # 当前带宽，初始化为None
        self.bandwidth: tp.Optional[float] = None
        # 目标带宽列表
        self.target_bandwidths = target_bandwidths
        # 编码器网络
        self.encoder = encoder
        # 量化器
        self.quantizer = quantizer
        # 解码器网络
        self.decoder = decoder
        # 音频采样率
        self.sample_rate = sample_rate
        # 音频通道数
        self.channels = channels
        # 是否应用归一化
        self.normalize = normalize
        # 片段持续时间
        self.segment = segment
        # 重叠比例
        self.overlap = overlap
        # 计算帧率，即每秒的帧数
        self.frame_rate = math.ceil(self.sample_rate / np.prod(self.encoder.ratios))
        # 模型名称
        self.name = name
        # 计算每个码本中的位数
        self.bits_per_codebook = int(math.log2(self.quantizer.bins))
        # 确保码本数量是2的幂
        assert 2 ** self.bits_per_codebook == self.quantizer.bins, \
            "quantizer bins must be a power of 2."

    @property
    def segment_length(self) -> tp.Optional[int]:
        """
        获取片段的长度（以样本数为单位）。
        如果没有设置片段持续时间，则返回None。
        """
        if self.segment is None:
            return None
        return int(self.segment * self.sample_rate)

    @property
    def segment_stride(self) -> tp.Optional[int]:
        """
        获取片段的步幅（以样本数为单位）。
        步幅是片段长度乘以（1 - 重叠比例）。
        如果片段长度未设置，则返回None。
        """
        segment_length = self.segment_length
        if segment_length is None:
            return None
        return max(1, int((1 - self.overlap) * segment_length))

    def encode(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
        """Given a tensor `x`, returns a list of frames containing
        the discrete encoded codes for `x`, along with rescaling factors
        for each segment, when `self.normalize` is True.

        Each frames is a tuple `(codebook, scale)`, with `codebook` of
        shape `[B, K, T]`, with `K` the number of codebooks.
        """
        """
        对输入张量 `x` 进行编码，返回一个包含离散编码码本的帧列表，
        以及当 `self.normalize` 为 True 时，每个片段的缩放因子。

        每个帧是一个元组 `(codebook, scale)`，其中 `codebook` 的形状为 `[B, K, T]`，`K` 是码本的数量。

        参数:
            x (torch.Tensor): 输入音频张量，形状为 `[B, C, T]`，其中 `B` 是批量大小，`C` 是通道数，`T` 是样本数。

        返回:
            List[EncodedFrame]: 编码帧列表。
        """
        # 确保输入张量是三维的
        assert x.dim() == 3
        # 获取通道数和长度
        _, channels, length = x.shape
        # 确保通道数在1到2之间（单声道或立体声）
        assert channels > 0 and channels <= 2
        # 获取片段长度
        segment_length = self.segment_length
        # 如果片段长度未设置，则将整个输入作为单个片段处理
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            # 否则，使用预定义的步幅
            stride = self.segment_stride  # type: ignore
            assert stride is not None

        encoded_frames: tp.List[EncodedFrame] = []
        # 按步幅遍历输入张量，提取每个片段
        for offset in range(0, length, stride):
            frame = x[:, :, offset: offset + segment_length]
            # 对每个片段进行编码，并添加到帧列表中
            encoded_frames.append(self._encode_frame(frame))
        # 返回编码后的帧列表
        return encoded_frames

    def _encode_frame(self, x: torch.Tensor) -> EncodedFrame:
        """
        对单个音频帧进行编码。

        参数:
            frame (torch.Tensor): 单个音频帧，形状为 `[B, C, T]`。

        返回:
            EncodedFrame: 编码帧，包含码本和缩放因子。
        """
        # 获取帧的长度
        length = x.shape[-1]
        # 计算帧的持续时间（秒）
        duration = length / self.sample_rate
        # 确保帧的持续时间不超过片段持续时间（如果有设置）
        assert self.segment is None or duration <= 1e-5 + self.segment

        if self.normalize:
            # 如果启用归一化，则将帧转换为单声道并计算音量
            mono = x.mean(dim=1, keepdim=True)
            # 计算缩放因子，避免除以零
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            # 调整缩放因子形状以便后续使用
            scale = scale.view(-1, 1)
        else:
            # 如果未启用归一化，则缩放因子为None
            scale = None

        # 通过编码器网络进行编码
        emb = self.encoder(x)
        # 使用量化器进行编码
        codes = self.quantizer.encode(emb, self.frame_rate, self.bandwidth)
        # 调整码本的维度顺序
        codes = codes.transpose(0, 1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        # codes 的形状为 [B, K, T]，其中 T 是帧数，K 是码本数量
        return codes, scale

    def decode(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
        """Decode the given frames into a waveform.
        Note that the output might be a bit bigger than the input. In that case,
        any extra steps at the end can be trimmed.
        """
        """
        将给定的编码帧解码为波形。
        注意，输出可能比输入稍大。在这种情况下，可以在末尾修剪多余的步骤。

        参数:
            encoded_frames (List[EncodedFrame]): 编码帧列表。

        返回:
            torch.Tensor: 解码后的波形张量。
        """
        segment_length = self.segment_length
        # 如果片段长度未设置，则断言只有一个编码帧
        if segment_length is None:
            assert len(encoded_frames) == 1
            # 解码单个帧
            return self._decode_frame(encoded_frames[0])

        # 解码所有编码帧
        frames = [self._decode_frame(frame) for frame in encoded_frames]
        # 使用线性重叠相加方法合并帧
        return _linear_overlap_add(frames, self.segment_stride or 1)

    def _decode_frame(self, encoded_frame: EncodedFrame) -> torch.Tensor:
        """
        解码单个编码帧。

        Args:
            encoded_frame (torch.Tensor): 编码帧，包含codes和scale。

        Returns:
            torch.Tensor: 解码后的帧。
        """
        codes, scale = encoded_frame
        # 交换维度顺序
        codes = codes.transpose(0, 1)
        # 使用量化器解码codes
        emb = self.quantizer.decode(codes)
        # 使用解码器生成输出
        out = self.decoder(emb)
        # 如果存在缩放因子，则应用缩放
        if scale is not None:
            out = out * scale.view(-1, 1, 1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播过程：编码输入并解码。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 解码后的输出张量。
        """
        # 编码输入
        frames = self.encode(x)
        # 解码编码后的帧，并截取与输入相同长度的部分
        return self.decode(frames)[:, :, :x.shape[-1]]

    def set_target_bandwidth(self, bandwidth: float):
        """
        设置目标带宽。

        Args:
            bandwidth (float): 目标带宽（kHz）。

        Raises:
            ValueError: 如果提供的带宽不在支持列表中。
        """
        if bandwidth not in self.target_bandwidths:
            raise ValueError(f"This model doesn't support the bandwidth {bandwidth}. "
                             f"Select one of {self.target_bandwidths}.")
        self.bandwidth = bandwidth

    def get_lm_model(self) -> LMModel:
        """Return the associated LM model to improve the compression rate.
        """
        """
        获取关联的语言模型（LM）以提高压缩率。

        Returns:
            LMModel: 加载并返回的语言模型实例。
        """
        device = next(self.parameters()).device
        # 初始化语言模型
        lm = LMModel(self.quantizer.n_q, self.quantizer.bins, num_layers=5, dim=200,
                     past_context=int(3.5 * self.frame_rate)).to(device)
        
        # 定义预训练模型的检查点名称
        checkpoints = {
            'encodec_24khz': 'encodec_lm_24khz-1608e3c0.th',
            'encodec_48khz': 'encodec_lm_48khz-7add9fc3.th',
        }
        try:
            # 根据当前模型名称获取对应的检查点名称
            checkpoint_name = checkpoints[self.name]
        except KeyError:
            raise RuntimeError("No LM pre-trained for the current Encodec model.")
        
        # 构建检查点的完整URL
        url = _get_checkpoint_url(ROOT_URL, checkpoint_name)
        # 从URL加载检查点
        state = torch.hub.load_state_dict_from_url(
            url, map_location='cpu', check_hash=True)  # type: ignore
        # 将加载的状态加载到语言模型中
        lm.load_state_dict(state)
        # 设置模型为评估模式
        lm.eval()
        return lm

    @staticmethod
    def _get_model(target_bandwidths: tp.List[float],
                   sample_rate: int = 24_000,
                   channels: int = 1,
                   causal: bool = True,
                   model_norm: str = 'weight_norm',
                   audio_normalize: bool = False,
                   segment: tp.Optional[float] = None,
                   name: str = 'unset'):
        """
        创建一个Encodec模型实例。

        Args:
            target_bandwidths (List[float]): 目标带宽列表。
            sample_rate (int, optional): 采样率，默认24kHz。
            channels (int, optional): 声道数，默认单声道。
            causal (bool, optional): 是否使用因果卷积，默认True。
            model_norm (str, optional): 模型归一化方式，默认为'weight_norm'。
            audio_normalize (bool, optional): 是否对音频进行归一化，默认False。
            segment (float, optional): 音频片段长度。
            name (str, optional): 模型名称，默认为'unset'。

        Returns:
            EncodecModel: 初始化后的Encodec模型实例。
        """
        # 初始化编码器
        encoder = m.SEANetEncoder(channels=channels, norm=model_norm, causal=causal)
        # 初始化解码器
        decoder = m.SEANetDecoder(channels=channels, norm=model_norm, causal=causal)
        # 计算量化器的量化级别n_q
        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / encoder.hop_length) * 10))

        # 初始化量化器
        quantizer = qt.ResidualVectorQuantizer(
            dimension=encoder.dimension,
            n_q=n_q,
            bins=1024,
        )

        # 初始化Encodec模型
        model = EncodecModel(
            encoder,
            decoder,
            quantizer,
            target_bandwidths,
            sample_rate,
            channels,
            normalize=audio_normalize,
            segment=segment,
            name=name,
        )
        return model

    @staticmethod
    def _get_pretrained(checkpoint_name: str, repository: tp.Optional[Path] = None):
        """
        从本地目录或URL加载预训练模型。

        Args:
            checkpoint_name (str): 检查点文件名。
            repository (Path, optional): 本地目录路径。如果提供，将从本地加载；否则，从URL加载。

        Returns:
            OrderedDict: 加载的模型状态字典。

        Raises:
            ValueError: 如果本地目录不存在或不是目录。
        """
        if repository is not None:
            if not repository.is_dir():
                raise ValueError(f"{repository} must exist and be a directory.")
            file = repository / checkpoint_name
            checksum = file.stem.split('-')[1]
            _check_checksum(file, checksum)
            return torch.load(file)
        else:
            url = _get_checkpoint_url(ROOT_URL, checkpoint_name)
            return torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=True)  # type:ignore

    @staticmethod
    def encodec_model_24khz(pretrained: bool = True, repository: tp.Optional[Path] = None):
        """Return the pretrained causal 24khz model.
        """
        """
        返回预训练的24kHz模型。

        Args:
            pretrained (bool, optional): 是否加载预训练模型，默认True。
            repository (Path, optional): 本地目录路径。如果提供，将从本地加载；否则，从URL加载。

        Returns:
            EncodecModel: 初始化并加载预训练权重的24kHz Encodec模型。
        """
        if repository:
            assert pretrained
        # 定义目标带宽
        target_bandwidths = [1.5, 3., 6, 12., 24.]
        # 定义检查点名称
        checkpoint_name = 'encodec_24khz-d7cc33bc.th'
        # 定义采样率
        sample_rate = 24_000
        # 定义声道数
        channels = 1
        # 创建模型实例
        model = EncodecModel._get_model(
            target_bandwidths, sample_rate, channels,
            causal=True, model_norm='weight_norm', audio_normalize=False,
            name='encodec_24khz' if pretrained else 'unset')
        
        # 如果需要加载预训练模型
        if pretrained:
            state_dict = EncodecModel._get_pretrained(checkpoint_name, repository)
            model.load_state_dict(state_dict)

        # 设置模型为评估模式
        model.eval()
        return model

    @staticmethod
    def encodec_model_48khz(pretrained: bool = True, repository: tp.Optional[Path] = None):
        """Return the pretrained 48khz model.
        """
        """
        返回预训练的48kHz模型。

        Args:
            pretrained (bool, optional): 是否加载预训练模型，默认True。
            repository (Path, optional): 本地目录路径。如果提供，将从本地加载；否则，从URL加载。

        Returns:
            EncodecModel: 初始化并加载预训练权重的48kHz Encodec模型。
        """
        if repository:
            assert pretrained
        # 定义目标带宽
        target_bandwidths = [3., 6., 12., 24.]
        # 定义检查点名称
        checkpoint_name = 'encodec_48khz-7e698e3e.th'
        # 定义采样率
        sample_rate = 48_000
        # 定义声道数
        channels = 2
        # 创建模型实例
        model = EncodecModel._get_model(
            target_bandwidths, sample_rate, channels,
            causal=False, model_norm='time_group_norm', audio_normalize=True,
            segment=1., name='encodec_48khz' if pretrained else 'unset')
        # 如果需要加载预训练模型
        if pretrained:
            state_dict = EncodecModel._get_pretrained(checkpoint_name, repository)
            model.load_state_dict(state_dict)
        # 设置模型为评估模式
        model.eval()
        return model


def test():
    """
    测试函数，用于验证不同带宽和采样率的Encodec模型对音频文件的编码和解码效果。
    """
    # 导入笛卡尔积工具，用于生成所有模型和带宽的组合
    from itertools import product
    # 导入torchaudio库，用于加载和保存音频文件
    import torchaudio

    # 定义要测试的带宽列表（单位：kHz）
    bandwidths = [3, 6, 12, 24]

    # 定义可用的Encodec模型，键为模型名称，值为对应的模型获取函数
    models = {
        'encodec_24khz': EncodecModel.encodec_model_24khz,  # 24kHz的Encodec模型
        'encodec_48khz': EncodecModel.encodec_model_48khz   # 48kHz的Encodec模型
    }

    # 遍历所有模型和带宽的组合
    for model_name, bw in product(models.keys(), bandwidths):
        # 根据模型名称获取对应的模型获取函数并实例化模型
        model = models[model_name]()
        # 设置模型的当前目标带宽
        model.set_target_bandwidth(bw)
        # 从模型名称中提取音频文件的后缀，例如 '24khz' 或 '48khz'
        audio_suffix = model_name.split('_')[1][:3]
        # 加载测试音频文件，文件名格式为 'test_24.wav' 或 'test_48.wav'
        wav, sr = torchaudio.load(f"test_{audio_suffix}.wav")
        # 截取音频样本，确保其长度不超过模型采样率的2倍
        wav = wav[:, :model.sample_rate * 2]
        # 在时间维度上增加一个维度，以适应模型的输入要求
        wav_in = wav.unsqueeze(0)
        # 将音频输入到模型中进行编码和解码
        wav_dec = model(wav_in)[0]
        # 确保编码前后的音频形状相同，如果不同则抛出异常
        assert wav.shape == wav_dec.shape, (wav.shape, wav_dec.shape)


if __name__ == '__main__':
    
    test()
