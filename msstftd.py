import typing as tp

import torchaudio
import torch
from torch import nn
from einops import rearrange

from modules import NormConv2d


# 定义类型别名
# 特征图类型，列表中包含多个张量
FeatureMapType = tp.List[torch.Tensor]
# 逻辑输出类型，张量类型
LogitsType = torch.Tensor
# 判别器的输出类型，由逻辑输出列表和特征图列表组成
DiscriminatorOutput = tp.Tuple[tp.List[LogitsType], tp.List[FeatureMapType]]


def get_2d_padding(kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)):
    """
    计算二维卷积的填充大小，以确保输出特征图的空间尺寸与输入相同。

    计算公式：
        padding_height = ((kernel_size_height - 1) * dilation_height) // 2
        padding_width = ((kernel_size_width - 1) * dilation_width) // 2

    Args:
        kernel_size (Tuple[int, int]): 卷积核的尺寸，表示为 (高度, 宽度)。
        dilation (Tuple[int, int], optional): 膨胀率，表示为 (高度, 宽度)。默认为 (1, 1)。

    Returns:
        Tuple[int, int]: 计算得到的填充大小，表示为 (填充高度, 填充宽度)。
    """
    # 计算填充高度、宽度，返回填充尺寸的二元组
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)


class DiscriminatorSTFT(nn.Module):
    """STFT sub-discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_fft (int): Size of FFT for each scale. Default: 1024
        hop_length (int): Length of hop between STFT windows for each scale. Default: 256
        kernel_size (tuple of int): Inner Conv2d kernel sizes. Default: ``(3, 9)``
        stride (tuple of int): Inner Conv2d strides. Default: ``(1, 2)``
        dilations (list of int): Inner Conv2d dilation on the time dimension. Default: ``[1, 2, 4]``
        win_length (int): Window size for each scale. Default: 1024
        normalized (bool): Whether to normalize by magnitude after stft. Default: True
        norm (str): Normalization method. Default: `'weight_norm'`
        activation (str): Activation function. Default: `'LeakyReLU'`
        activation_params (dict): Parameters to provide to the activation function.
        growth (int): Growth factor for the filters. Default: 1
    """
    """
    STFT子判别器。
    
    Args:
        filters (int): 卷积层中的滤波器数量。
        in_channels (int): 输入通道数。默认: 1。
        out_channels (int): 输出通道数。默认: 1。
        n_fft (int): 每个尺度的FFT大小。默认: 1024。
        hop_length (int): 每个尺度STFT窗口之间的跳步长度。默认: 256。
        kernel_size (tuple of int): 内部Conv2d卷积核大小。默认: ``(3, 9)``。
        stride (tuple of int): 内部Conv2d卷积步幅。默认: ``(1, 2)``。
        dilations (list of int): 内部Conv2d在时间维度上的膨胀率。默认: ``[1, 2, 4]``。
        win_length (int): 每个尺度的窗口大小。默认: 1024。
        normalized (bool): 是否在STFT后进行幅度归一化。默认: True。
        norm (str): 归一化方法。默认: `'weight_norm'`。
        activation (str): 激活函数类型。默认: `'LeakyReLU'`。
        activation_params (dict): 提供给激活函数的参数。
        growth (int): 滤波器的增长因子。默认: 1。
    """
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1,
                 n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024, max_filters: int = 1024,
                 filters_scale: int = 1, kernel_size: tp.Tuple[int, int] = (3, 9), dilations: tp.List = [1, 2, 4],
                 stride: tp.Tuple[int, int] = (1, 2), normalized: bool = True, norm: str = 'weight_norm',
                 activation: str = 'LeakyReLU', activation_params: dict = {'negative_slope': 0.2}):
        super().__init__()
        # 检查kernel_size和stride是否为二元组
        assert len(kernel_size) == 2
        assert len(stride) == 2

        # 初始化参数
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)

        # 初始化Spectrogram变换
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window_fn=torch.hann_window,
            normalized=self.normalized, center=False, pad_mode=None, power=None)
        
        # 计算输入通道数（实部和虚部）
        spec_channels = 2 * self.in_channels

        # 初始化卷积层列表
        self.convs = nn.ModuleList()
        # 添加第一个卷积层
        self.convs.append(
            NormConv2d(spec_channels, self.filters, kernel_size=kernel_size, padding=get_2d_padding(kernel_size))
        )
        # 计算输入通道数
        in_chs = min(filters_scale * self.filters, max_filters)
        # 添加中间的卷积层
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(NormConv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride,
                                         dilation=(dilation, 1), padding=get_2d_padding(kernel_size, (dilation, 1)),
                                         norm=norm))
            in_chs = out_chs
        # 计算最终的输出通道数
        out_chs = min((filters_scale ** (len(dilations) + 1)) * self.filters, max_filters)
        # 添加最后一个卷积层
        self.convs.append(NormConv2d(in_chs, out_chs, kernel_size=(kernel_size[0], kernel_size[0]),
                                     padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                                     norm=norm))
        # 添加最终的卷积层
        self.conv_post = NormConv2d(out_chs, self.out_channels,
                                    kernel_size=(kernel_size[0], kernel_size[0]),
                                    padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                                    norm=norm)

    def forward(self, x: torch.Tensor):
        """
        前向传播过程。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: 输出特征图和中间特征图列表。
        """
        # 用于存储中间特征图
        fmap = []
        # 计算频谱 [B, 2, Freq, Frames, 2]
        z = self.spec_transform(x)  # [B, 2, Freq, Frames, 2]
        # 拼接实部和虚部
        z = torch.cat([z.real, z.imag], dim=1)
        # 调整维度顺序
        z = rearrange(z, 'b c w t -> b c t w')
        for i, layer in enumerate(self.convs):
            # 应用卷积层
            z = layer(z)
            # 应用激活函数
            z = self.activation(z)
            # 存储中间特征图
            fmap.append(z)
        # 应用最终的卷积层
        z = self.conv_post(z)
        return z, fmap


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-Scale STFT (MS-STFT) discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_ffts (Sequence[int]): Size of FFT for each scale
        hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale
        win_lengths (Sequence[int]): Window size for each scale
        **kwargs: additional args for STFTDiscriminator
    """
    """
    多尺度短时傅里叶变换（MS-STFT）判别器。

    Args:
        filters (int): 卷积层中的滤波器数量。
        in_channels (int): 输入通道数。默认: 1。
        out_channels (int): 输出通道数。默认: 1。
        n_ffts (List[int]): 每个尺度的FFT大小列表。
        hop_lengths (List[int]): 每个尺度的STFT窗口之间的跳步长度列表。
        win_lengths (List[int]): 每个尺度的窗口大小列表。
        **kwargs: 传递给STFTDiscriminator的其他参数。
    """
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1,
                 n_ffts: tp.List[int] = [1024, 2048, 512], hop_lengths: tp.List[int] = [256, 512, 128],
                 win_lengths: tp.List[int] = [1024, 2048, 512], **kwargs):
        super().__init__()
        # 检查每个尺度的参数列表长度是否一致
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        # 初始化判别器列表
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(filters, in_channels=in_channels, out_channels=out_channels,
                              n_fft=n_ffts[i], win_length=win_lengths[i], hop_length=hop_lengths[i], **kwargs)
            for i in range(len(n_ffts))
        ])
        # 记录判别器的数量
        self.num_discriminators = len(self.discriminators)

    def forward(self, x: torch.Tensor) -> DiscriminatorOutput:
        """
        前向传播过程。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            DiscriminatorOutput: 包含每个判别器的逻辑输出和特征图的元组。
        """
        # 用于存储每个判别器的逻辑输出
        logits = []
        # 用于存储每个判别器的特征图
        fmaps = []
        for disc in self.discriminators:
            # 对输入进行判别
            logit, fmap = disc(x)
            # 添加逻辑输出
            logits.append(logit)
            # 添加特征图
            fmaps.append(fmap)
        return logits, fmaps


def test():
    """
    测试函数，用于验证多尺度STFT判别器（MultiScaleSTFTDiscriminator）的输出是否符合预期。
    """
    # 实例化一个多尺度STFT判别器，滤波器数量设置为32
    disc = MultiScaleSTFTDiscriminator(filters=32)

    # 生成两个随机张量，形状均为 [1, 1, 24000]，分别代表真实音频和生成音频
    # 真实音频样本
    y = torch.randn(1, 1, 24000)
    # 生成音频样本（假样本）
    y_hat = torch.randn(1, 1, 24000)

    # 将真实音频输入到判别器中，得到逻辑输出和特征图
    # y_disc_r: 真实样本的逻辑输出, fmap_r: 真实样本的特征图
    y_disc_r, fmap_r = disc(y)

    # 将生成音频输入到判别器中，得到逻辑输出和特征图
    # y_disc_gen: 生成样本的逻辑输出, fmap_gen: 生成样本的特征图
    y_disc_gen, fmap_gen = disc(y_hat)

    # 所有列表的长度应等于判别器的数量
    assert len(y_disc_r) == len(y_disc_gen) == len(fmap_r) == len(fmap_gen) == disc.num_discriminators
    
    # 所有特征图列表中的特征图数量应等于5
    assert all([len(fm) == 5 for fm in fmap_r + fmap_gen])
    # 所有特征图的前两个维度应等于 [1, 32]
    assert all([list(f.shape)[:2] == [1, 32] for fm in fmap_r + fmap_gen for f in fm])
    # 所有逻辑输出的维度数量应等于4
    assert all([len(logits.shape) == 4 for logits in y_disc_r + y_disc_gen])


if __name__ == '__main__':
    test()
