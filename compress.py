import io
import math
import struct
import time
import typing as tp

import torch

import binary
from quantization.ac import ArithmeticCoder, ArithmeticDecoder, build_stable_quantized_cdf
from model import EncodecModel, EncodedFrame


MODELS = {
    'encodec_24khz': EncodecModel.encodec_model_24khz, # 24kHz的Encodec模型
    'encodec_48khz': EncodecModel.encodec_model_48khz, # 48kHz的Encodec模型
}


def compress_to_file(model: EncodecModel, wav: torch.Tensor, fo: tp.IO[bytes],
                     use_lm: bool = True):
    """Compress a waveform to a file-object using the given model.

    Args:
        model (EncodecModel): a pre-trained EncodecModel to use to compress the audio.
        wav (torch.Tensor): waveform to compress, should have a shape `[C, T]`, with `C`
            matching `model.channels`, and the proper sample rate (e.g. `model.sample_rate`).
            Use `utils.convert_audio` if this is not the case.
        fo (IO[bytes]): file-object to which the compressed bits will be written.
            See `compress` if you want obtain a `bytes` object instead.
        use_lm (bool): if True, use a pre-trained language model to further
            compress the stream using Entropy Coding. This will slow down compression
            quite a bit, expect between 20 to 30% of size reduction.
    """
    """
    使用给定的模型将波形压缩到文件对象中。

    Args:
        model (EncodecModel): 用于压缩音频的预训练Encodec模型。
        wav (torch.Tensor): 要压缩的波形，张量形状应为 `[C, T]`，其中 `C` 应与 `model.channels` 匹配，且采样率正确（例如 `model.sample_rate`）。
            如果不是这种情况，请使用 `utils.convert_audio` 进行转换。
        fo (IO[bytes]): 写入压缩比特流的文件对象。如果需要获得 `bytes` 对象，请使用 `compress` 函数。
        use_lm (bool, optional): 如果为True，则使用预训练的语言模型通过熵编码进一步压缩流。这会显著减慢压缩速度，但预计可以减少20%到30%的体积。默认为True。
    """
    # 确保输入张量的维度是否为2
    assert wav.dim() == 2, "Only single waveform can be encoded."
    if model.name not in MODELS:
        # 检查模型名称是否在支持的模型列表中
        raise ValueError(f"The provided model {model.name} is not supported.")

    if use_lm:
        # 获取语言模型
        lm = model.get_lm_model()

    with torch.no_grad():
        # 对波形进行编码
        frames = model.encode(wav[None])
    
    # 构建元数据字典
    metadata = {
        'm': model.name,                 # 模型名称
        'al': wav.shape[-1],             # 音频长度
        'nc': frames[0][0].shape[1],     # 码书数量
        'lm': use_lm,                    # 是否使用语言模型
    }
    # 写入头部信息到文件对象
    binary.write_ecdc_header(fo, metadata)

    # 遍历每个编码帧和缩放因子
    for (frame, scale) in frames:
        if scale is not None:
            # 如果有缩放因子，则写入到文件对象
            fo.write(struct.pack('!f', scale.cpu().item()))
        # 获取帧的形状
        _, K, T = frame.shape
        if use_lm:
            coder = ArithmeticCoder(fo) # 初始化算术编码器
            states: tp.Any = None # 初始化状态
            offset = 0 # 初始化偏移量
            input_ = torch.zeros(1, K, 1, dtype=torch.long, device=wav.device) # 初始化输入张量
        else:
            # 初始化比特打包器
            packer = binary.BitPacker(model.bits_per_codebook, fo)

        # 遍历时间步
        for t in range(T):
            if use_lm:
                with torch.no_grad():
                    # 使用语言模型计算概率
                    probas, states, offset = lm(input_, states, offset)
                # We emulate a streaming scenario even though we do not provide an API for it.
                # This gives us a more accurate benchmark.
                # 模拟流式场景，即使我们没有提供API。这为我们提供了一个更准确的基准。
                input_ = 1 + frame[:, :, t: t + 1]
            # 遍历每个码字
            for k, value in enumerate(frame[0, :, t].tolist()):
                if use_lm:
                    # 构建稳定的量化CDF
                    q_cdf = build_stable_quantized_cdf(
                        probas[0, :, k, 0], coder.total_range_bits, check=False)
                    # 将值推入算术编码器
                    coder.push(value, q_cdf)
                else:
                    # 将值推入比特打包器
                    packer.push(value)
        if use_lm:
            # 刷新算术编码器
            coder.flush()
        else:
            # 刷新比特打包器
            packer.flush()


def decompress_from_file(fo: tp.IO[bytes], device='cpu') -> tp.Tuple[torch.Tensor, int]:
    """Decompress from a file-object.
    Returns a tuple `(wav, sample_rate)`.

    Args:
        fo (IO[bytes]): file-object from which to read. If you want to decompress
            from `bytes` instead, see `decompress`.
        device: device to use to perform the computations.
    """
    """
    从文件对象中解压缩音频数据。
    返回一个元组 `(wav, sample_rate)`。

    Args:
        fo (IO[bytes]): 要从中读取数据的文件对象。如果需要从 `bytes` 解压，请参见 `decompress` 函数。
        device: 用于执行计算的计算设备。

    Returns:
        Tuple[torch.Tensor, int]: 解压后的波形张量和采样率。
    """
    # 从文件对象中读取头部信息
    metadata = binary.read_ecdc_header(fo)

    # 从元数据中提取模型名称、音频长度、码书数量和是否使用语言模型
    model_name = metadata['m']
    audio_length = metadata['al']
    num_codebooks = metadata['nc']
    use_lm = metadata['lm']

    # 确保音频长度和码书数量为整数类型
    assert isinstance(audio_length, int)
    assert isinstance(num_codebooks, int)

    # 检查模型名称是否在支持的模型列表中
    if model_name not in MODELS:
        raise ValueError(f"The audio was compressed with an unsupported model {model_name}.")
    
    # 实例化模型并移动到指定设备
    model = MODELS[model_name]().to(device)

    # 如果使用语言模型，则获取语言模型
    if use_lm:
        lm = model.get_lm_model()

    # 初始化帧列表
    frames: tp.List[EncodedFrame] = []

    # 获取片段长度和步幅，如果未定义，则默认使用音频长度
    segment_length = model.segment_length or audio_length
    segment_stride = model.segment_stride or audio_length

    # 遍历每个片段
    for offset in range(0, audio_length, segment_stride):
        # 计算当前片段的长度
        this_segment_length = min(audio_length - offset, segment_length)
        # 计算当前片段的帧数
        frame_length = int(math.ceil(this_segment_length * model.frame_rate / model.sample_rate))
        if model.normalize:
            # 如果模型需要归一化，则读取缩放因子
            scale_f, = struct.unpack('!f', binary._read_exactly(fo, struct.calcsize('!f')))
            scale = torch.tensor(scale_f, device=device).view(1)
        else:
            # 否则，缩放因子为 None
            scale = None
        if use_lm:
            # 如果使用语言模型，则初始化算术解码器、状态和输入
            decoder = ArithmeticDecoder(fo)
            states: tp.Any = None
            offset = 0
            input_ = torch.zeros(1, num_codebooks, 1, dtype=torch.long, device=device)
        else:
            # 否则，初始化比特解包器
            unpacker = binary.BitUnpacker(model.bits_per_codebook, fo)
        
        # 初始化帧张量
        frame = torch.zeros(1, num_codebooks, frame_length, dtype=torch.long, device=device)
        # 遍历每个时间步
        for t in range(frame_length):
            if use_lm:
                with torch.no_grad():
                    # 使用语言模型计算概率
                    probas, states, offset = lm(input_, states, offset)
            code_list: tp.List[int] = []
            for k in range(num_codebooks):
                if use_lm:
                    # 如果使用语言模型，则构建稳定的量化CDF并解码
                    q_cdf = build_stable_quantized_cdf(
                        probas[0, :, k, 0], decoder.total_range_bits, check=False)
                    code = decoder.pull(q_cdf)
                else:
                    # 否则，直接从比特解包器中解码
                    code = unpacker.pull()
                if code is None:
                    raise EOFError("The stream ended sooner than expected.")
                code_list.append(code)
            # 将解码后的码字转换为张量并赋值给当前帧
            codes = torch.tensor(code_list, dtype=torch.long, device=device)
            frame[0, :, t] = codes
            if use_lm:
                input_ = 1 + frame[:, :, t: t + 1]
        # 将当前帧和缩放因子添加到帧列表中
        frames.append((frame, scale))
    with torch.no_grad():
        wav = model.decode(frames)
    # 返回解压后的波形和采样率
    return wav[0, :, :audio_length], model.sample_rate


def compress(model: EncodecModel, wav: torch.Tensor, use_lm: bool = False) -> bytes:
    """Compress a waveform using the given model. Returns the compressed bytes.

    Args:
        model (EncodecModel): a pre-trained EncodecModel to use to compress the audio.
        wav (torch.Tensor): waveform to compress, should have a shape `[C, T]`, with `C`
            matching `model.channels`, and the proper sample rate (e.g. `model.sample_rate`).
            Use `utils.convert_audio` if this is not the case.
        use_lm (bool): if True, use a pre-trained language model to further
            compress the stream using Entropy Coding. This will slow down compression
            quite a bit, expect between 20 to 30% of size reduction.
    """
    """
    使用给定的模型压缩波形。返回压缩后的字节。

    Args:
        model (EncodecModel): 预训练的Encodec模型，用于压缩音频。
        wav (torch.Tensor): 要压缩的波形张量，形状应为 `[C, T]`，其中 `C` 应与 `model.channels` 匹配，且采样率正确（例如 `model.sample_rate`）。
            如果不是这种情况，请使用 `utils.convert_audio` 进行转换。
        use_lm (bool, optional): 如果为True，则使用预训练的语言模型通过熵编码进一步压缩流。这会显著减慢压缩速度，但预计可以减少20%到30%的体积。默认为False。

    Returns:
        bytes: 压缩后的字节数据。
    """
    # 创建一个内存中的字节流对象
    fo = io.BytesIO()
    # 使用 compress_to_file 函数将波形数据压缩到字节流对象中
    compress_to_file(model, wav, fo, use_lm=use_lm)
    # 获取字节流中的字节数据并返回
    return fo.getvalue()


def decompress(compressed: bytes, device='cpu') -> tp.Tuple[torch.Tensor, int]:
    """Decompress from a file-object.
    Returns a tuple `(wav, sample_rate)`.

    Args:
        compressed (bytes): compressed bytes.
        device: device to use to perform the computations.
    """
    """
    从压缩的字节数据中解压缩音频数据。
    返回一个元组 `(wav, sample_rate)`。

    Args:
        compressed (bytes): 压缩后的字节数据。
        device (str, optional): 用于执行计算的计算设备。默认为 'cpu'。

    Returns:
        Tuple[torch.Tensor, int]: 解压后的波形张量和采样率。
    """
    # 创建一个内存中的字节流对象，并将压缩后的字节数据写入其中
    fo = io.BytesIO(compressed)
    # 使用 decompress_from_file 函数从字节流对象中解压缩音频数据
    return decompress_from_file(fo, device=device)


def test():
    """
    测试函数，用于验证不同模型和是否使用语言模型（LM）对音频压缩和解压的效果。

    测试流程：
    1. 遍历所有预定义的模型。
    2. 加载对应的测试音频文件。
    3. 设置目标带宽为12。
    4. 对于每个模型，分别测试是否使用语言模型（LM）的压缩和解压性能。
    5. 打印压缩和解压的时间以及压缩后的比特率（kbps）。
    6. 解压后的音频形状与原始音频形状一致。
    """
    # 设置PyTorch的线程数为1，以避免多线程干扰测试结果
    import torchaudio
    torch.set_num_threads(1)

    # 遍历所有预定义的模型名称
    for name in MODELS.keys():
        # 实例化模型
        model = MODELS[name]()
        # 计算采样率（kHz）
        sr = model.sample_rate // 1000
        # 加载测试音频文件，文件名格式为 'test_24k.wav' 或 'test_48k.wav'
        x, _ = torchaudio.load(f'test_{sr}k.wav')
        # 截取音频样本，确保其长度不超过模型采样率的5倍
        x = x[:, :model.sample_rate * 5]
        # 设置模型的目标带宽为12
        model.set_target_bandwidth(12)

        # 遍历是否使用语言模型（LM）的两种情况
        for use_lm in [False, True]:
            print(f"Doing {name}, use_lm={use_lm}")
            # 记录开始时间
            begin = time.time()
            # 对音频进行压缩
            res = compress(model, x, use_lm=use_lm)
            # 计算压缩时间
            t_comp = time.time() - begin
            # 对压缩后的数据进行解压
            x_dec, _ = decompress(res)
            # 计算解压时间
            t_decomp = time.time() - begin - t_comp
            # 计算压缩后的比特率（kbps）
            kbps = 8 * len(res) / 1000 / (x.shape[-1] / model.sample_rate)
            # 输出压缩和解压的性能指标
            print(f"kbps: {kbps:.1f}, time comp: {t_comp:.1f} sec. "
                  f"time decomp:{t_decomp:.1f}.")
            # 确保解压后的音频形状与原始音频形状一致
            assert x_dec.shape == x.shape


if __name__ == '__main__':
    test()
