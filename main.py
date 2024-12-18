import argparse
from pathlib import Path
import sys
import torchaudio

from compress import compress, decompress, MODELS
from utils import save_audio, convert_audio


# 定义文件后缀
SUFFIX = '.ecdc'


def get_parser():
    """
    创建命令行参数解析器。

    返回:
        argparse.ArgumentParser: 配置好的命令行参数解析器。
    """
    parser = argparse.ArgumentParser(
        'encodec',
        description='High fidelity neural audio codec. '
                    'If input is a .ecdc, decompresses it. '
                    'If input is .wav, compresses it. If output is also wav, '
                    'do a compression/decompression cycle.')
    # 添加输入文件参数
    parser.add_argument(
        'input', type=Path,
        help='Input file, whatever is supported by torchaudio on your system.')
    # 添加输出文件参数（可选）
    parser.add_argument(
        'output', type=Path, nargs='?',
        help='Output file, otherwise inferred from input file.')
    # 添加带宽参数
    parser.add_argument(
        '-b', '--bandwidth', type=float, default=6, choices=[1.5, 3., 6., 12., 24.],
        help='Target bandwidth (1.5, 3, 6, 12 or 24). 1.5 is not supported with --hq.')
    # 添加高质量模式参数
    parser.add_argument(
        '-q', '--hq', action='store_true',
        help='Use HQ stereo model operating on 48 kHz sampled audio.')
    # 添加语言模型参数
    parser.add_argument(
        '-l', '--lm', action='store_true',
        help='Use a language model to reduce the model size (5x slower though).')
    # 添加覆盖输出文件参数
    parser.add_argument(
        '-f', '--force', action='store_true',
        help='Overwrite output file if it exists.')
    # 添加解压缩后缀参数
    parser.add_argument(
        '-s', '--decompress_suffix', type=str, default='_decompressed',
        help='Suffix for the decompressed output file (if no output path specified)')
    # 添加自动缩放参数
    parser.add_argument(
        '-r', '--rescale', action='store_true',
        help='Automatically rescale the output to avoid clipping.')
    return parser


def fatal(*args):
    """
    打印错误信息到标准错误并退出程序。

    Args:
        *args: 可变数量的位置参数，打印为错误信息。
    """
    print(*args, file=sys.stderr)
    sys.exit(1)


def check_output_exists(args):
    """
    检查输出路径是否存在。如果输出目录不存在，则终止程序。
    如果输出文件已存在且未使用 -f / --force 参数，则终止程序。

    Args:
        args: 解析后的命令行参数。
    """
    if not args.output.parent.exists():
        fatal(f"Output folder for {args.output} does not exist.")
    if args.output.exists() and not args.force:
        fatal(f"Output file {args.output} exist. Use -f / --force to overwrite.")


def check_clipping(wav, args):
    """
    检查音频是否发生削波。如果音频的最大绝对值超过0.99，则发出警告。
    如果使用了 -r / --rescale 参数，则不进行此检查。

    Args:
        wav (torch.Tensor): 要检查的音频张量。
        args: 解析后的命令行参数。
    """
    if args.rescale:
        return
    mx = wav.abs().max()
    limit = 0.99
    if mx > limit:
        print(
            f"Clipping!! max scale {mx}, limit is {limit}. "
            "To avoid clipping, use the `-r` option to rescale the output.",
            file=sys.stderr)


def main():
    """
    主函数，执行压缩或解压缩操作。

    流程：
    1. 解析命令行参数。
    2. 检查输入文件是否存在。
    3. 根据输入文件的后缀决定执行压缩还是解压缩。
    4. 如果是解压缩：
        - 如果未指定输出文件，则生成默认的输出文件名。
        - 检查输出路径是否存在。
        - 解压缩输入文件。
        - 检查是否发生削波。
        - 保存解压后的音频。
    5. 如果是压缩：
        - 如果未指定输出文件，则生成默认的输出文件名。
        - 检查输出路径是否存在。
        - 加载并转换音频文件。
        - 压缩音频。
        - 如果输出文件后缀为 .ecdc，则直接保存压缩后的数据。
        - 如果输出文件后缀为 .wav，则解压缩并保存音频。
    """
    # 解析命令行参数
    args = get_parser().parse_args()
    # 检查输入文件是否存在
    if not args.input.exists():
        fatal(f"Input file {args.input} does not exist.")

    # 判断输入文件的后缀是否为 .ecdc，如果是，则执行解压缩
    if args.input.suffix.lower() == SUFFIX:
        # Decompression
        if args.output is None:
            # 如果未指定输出文件，则生成默认的输出文件名
            args.output = args.input.with_name(args.input.stem + args.decompress_suffix).with_suffix('.wav')
        elif args.output.suffix.lower() != '.wav':
            fatal("Output extension must be .wav")
        # 检查输出路径是否存在
        check_output_exists(args)
        # 解压缩输入文件
        out, out_sample_rate = decompress(args.input.read_bytes())
        # 检查是否发生削波
        check_clipping(out, args)
        # 保存解压后的音频
        save_audio(out, args.output, out_sample_rate, rescale=args.rescale)
    else:
        # Compression
        if args.output is None:
            # 如果未指定输出文件，则生成默认的输出文件名
            args.output = args.input.with_suffix(SUFFIX)
        elif args.output.suffix.lower() not in [SUFFIX, '.wav']:
            # 如果指定了输出文件但后缀不是 .wav 或 .ecdc，则终止程序
            fatal(f"Output extension must be .wav or {SUFFIX}")
        # 检查输出路径是否存在
        check_output_exists(args)

        # 选择模型
        model_name = 'encodec_48khz' if args.hq else 'encodec_24khz'
        # 加载模型
        model = MODELS[model_name]()
        # 检查带宽是否被模型支持
        if args.bandwidth not in model.target_bandwidths:
            fatal(f"Bandwidth {args.bandwidth} is not supported by the model {model_name}")
        # 设置模型的目标带宽
        model.set_target_bandwidth(args.bandwidth)

        # 加载并转换音频
        wav, sr = torchaudio.load(args.input)
        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        # 压缩音频
        compressed = compress(model, wav, use_lm=args.lm)
        # 根据输出文件的后缀决定保存方式
        if args.output.suffix.lower() == SUFFIX:
            args.output.write_bytes(compressed)
        else:
            # Directly run decompression stage
            # 如果输出文件后缀为 .wav，则解压缩并保存音频
            assert args.output.suffix.lower() == '.wav'
            out, out_sample_rate = decompress(compressed)
            check_clipping(out, args)
            save_audio(out, args.output, out_sample_rate, rescale=args.rescale)


if __name__ == '__main__':
    main()
