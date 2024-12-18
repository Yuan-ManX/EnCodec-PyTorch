import io
import json
import struct
import typing as tp


# 定义结构体格式，用于打包头部信息
# 格式为：'!4sBI' 表示：
#   - '!4s'：4字节的固定魔术码（网络字节序，大端）
#   - 'B'：1字节的无符号整数（协议版本）
#   - 'I'：4字节的无符号整数（头部大小）
_encodec_header_struct = struct.Struct('!4sBI')

# 定义魔术码，用于标识ECDC格式
_ENCODEC_MAGIC = b'ECDC'


def write_ecdc_header(fo: tp.IO[bytes], metadata: tp.Any):
    """
    将ECDC格式的头部信息写入到文件对象中。

    头部信息格式如下：
    - 魔术码（4字节）：固定为 'ECDC'，用于标识ECDC格式。
    - 协议版本（1字节）：当前版本为0。
    - 头部大小（4字节）：后续JSON头部信息的长度（以字节为单位）。
    - JSON头部信息：包含解码所需的所有信息。
    - 原始字节流：需要根据JSON头部信息进行解释。

    Args:
        fo (IO[bytes]): 写入头部信息的文件对象。
        metadata (Any): 需要序列化为JSON的元数据。
    """
    # 将元数据序列化为JSON格式，并编码为UTF-8字节
    meta_dumped = json.dumps(metadata).encode('utf-8')
    # 定义协议版本
    version = 0
    # 使用预定义的结构体格式打包魔术码、协议版本和头部大小
    header = _encodec_header_struct.pack(_ENCODEC_MAGIC, version, len(meta_dumped))
    # 将打包好的头部信息写入文件对象
    fo.write(header)
    # 将序列化的元数据写入文件对象
    fo.write(meta_dumped)
    # 刷新文件对象，确保所有数据都被写入
    fo.flush()


def _read_exactly(fo: tp.IO[bytes], size: int) -> bytes:
    """
    从文件对象中精确读取指定数量的字节。

    如果在读取过程中遇到文件结束（EOF），则抛出EOFError异常。

    Args:
        fo (IO[bytes]): 要读取数据的文件对象。
        size (int): 要读取的字节数。

    Returns:
        bytes: 读取到的字节数据。

    Raises:
        EOFError: 如果在读取到指定数量的字节之前遇到文件结束。
    """
    # 初始化缓冲区为空字节串
    buf = b""
    while len(buf) < size:
        # 从文件对象中读取最多'size'字节的数据
        new_buf = fo.read(size)
        if not new_buf:
            # 如果没有读取到任何数据，抛出EOFError异常
            raise EOFError("Impossible to read enough data from the stream, "
                           f"{size} bytes remaining.")
        # 将新读取的数据添加到缓冲区
        buf += new_buf
        # 更新剩余需要读取的字节数
        size -= len(new_buf)
    # 返回读取到的字节数据
    return buf


def read_ecdc_header(fo: tp.IO[bytes]):
    """
    从文件对象中读取ECDC格式的头部信息。

    头部信息格式如下：
    - 魔术码（4字节）：固定为 'ECDC'，用于标识ECDC格式。
    - 协议版本（1字节）：当前版本为0。
    - 头部大小（4字节）：后续JSON头部信息的长度（以字节为单位）。
    - JSON头部信息：包含解码所需的所有信息。

    Args:
        fo (IO[bytes]): 要读取头部信息的文件对象。

    Returns:
        dict: 解析后的JSON头部信息。

    Raises:
        ValueError: 如果文件不是ECDC格式或版本不支持。
        EOFError: 如果在读取到指定数量的字节之前遇到文件结束。
    """
    # 从文件对象中读取固定大小的头部字节（魔术码、协议版本和头部大小）
    header_bytes = _read_exactly(fo, _encodec_header_struct.size)
    # 使用预定义的结构体格式解包头部字节
    magic, version, meta_size = _encodec_header_struct.unpack(header_bytes)
    # 检查魔术码是否为 'ECDC'
    if magic != _ENCODEC_MAGIC:
        raise ValueError("File is not in ECDC format.")
    # 检查协议版本是否为0
    if version != 0:
        raise ValueError("Version not supported.")
    # 从文件对象中读取JSON头部信息的字节数据
    meta_bytes = _read_exactly(fo, meta_size)
    # 将字节数据解码为UTF-8字符串并解析为JSON对象
    return json.loads(meta_bytes.decode('utf-8'))


class BitPacker:
    """Simple bit packer to handle ints with a non standard width, e.g. 10 bits.
    Note that for some bandwidth (1.5, 3), the codebook representation
    will not cover an integer number of bytes.

    Args:
        bits (int): number of bits per value that will be pushed.
        fo (IO[bytes]): file-object to push the bytes to.
    """
    """
    简单的位打包器，用于处理非标准宽度的整数，例如10位。
    注意，对于某些带宽（1.5, 3），码书的表示可能不会覆盖整数个字节。

    Args:
        bits (int): 每个要推送的值所占用的位数。
        fo (IO[bytes]): 用于推送字节的文件对象。
    """
    def __init__(self, bits: int, fo: tp.IO[bytes]):
        """
        初始化BitPacker实例。

        Args:
            bits (int): 每个要推送的值所占用的位数。
            fo (IO[bytes]): 用于推送字节的文件对象。
        """
        # 当前累积的值，初始为0
        self._current_value = 0
        # 当前累积的位数，初始为0
        self._current_bits = 0
        # 每个值的位数
        self.bits = bits
        # 文件对象，用于写入字节
        self.fo = fo

    def push(self, value: int):
        """Push a new value to the stream. This will immediately
        write as many uint8 as possible to the underlying file-object."""
        """
        将一个新值推送到流中。这将立即尽可能多地将uint8写入底层文件对象。

        Args:
            value (int): 要推送的整数值。
        """
        # 将新值左移当前累积的位数，并加到当前累积值上
        self._current_value += (value << self._current_bits)
        # 增加累积的位数
        self._current_bits += self.bits
        # 当累积的位数达到或超过8位时，写入一个完整的字节
        while self._current_bits >= 8:
            # 获取当前累积值的最低8位
            lower_8bits = self._current_value & 0xff
            # 减少累积的位数
            self._current_bits -= 8
            # 将当前累积值右移8位，准备写入下一个字节
            self._current_value >>= 8
            # 将最低8位写入文件对象
            self.fo.write(bytes([lower_8bits]))

    def flush(self):
        """Flushes the remaining partial uint8, call this at the end
        of the stream to encode."""
        """
        刷新剩余的部分uint8，在流的末尾调用此方法以完成编码。
        """
        if self._current_bits:
            # 如果还有剩余的位数，则将当前累积值写入文件对象
            self.fo.write(bytes([self._current_value]))
            # 重置当前累积值
            self._current_value = 0
            # 重置当前累积位数
            self._current_bits = 0
        # 刷新文件对象，确保所有数据都被写入
        self.fo.flush()


class BitUnpacker:
    """BitUnpacker does the opposite of `BitPacker`.

    Args:
        bits (int): number of bits of the values to decode.
        fo (IO[bytes]): file-object to push the bytes to.
        """
    """
    BitUnpacker 是 `BitPacker` 的反向操作。

    Args:
        bits (int): 要解码的值的位数。
        fo (IO[bytes]): 用于读取字节的文件对象。
    """
    def __init__(self, bits: int, fo: tp.IO[bytes]):
        """
        初始化BitUnpacker实例。

        Args:
            bits (int): 要解码的值的位数。
            fo (IO[bytes]): 用于读取字节的文件对象。
        """
        # 每个值的位数
        self.bits = bits
        # 文件对象，用于读取字节
        self.fo = fo
        # 用于提取当前值的掩码
        self._mask = (1 << bits) - 1
        # 当前累积的值，初始为0
        self._current_value = 0
        # 当前累积的位数，初始为0
        self._current_bits = 0

    def pull(self) -> tp.Optional[int]:
        """
        Pull a single value from the stream, potentially reading some
        extra bytes from the underlying file-object.
        Returns `None` when reaching the end of the stream.
        """
        """
        从流中提取单个值，可能会从底层文件对象中读取一些额外的字节。
        当到达流的末尾时返回 `None`。

        Returns:
            Optional[int]: 提取的整数值，如果没有更多的数据则返回 `None`。
        """
        while self._current_bits < self.bits:
            # 从文件对象中读取1个字节
            buf = self.fo.read(1)
            if not buf:
                # 如果没有读取到任何数据，返回 `None`
                return None
            # 获取读取到的字节
            character = buf[0]
            # 将字节左移当前累积的位数，并加到当前累积值上
            self._current_value += character << self._current_bits
            # 增加累积的位数
            self._current_bits += 8

        # 使用掩码提取当前值
        out = self._current_value & self._mask
        # 将当前累积值右移 `bits` 位，准备提取下一个值
        self._current_value >>= self.bits
        # 减少累积的位数
        self._current_bits -= self.bits
        # 返回提取的值
        return out


def test():
    """
    测试函数，用于验证BitPacker和BitUnpacker的正确性。

    测试流程：
    1. 设置随机种子以确保可重复性。
    2. 进行多次重复测试。
    3. 对于每次重复：
        - 生成随机长度和位数。
        - 生成随机整数列表作为原始令牌。
        - 使用BitPacker将令牌打包到字节缓冲区中。
        - 使用BitUnpacker从字节缓冲区中解包令牌。
        - 断言解包后的令牌列表长度与原始令牌列表长度一致。
        - 断言解包后的令牌列表长度不超过原始令牌列表长度加上可能的额外值。
        - 断言每个解包后的令牌与原始令牌一致。
    """
    # 设置随机种子以确保可重复性
    import torch
    torch.manual_seed(1234)
    # 进行4次重复测试
    for rep in range(4):
        # 生成随机长度，范围在10到1999之间
        length: int = torch.randint(10, 2_000, (1,)).item()
        # 生成随机位数，范围在1到15之间
        bits: int = torch.randint(1, 16, (1,)).item()
        # 生成随机整数列表作为原始令牌
        tokens: tp.List[int] = torch.randint(2 ** bits, (length,)).tolist()
        # 初始化重建的令牌列表
        rebuilt: tp.List[int] = []
        # 创建一个内存中的字节缓冲区
        buf = io.BytesIO()
        # 初始化BitPacker
        packer = BitPacker(bits, buf)
        for token in tokens:
            # 将每个令牌推送到BitPacker中
            packer.push(token)
        # 刷新BitPacker，确保所有数据都被写入
        packer.flush()
        # 将字节缓冲区的指针重置到开头
        buf.seek(0)
        # 初始化BitUnpacker
        unpacker = BitUnpacker(bits, buf)
        while True:
            # 从BitUnpacker中提取值
            value = unpacker.pull()
            if value is None:
                # 如果没有更多的数据，则退出循环
                break
            # 将提取的值添加到重建的令牌列表中
            rebuilt.append(value)
        # 确保重建的令牌列表长度不小于原始令牌列表长度
        assert len(rebuilt) >= len(tokens), (len(rebuilt), len(tokens))
        # The flushing mechanism might lead to "ghost" values at the end of the stream.
        # 确保重建的令牌列表长度不超过原始令牌列表长度加上可能的额外值
        assert len(rebuilt) <= len(tokens) + 8 // bits, (len(rebuilt), len(tokens), bits)
        for idx, (a, b) in enumerate(zip(tokens, rebuilt)):
            # 确保每个解包后的令牌与原始令牌一致
            assert a == b, (idx, a, b)


if __name__ == '__main__':
    test()
