from struct import pack, unpack


_UINT_16_SIZE = 2
_UINT_32_SIZE = 4


class Encoder:
    _data: bytearray

    def __init__(self):
        self._data = bytearray()

    def write_uint16(self, data: int):
        self._data.extend(int.to_bytes(data, _UINT_16_SIZE, "big"))

    def write_uint32(self, data: int):
        self._data.extend(int.to_bytes(data, _UINT_32_SIZE, "big"))

    def write_float(self, data: float):
        self.write_uint32(unpack(">L", pack(">f", data))[0])

    def write_str(self, data: str):
        data = data.encode()

        self._data.append(len(data))
        self._data.extend(data)

    def write_sized_str(self, data: str):
        self._data.extend(data.encode())

    def finish(self):
        if len(self._data) > 128:
            raise RuntimeError(
                f"Encoder with data {self._data} is too large({len(self._data)}, can not exceed 128 bytes."
            )

        return bytes(self._data)


class Decoder:
    _position: int
    _data: bytes

    def __init__(self, data: bytes):
        self._position = 0
        self._data = data

    def read_uint16(self):
        value = int.from_bytes(self._data[self._position:self._position + _UINT_16_SIZE], "big")
        self._position += _UINT_16_SIZE

        return value

    def read_uint32(self):
        value = int.from_bytes(self._data[self._position:self._position + _UINT_32_SIZE], "big")
        self._position += _UINT_32_SIZE

        return value

    def read_float(self) -> float:
        return unpack(">f", pack(">L", self.read_uint32()))[0]

    def read_str(self):
        length = self._data[self._position]
        self._position += 1

        value = self._data[self._position:self._position + length]
        self._position += length

        return value.decode()

    def read_sized_str(self, length: int):
        value = self._data[self._position:self._position + length]
        self._position += length

        return value.decode()

    @property
    def eof(self):
        return self._position >= len(self._data)
