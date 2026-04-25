import struct 

class Float32:
    def __init__(self, _input: float) -> None:
        self._input = _input
        bits = struct.unpack('>I', struct.pack('>f', _input))[0]
        self.sign = bits >> 31
        self.exp = (bits >> 23) & 0xFF
        self.mantissa = bits & 0x7FFFFF

    def __repr__(self) -> str:
        return (    
            f"float={self._input} sign={self.sign} | exp={self.exp} ({self.exp - 127:+d}) | mantissa={self.mantissa:023b}"
        )
    
    def to_float(self) -> float:
        return (-1)**self.sign * 2**(self.exp - 127) * (1 + self.mantissa / 2**23)


class Float16:
    def __init__(self, _input: float) -> None:
        self._input = _input
        bits = struct.unpack('>I', struct.pack('>f', _input))[0]
        self.sign = bits >> 16