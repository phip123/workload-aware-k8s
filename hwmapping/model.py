from enum import Enum

"""
Bins:    |    LOW     |    MEDIUM    |     HIGH     | VERY_HIGH

Cores:   | 1-2        |   4 - 8      |  16 - 32     | > 32
RAM:     | 1-2        |   4 - 8      |  16 - 32     | > 32
CpuMhz:  | <= 1.5     |   1.6 - 2.2  |     < 3.5    | > 3.5
GpuMHz:  | <= 1000    |   <= 1200    |  <= 1500     | > 1700
VRAM:    | <= 2       |   4 - 8      |   < 32       | > 32    
Network: | <= 150Mbps | <= 500 Mbps  | <=1 Gbit     | >= 10 Gbit 
"""

class Bins(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4


class Location(Enum):
    CLOUD = 1
    EDGE = 2
    MEC = 3
    MOBILE = 4


class Disk(Enum):
    HDD = 1
    SSD = 2
    NVME = 3
    FLASH = 4
    SD = 5


class Accelerator(Enum):
    NONE = 1
    GPU = 2
    TPU = 3


class Connection(Enum):
    MOBILE = 1
    WIFI = 2
    ETHERNET = 3


class Arch(Enum):
    ARM32 = 1
    X86 = 2
    AARCH64 = 3


class GpuModel(Enum):
    TURING = 1
    PASCAL = 2
    MAXWELL = 3
    VOLTA = 4


class CpuModel(Enum):
    I7 = 1
    XEON = 2
    ARM = 3
