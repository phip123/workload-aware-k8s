from dataclasses import dataclass


@dataclass
class ResourceUsage:
    cpu: float
    blkio: float
    net: float
    gpu: float
    ram: float
