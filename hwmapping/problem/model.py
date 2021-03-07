from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

from hwmapping.cluster import State, Cluster
from hwmapping.device import Device
from hwmapping.evaluation.fetdistributions import min_max_execution_times
from hwmapping.model import Bins, Connection, Location, Accelerator, Arch, GpuModel, CpuModel, Disk


@dataclass
class LabelProblemSolverSettings:
    name: str
    reconcile_interval: int
    settings: Dict
    parallel: bool = False
    workers: int = 1


@dataclass
class ProblemInstance:
    cluster: Cluster  # the clusters to look for requirements
    state: State
    performance_weight: float = 1  # weights performance, lower values (<1) increase the impact
    variety_weight: float = 1  # weights variety, lower values (<1) increase the impact
    filtered_devices: List[
        Device] = list  # the GA will first filter out all devices that are not able to run at least one function of the cluster
    consider_all: bool = False

    def scale(self, value: float, service: str):
        amin, amax = min_max_execution_times[service]
        if amin == amax:
            return 1
        a = (value - amin) / (amax - amin)
        return 1 - a


@dataclass
class Requirements:
    arch: Dict[Arch, float]
    accelerator: Dict[Accelerator, float]
    cores: Dict[Bins, float]
    disk: Dict[Disk, float]
    location: Dict[Location, float]
    connection: Dict[Connection, float]
    network: Dict[Bins, float]
    cpu_mhz: Dict[Bins, float]
    cpu: Dict[CpuModel, float]
    ram: Dict[Bins, float]
    gpu_vram: Dict[Bins, float]
    gpu_mhz: Dict[Bins, float]
    gpu_model: Dict[GpuModel, float]

    def __str__(self):
        def join(d: Dict) -> str:
            return "\n".join(['%s:: %s' % (key, value) for (key, value) in d.items()])

        text = "---------------------------"
        for name, c in self.characteristics:
            text += f'\n--------{name}---------\n'
            text += join(c)
            text += '\n'
        return text

    def __map(self, d: Dict[Enum, float]) -> Dict[str, float]:
        data = {}
        for k, v in d.items():
            data[f'{str(k.name)}'] = v
        return data

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            'device.edgerun.io/arch': self.__map(self.arch),
            'device.edgerun.io/accelerator': self.__map(self.accelerator),
            'device.edgerun.io/cores': self.__map(self.cores),
            'device.edgerun.io/disk': self.__map(self.disk),
            'device.edgerun.io/location': self.__map(self.location),
            'device.edgerun.io/connection': self.__map(self.connection),
            'device.edgerun.io/network': self.__map(self.network),
            'device.edgerun.io/cpu_mhz': self.__map(self.cpu_mhz),
            'device.edgerun.io/cpu': self.__map(self.cpu),
            'device.edgerun.io/ram': self.__map(self.ram),
            'device.edgerun.io/vram_bin': self.__map(self.gpu_vram),
            'device.edgerun.io/gpu_mhz': self.__map(self.gpu_mhz),
            'device.edgerun.io/gpu_model': self.__map(self.gpu_model),
        }

    @property
    def characteristics(self):
        return [
            (Arch, self.arch),
            (Accelerator, self.accelerator),
            (Bins, self.cores),
            (Disk, self.disk),
            (Location, self.location),
            (Connection, self.connection),
            (Bins, self.network),
            (Bins, self.cpu_mhz),
            (Bins, self.cpu),
            (Bins, self.ram),
            (Bins, self.gpu_vram),
            (Bins, self.gpu_mhz),
            (GpuModel, self.gpu_model)
        ]

    @staticmethod
    def fields():
        return [
            ('arch', Arch),
            ('accelerator', Accelerator),
            ('cores', Bins),
            ('disk', Disk),
            ('location', Location),
            ('connection', Connection),
            ('network', Bins),
            ('cpu_mhz', Bins),
            ('cpu', CpuModel),
            ('ram', Bins),
            ('gpu_vram', Bins),
            ('gpu_mhz', Bins),
            ('gpu_model', GpuModel)
        ]


@dataclass
class Solution:
    device: List[Tuple[bool, Tuple[Device, float]]]


@dataclass
class Result:
    instance: ProblemInstance
    duration: float
    requirements: Requirements
    devices: List[Device]
    mean_fet: float
    successful: bool
    # includes the best set of variables found and the value of the given function associated to it
    output_dict: Dict
    # a list including the convergence of the algorithm over iterations
    report: List[float]
    ga_run_settings: Dict
    algorithm_params: Dict

    def __str__(self):
        return "Mean FET over all devices: %s" % self.mean_fet
