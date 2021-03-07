from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np

from hwmapping.device import Device
from hwmapping.faas.system import FunctionDefinition
from hwmapping.resource import ResourceUsage


@dataclass
class Cluster:
    id: str
    functions: Dict[str, FunctionDefinition]  # key is pod name

    @property
    def function_definitions(self):
        return list(self.functions.values())

    def avg_resource_usage(self, hosts: List[str]) -> ResourceUsage:
        usages = []

        for x in self.function_definitions:
            for host in hosts:
                usages.append(x.get_resources_for_node(host))
        total_count = len(usages)
        cpu_sum = np.sum([usage.cpu for usage in usages])
        net_sum = np.sum([usage.net for usage in usages])
        blkio_sum = np.sum([usage.blkio for usage in usages])
        gpu_sum = np.sum([usage.blkio for usage in usages])
        ram_sum = np.sum([usage.blkio for usage in usages])
        return ResourceUsage(
            cpu=cpu_sum / total_count,
            blkio=blkio_sum / total_count,
            ram=ram_sum / total_count,
            net=net_sum / total_count,
            gpu=gpu_sum / total_count
        )

    def get_function(self, name: str) -> Optional[FunctionDefinition]:
        return self.functions.get(name, None)

    def contains_function(self, name: str) -> bool:
        return self.get_function(name) is not None


class State:

    def __init__(self, devices: List[Device], clusters: Dict[str, Cluster] = None):
        self.devices = devices
        self.clusters = clusters or {}

    def get_cluster_for_pod(self, fn: FunctionDefinition) -> Optional[Cluster]:
        assigned_cluster = [x for x in self.clusters.values() if x.contains_function(fn.name)]
        return assigned_cluster[0] if len(assigned_cluster) == 1 else None
