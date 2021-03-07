from typing import List

from ether.cell import LANCell
from ether.core import Node
from sim.topology import Topology
from skippy.core.storage import StorageIndex

from hwmapping.evaluation.scenarios import HeterogeneousUrbanSensingScenario


def all_internet_topology(nodes: List[Node]) -> Topology:
    t = Topology()
    for node in nodes:
        cell = LANCell(nodes=[node], backhaul='internet')
        t.add(cell)
    t.init_docker_registry()

    return t


def urban_sensing_topology(nodes: List[Node], storage_index: StorageIndex) -> Topology:
    t = Topology()
    HeterogeneousUrbanSensingScenario(nodes, storage_index).materialize(t)
    t.init_docker_registry()

    return t
