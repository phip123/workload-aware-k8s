#!/usr/bin/env python
# coding: utf-8


import pickle
import random
from collections import defaultdict

import numpy as np
from skippy.core.scheduler import Scheduler
from skippy.core.storage import StorageIndex

from hwmapping.calculations import calculate_diff_entropy as heterogeneity_score, calculate_requirements
from hwmapping.cli.eval_sim import save_sim_result, run_sim
from hwmapping.device import ArchProperties
from hwmapping.etheradapter import convert_to_ether_nodes, convert_to_devices
from hwmapping.evaluation import images
from hwmapping.evaluation.benchmarks.sine import SineBenchmark
from hwmapping.evaluation.deployments import create_all_deployments
from hwmapping.evaluation.fetdistributions import execution_time_distributions
from hwmapping.evaluation.functionsim import PythonHttpSimulatorFactory
from hwmapping.evaluation.resources import resources_per_node_image
from hwmapping.evaluation.results import set_requirements
from hwmapping.evaluation.run import EnvSettings
from hwmapping.evaluation.topology import urban_sensing_topology
from hwmapping.faas.predicates import NodeHasAcceleratorPred, NodeHasFreeTpu, NodeHasFreeGpu, CanRunPred
from hwmapping.faas.system import *
from hwmapping.generator import GeneratorSettings, generate_devices, xeon_reqs
from hwmapping.model import *
from hwmapping.notebook import skippy, ga

logging.basicConfig(level=logging.INFO)

base_reqs = xeon_reqs()

test_settings = GeneratorSettings(
    arch={
        Arch.X86: 0.3,
        Arch.AARCH64: 0.5,
        Arch.ARM32: 0.2
    },
    properties={
        Arch.X86: ArchProperties(
            arch=Arch.X86,
            accelerator={
                Accelerator.NONE: 0.9,
                Accelerator.GPU: 0.1,
                Accelerator.TPU: 0
            },
            cores={
                Bins.LOW: 0,
                Bins.MEDIUM: 0,
                Bins.HIGH: 0.7,
                Bins.VERY_HIGH: 0.3
            },
            location={
                Location.CLOUD: 0.6,
                Location.MEC: 0.4,
                Location.EDGE: 0,
                Location.MOBILE: 0
            },
            connection={
                Connection.ETHERNET: 1,
                Connection.WIFI: 0,
                Connection.MOBILE: 0
            },
            network={
                Bins.LOW: 0,
                Bins.MEDIUM: 0,
                Bins.HIGH: 0.1,
                Bins.VERY_HIGH: 0.9
            },
            cpu_mhz={
                Bins.LOW: 0,
                Bins.MEDIUM: 0.7,
                Bins.HIGH: 0.25,
                Bins.VERY_HIGH: 0.05
            },
            cpu={
                CpuModel.XEON: 0.7,
                CpuModel.I7: 0.3
            },
            ram={
                Bins.LOW: 0,
                Bins.MEDIUM: 0.05,
                Bins.HIGH: 0.45,
                Bins.VERY_HIGH: 0.5
            },
            gpu_vram={
                Bins.LOW: 0,
                Bins.MEDIUM: 0,
                Bins.HIGH: 0.9,
                Bins.VERY_HIGH: 0.1
            },
            gpu_model={
                GpuModel.TURING: 1,
            },
            gpu_mhz={
                Bins.LOW: 0,
                Bins.MEDIUM: 0,
                Bins.HIGH: 1,
                Bins.VERY_HIGH: 0
            },
            disk={
                Disk.SSD: 1,
                Disk.SD: 0,
                Disk.NVME: 0,
                Disk.FLASH: 0,
                Disk.HDD: 0
            }
        ),
        Arch.AARCH64: ArchProperties(
            arch=Arch.AARCH64,
            accelerator={
                Accelerator.NONE: 0.2,
                Accelerator.GPU: 0.7,
                Accelerator.TPU: 0.1
            },
            cores={
                Bins.LOW: 0,
                Bins.MEDIUM: 0.9,
                Bins.HIGH: 0.1,
                Bins.VERY_HIGH: 0
            },
            location={
                Location.CLOUD: 0,
                Location.MEC: 0.2,
                Location.EDGE: 0.8,
                Location.MOBILE: 0
            },
            connection={
                Connection.ETHERNET: 0.2,
                Connection.WIFI: 0.8,
                Connection.MOBILE: 0
            },
            network={
                Bins.LOW: 0.1,
                Bins.MEDIUM: 0.7,
                Bins.HIGH: 0.2,
                Bins.VERY_HIGH: 0
            },
            cpu_mhz={
                Bins.LOW: 0.1,
                Bins.MEDIUM: 0.8,
                Bins.HIGH: 0.1,
                Bins.VERY_HIGH: 0
            },
            cpu={
                CpuModel.ARM: 1
            },
            ram={
                Bins.LOW: 0.3,
                Bins.MEDIUM: 0.5,
                Bins.HIGH: 0.2,
                Bins.VERY_HIGH: 0
            },
            gpu_vram={
                Bins.LOW: 0,
                Bins.MEDIUM: 0.9,
                Bins.HIGH: 0.1,
                Bins.VERY_HIGH: 0
            },
            gpu_model={
                GpuModel.PASCAL: 0.3,
                GpuModel.MAXWELL: 0.4,
                GpuModel.TURING: 0.3
            },
            gpu_mhz={
                Bins.LOW: 0,
                Bins.MEDIUM: 0.9,
                Bins.HIGH: 0.1,
                Bins.VERY_HIGH: 0
            },
            disk={
                Disk.SSD: 0,
                Disk.SD: 0.5,
                Disk.NVME: 0,
                Disk.FLASH: 0.5,
                Disk.HDD: 0
            }
        ),
        Arch.ARM32: ArchProperties(
            arch=Arch.ARM32,
            accelerator={
                Accelerator.NONE: 1,
                Accelerator.GPU: 0,
                Accelerator.TPU: 0
            },
            cores={
                Bins.LOW: 0.5,
                Bins.MEDIUM: 0.5,
                Bins.HIGH: 0,
                Bins.VERY_HIGH: 0
            },
            location={
                Location.CLOUD: 0,
                Location.MEC: 0,
                Location.EDGE: 0.9,
                Location.MOBILE: 0.1
            },
            connection={
                Connection.ETHERNET: 0.05,
                Connection.WIFI: 0.85,
                Connection.MOBILE: 0.1
            },
            network={
                Bins.LOW: 0.6,
                Bins.MEDIUM: 0.4,
                Bins.HIGH: 0,
                Bins.VERY_HIGH: 0
            },
            cpu_mhz={
                Bins.LOW: 0.5,
                Bins.MEDIUM: 0.5,
                Bins.HIGH: 0,
                Bins.VERY_HIGH: 0
            },
            cpu={
                CpuModel.ARM: 1
            },
            ram={
                Bins.LOW: 0.4,
                Bins.MEDIUM: 0.6,
                Bins.HIGH: 0,
                Bins.VERY_HIGH: 0
            },
            disk={
                Disk.SSD: 0,
                Disk.SD: 1,
                Disk.NVME: 0,
                Disk.FLASH: 0,
                Disk.HDD: 0
            },
            gpu_vram={},
            gpu_model={},
            gpu_mhz={},
        )
    }
)

use_predefined_devices = True

if use_predefined_devices:
    with open('data/collections/collection_01_04_2021/ga_devices/hybrid_balanced_score_7.384.pkl', 'rb') as fd:
        devices = pickle.load(fd)
else:
    num_devices = 100
    devices = generate_devices(num_devices, test_settings)
print(len(devices))
print(heterogeneity_score(base_reqs, calculate_requirements(devices)))

ether_nodes = convert_to_ether_nodes(devices)
print(ether_nodes[0])

device_types = np.unique(list(map(lambda e: e.name[:e.name.rindex('_')], ether_nodes)))
devices_by_type = defaultdict(list)
for device in ether_nodes:
    devices_by_type[device.name[:device.name.rindex('_')]].append(device)

print('\navailable nodes')
for device_type in device_types:
    print(device_type, len(devices_by_type[device_type]))

print(len(ether_nodes))
print(len(devices))

print(heterogeneity_score(base_reqs, calculate_requirements(convert_to_devices(ether_nodes))))

fet_oracle = FetOracle(execution_time_distributions)
resource_oracle = ResourceOracle(resources_per_node_image)

deployments = list(create_all_deployments(fet_oracle, resource_oracle).values())
function_images = images.all_images

predicates = []
predicates.extend(Scheduler.default_predicates)
predicates.extend([
    CanRunPred(fet_oracle, resource_oracle),
    NodeHasAcceleratorPred(),
    NodeHasFreeGpu(),
    NodeHasFreeTpu()
])

priorities = []
skippy_priorities = skippy.get_priorities(
    latency_weight=1,
    data_weight=1
)

ga_priorities = ga.get_priorities(
    fet_oracle,
    resource_oracle,
    capability_weight=1,
    contention_weight=1,
    fet_weight=1
)

np.random.seed(1234)
random.seed(1234)

priorities.extend(skippy_priorities)
priorities.extend(ga_priorities)

sched_params = {
    'percentage_of_nodes_to_score': 100,
    'priorities': priorities,
    'predicates': predicates
}


model_folder = './data/collections/collection_01_04_2021/ml'

duration = 200
max_rps = 300
period = 75
benchmark = SineBenchmark('mixed',
                          duration=duration, max_rps=max_rps, period=period, model_folder=model_folder)

ga_file = 'data/collections/collection_01_04_2021/solutions/req_creation_01_13_2021_22_21_24/mixed/ga_results/k_7_clustering_4c07_edge_cloudlet.pkl'
with open(ga_file, 'rb') as fd:
    ga_run = pickle.load(fd)

set_requirements(benchmark, ga_run)
type_run = 'ga'
settings = {
    'percentage_nodes_to_score': 100,
    'latency_weight': 1,
    'data_weight': 1,
    'contention_weight': 1,
    'capability_weight': 1,
    'fet_weight': 1,
    'duration': duration,
    'max_rps': max_rps,
    'period': period,
    'type': 'sine',
    'optimization': type_run
}
result = run_sim((benchmark, 'all', sched_params, ga_run, settings))
save_sim_result('./data/collections/collection_01_04_2021/adhoc', result)
