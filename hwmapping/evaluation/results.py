import json
from dataclasses import dataclass
from typing import Dict, List

from hwmapping.device import Device
from hwmapping.evaluation.benchmark import BenchmarkBase
from hwmapping.faas.system import FunctionDefinition
from hwmapping.problem.ga import GaAlgorithmParams, GaRunSettings
from hwmapping.problem.model import Result


@dataclass
class SingleRunGaResult:
    device_file: str
    devices: List[Device]
    clustering_file: str
    clustering: Dict
    results: List[Result]
    settings: GaRunSettings
    algorithm_params: GaAlgorithmParams


@dataclass
class SingleRunSimResult:
    type_bench: str  # either sine or constant
    type_run: str  # either ga, vanilla or priorities
    data: Dict  # contains all dataframes
    settings: Dict  # contains benchmark settigns
    sched_params: Dict
    single_run_ga_result: SingleRunGaResult


def set_requirements(benchmark: BenchmarkBase, single_run_result: SingleRunGaResult):
    with open(single_run_result.clustering_file, 'r') as fd:
        clustering = json.load(fd)
    results = single_run_result.results
    for deployment in benchmark.deployments:
        function: FunctionDefinition
        for function in deployment.get_services():
            cluster = clustering[function.image]
            result = list(filter(lambda r: r.instance.cluster.id == cluster, results))[0]
            function.labels['device.edgerun.io/requirements'] = str(result.requirements.to_dict())
