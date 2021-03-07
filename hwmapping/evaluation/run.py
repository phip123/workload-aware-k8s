import datetime
import os
import pickle
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Callable, Optional, Dict

import numpy as np
from ether.core import Node
from sim.benchmark import Benchmark
from sim.core import Environment
from sim.docker import ContainerRegistry
from sim.faas import SimulatorFactory
from sim.faassim import Simulation
from sim.logging import SimulatedClock, RuntimeLogger, NullLogger
from sim.resource import MetricsServer
from sim.skippy import SimulationClusterContext
from sim.topology import Topology
from skippy.core.scheduler import Scheduler
from skippy.core.storage import StorageIndex

from hwmapping.evaluation.oracle import ResourceOracle
from hwmapping.evaluation.resources import resources_per_node_image
from hwmapping.faas.metrics import OurMetrics
from hwmapping.faas.resource import ResourceMonitor
from hwmapping.faas.scaler import HorizontalPodAutoscaler
from hwmapping.faas.system import OurFaas, FunctionDeployment
from hwmapping.problem.faasadapter import LabelSolverProcess
from hwmapping.problem.model import LabelProblemSolverSettings


@dataclass
class HorizontalPodAutoscalerSettings:
    average_window: int = 10
    reconcile_interval: int = 15
    target_tolerance = 0.1


@dataclass
class EnvSettings:
    simulator_factory: SimulatorFactory
    # mk_topology: Callable[[List[Node]], Topology]
    sched_params: Dict
    scale_by_requests: bool = False
    scale_by_resources: bool = True
    scale_by_requests_per_replica: bool = False
    scale_by_queue_requests_per_replica: bool = False
    hpaSettings: HorizontalPodAutoscalerSettings = HorizontalPodAutoscalerSettings()
    label_problem_solver_settings: Optional[LabelProblemSolverSettings] = None
    storage_index: StorageIndex = StorageIndex()
    null_logger: bool = False


@dataclass
class BatchExperimentSettings:
    benchmark: Benchmark
    env_settings: EnvSettings
    deployments: List[FunctionDeployment]
    ether_nodes: List[List[Node]]
    exp_name: str
    interarrival_times: np.ndarray
    scores: List[float]
    distribution: str
    distribution_params: tuple
    arrival_pattern_file: str
    max_rps: int
    save: bool


def save_results(exp_id: str, sim: Simulation, df_names: List[str], folders: List[str]):
    for df_name, folder in zip(df_names, folders):
        data = sim.env.metrics.extract_dataframe(df_name)
        file = os.path.join(folder, f'{exp_id}.csv')
        data.to_csv(file)


def run_single(benchmark: Benchmark, env_settings: EnvSettings, topology: Topology):
    env = configure_env(env_settings, topology)
    sim = Simulation(env.topology, benchmark, env=env)
    sim.run()
    return sim


def run_batch(args):
    print('run sim')
    print('aaaa', args)
    exp_id: str = args[0]
    df_folders: List[str] = args[1]
    df_folder_paths: List[str] = args[2]
    settings: BatchExperimentSettings = args[3]
    topology: Topology = args[4]

    env = configure_env(settings.env_settings, topology)

    sim = Simulation(env.topology, settings.benchmark, env=env)
    sim.run()
    if settings.save:
        save_results(exp_id, sim, df_folders, df_folder_paths)


def configure_env(settings: EnvSettings, topology: Topology):
    env = Environment()
    env.simulator_factory = settings.simulator_factory
    if settings.null_logger:
        env.metrics = OurMetrics(env, log=NullLogger())
    else:
        env.metrics = OurMetrics(env, log=RuntimeLogger(SimulatedClock(env)))
    env.topology = topology
    env.faas = OurFaas(env, settings.scale_by_requests, settings.scale_by_requests_per_replica,
                       settings.scale_by_queue_requests_per_replica)
    env.container_registry = ContainerRegistry()
    env.storage_index = settings.storage_index
    env.cluster = SimulationClusterContext(env)
    env.scheduler = Scheduler(env.cluster, **settings.sched_params)
    env.metrics_server = MetricsServer()

    # TODO inject resource oracle
    resource_monitor = ResourceMonitor(env, ResourceOracle(resources_per_node_image))
    env.background_processes.append(lambda env: resource_monitor.run())

    if settings.scale_by_resources:
        hpa_settings = settings.hpaSettings
        hpa = HorizontalPodAutoscaler(
            env,
            average_window=hpa_settings.average_window,
            reconcile_interval=hpa_settings.reconcile_interval,
            target_tolerance=hpa_settings.target_tolerance
        )
        env.background_processes.append(lambda env: hpa.run())

    if settings.label_problem_solver_settings is not None:
        solver = LabelSolverProcess(settings.label_problem_solver_settings)
        env.background_processes.append(lambda env: solver.solve(env))
    return env


def execute_batch_experiments(settings: BatchExperimentSettings, mk_topology: Callable[[List[Node]], Topology]):
    now = datetime.datetime.now()
    now = now.strftime('%Y_%m_%d_%H_%M_%S')
    exp_name = f'{settings.exp_name}_{now}'
    ether_nodes = settings.ether_nodes
    # will be pickled for later usage
    exp_desc = {
        'distribution': settings.distribution,
        'distribution_params': settings.distribution_params,
        'arrival_pattern_file': settings.arrival_pattern_file,
        'max_rps': settings.max_rps,
        'name': exp_name,
        'length': len(ether_nodes),
        'scores': [],
        'nodes': ether_nodes,
        'interarrival_times': settings.interarrival_times,
        'sched_params': settings.env_settings.sched_params,
        'deployments': settings.deployments
    }

    folder = f'./{exp_name}'

    if settings.save:
        Path(folder).mkdir(parents=True, exist_ok=True)

    df_folders = ['invocations', 'scale', 'schedule', 'replica_deployment',
                  'function_deployments', 'function_deployment', 'function_deployment_lifecycle',
                  'functions', 'flow', 'network', 'utilization']

    df_folder_paths = []

    # create subfolders where dfs are stored
    for df_folder in df_folders:
        path = os.path.join(folder, df_folder)
        df_folder_paths.append(path)
        if settings.save:
            Path(path).mkdir(parents=True, exist_ok=True)

    # create params
    execution_params = []
    for index, nodes in enumerate(ether_nodes):
        exp_id = str(index)
        exp_desc['scores'].append(settings.scores[index])
        topology = mk_topology(ether_nodes[index])
        execution_params.append((exp_id, df_folders, df_folder_paths, settings, topology))

    execution_params = np.array(execution_params)
    splits = list(filter(lambda l: len(l) > 0, map(lambda l: l[0], np.array_split(execution_params, 10))))
    with ProcessPoolExecutor() as executor:
        for _ in executor.map(run_batch, splits):
            pass
    # with Pool(4) as p:
    #     p.map_async(run_batch, splits)
    #     p.close()
    #     p.join()
    # run_batch(params)

    if settings.save:
        file_name = f'description.pickle'
        with open(os.path.join(folder, file_name), 'wb+') as fd:
            pickle.dump(exp_desc, fd)
