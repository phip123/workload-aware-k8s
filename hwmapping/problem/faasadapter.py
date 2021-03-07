import logging
import time
from concurrent.futures.process import ProcessPoolExecutor
from typing import Generator, Any, Dict, List

import simpy
from sim.core import Environment

from hwmapping.cluster import Cluster, State
from hwmapping.device import Device
from hwmapping.etheradapter import convert_to_devices
from hwmapping.faas.system import OurFaas
from hwmapping.problem.ga import execute_ga, execute_ga_parallel
from hwmapping.problem.model import LabelProblemSolverSettings, ProblemInstance, Requirements


def create_clusters(env: Environment) -> Dict[str, Cluster]:
    faas: OurFaas = env.faas
    clusters = {}
    deployments = list(faas.functions_deployments.values())
    for deployment in deployments:
        for service in deployment.get_services():
            cluster = service.labels['cluster']
            if clusters.get(cluster, None) is None:
                clusters[cluster] = Cluster(cluster, {service.name: service})
            else:
                clusters[cluster].functions[service.name] = service
    return clusters


def get_devices(env: Environment) -> List[Device]:
    return convert_to_devices(env.topology.get_nodes())


def set_reqs_for_cluster(cluster: Cluster, requirements: Requirements, env: Environment):
    """
    Sets the new requirements for each function definition
    """
    faas: OurFaas = env.faas
    for fn_deployment in faas.functions_deployments.values():
        for service in fn_deployment.get_services():
            service_cluster = service.labels.get('cluster', None)
            if service_cluster is not None and service_cluster == cluster.id:
                service.labels['device.edgerun.io/requirements'] = str(requirements.to_dict())


class LabelSolver:

    def __init__(self, settings: LabelProblemSolverSettings):
        self.settings = settings
        self.clusters = None
        self.devices = None
        self.state = None

    def solve(self, env: Environment) -> Generator[simpy.events.Event, Any, Any]:
        yield env.timeout(0)


class LabelSolverProcess:

    def __init__(self, settings: LabelProblemSolverSettings):
        self.settings = settings
        self.reconcile_interval = settings.reconcile_interval
        self.solver: LabelSolver = get_label_solver(settings)
        self.running = True

    def solve(self, env: Environment) -> Generator[simpy.events.Event, Any, Any]:
        while self.running:
            yield env.timeout(self.reconcile_interval)
            yield from self.solver.solve(env)
            # TODO remove when contention is implemented
            self.stop()

    def stop(self):
        self.running = False


def get_label_solver(settings: LabelProblemSolverSettings) -> LabelSolver:
    if settings.name == 'ga':
        return GeneticLabelProblemSolver(settings)

    raise ValueError(f'unknown solver: {settings.name}')


class GeneticLabelProblemSolver(LabelSolver):

    def __init__(self, settings: LabelProblemSolverSettings):
        super().__init__(settings)
        self.algorithm_params = settings.settings.get('ga_algorithm_parameters', None)
        self.run_settings = settings.settings.get('ga_run_settings', None)

    def solve(self, env: Environment) -> Generator[simpy.events.Event, Any, Any]:
        logging.info('Calculating Pod Labels')
        start = time.time()

        if self.clusters is None or len(self.clusters) == 0:
            # TODO caching because this may bottleneck - needs to figure out if clusters/devices have changed
            self.clusters: Dict[str, Cluster] = create_clusters(env)
            self.devices = get_devices(env)
            self.state = State(self.devices, self.clusters)

        results = []
        if self.settings.parallel:
            self.execute_ga_parallel(results)
        else:
            self.execute_ga_single_threaded(results)

        for result in results:
            set_reqs_for_cluster(result.instance.cluster, result.requirements, env)
        end = time.time()
        logging.info("Done calculating pods")
        yield env.timeout(end - start)

    def execute_ga_parallel(self, results):
        params = []
        for cluster_id, cluster in self.clusters.items():
            instance = ProblemInstance(
                cluster=cluster,
                state=self.state,
                performance_weight=self.settings.settings['performance_weight'],
                variety_weight=self.settings.settings['variety_weight']
            )

            params.append((instance, self.run_settings, self.algorithm_params))
        with ProcessPoolExecutor(max_workers=self.settings.workers) as executor:
            for result in executor.map(execute_ga_parallel, params):
                results.append(result)

    def execute_ga_single_threaded(self, results):
        for cluster_id, cluster in self.clusters.items():
            instance = ProblemInstance(
                cluster=cluster,
                state=self.state,
                performance_weight=self.settings.settings['performance_weight'],
                variety_weight=self.settings.settings['variety_weight']
            )

            results.append(execute_ga(instance, self.run_settings, self.algorithm_params))
