import itertools
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga, geneticalgorithm2

from hwmapping.calculations import sum_fet, calculate_requirements, calculate_diff_entropy
from hwmapping.device import Device
from hwmapping.faas.system import FunctionDefinition
from hwmapping.generator import xeon_reqs
from hwmapping.problem.model import ProblemInstance, Result


@dataclass
class GaRunSettings:
    no_plot: bool = True
    disable_progress_bar: bool = True
    start_generation: Optional[Dict] = None
    studEA: bool = False
    seed: int = 1234


@dataclass
class GaAlgorithmParams:
    max_num_iteration: int = 200
    population_size: int = 50
    mutation_probability: float = 0.1
    elit_ratio: float = 0.01
    crossover_probability: float = 0.5
    parents_portion: float = 0.3
    crossover_type: str = 'uniform'
    mutation_type: str = 'uniform_by_center'
    selection_type: str = 'roulette'
    max_iteration_without_improve: Optional[bool] = None


def base_requirements():
    # return some homogeneous config
    return xeon_reqs()


mean_fets = []


def fitness_score_device_type(X: np.ndarray, instance: ProblemInstance):
    devices = []
    filtered_devices = instance.filtered_devices
    for index, device in enumerate(filtered_devices):
        if X[index]:
            for d in instance.state.devices:
                if d.id == device.id:
                    devices.append(device)
    if len(devices) == 0:
        # because we minimize and only have values < 0 this will be worst
        return 1

    requirements = calculate_requirements(devices)

    # emd tries to show how far the requirements are away from each other
    het = calculate_diff_entropy(base_requirements(), requirements)
    try:
        fet, counter = sum_fet(devices, instance)
        mean_fets.append(fet / counter)
    except ValueError:
        # shouldn't happen, because GA uses the filtered devices
        # signals that a device is in the solution that can't execute the function, i.e. has no FET assigned
        return 1

    # performance factor, this favors faster settings
    performance = (fet / counter) * instance.performance_weight
    # variety factor, this favors more general settings
    if het == 0:
        variety = 0
    else:
        variety = (het / 16)
        variety = variety * instance.variety_weight

    fitness = -(performance + variety)
    return fitness


def fitness_score_individual(X: np.ndarray, instance: ProblemInstance):
    devices = []
    filtered_devices = instance.filtered_devices
    for index, device in enumerate(filtered_devices):
        if X[index]:
            devices.append(device)
    if len(devices) == 0:
        # because we minimize and only have values < 0 this will be worst
        return 1

    requirements = calculate_requirements(devices)

    # emd tries to show how far the requirements are away from each other
    het = calculate_diff_entropy(base_requirements(), requirements)
    try:
        fet, counter = sum_fet(devices, instance)
        mean_fets.append(fet / counter)
    except ValueError:
        # shouldn't happen, because GA uses the filtered devices
        # signals that a device is in the solution that can't execute the function, i.e. has no FET assigned
        return 1


    # performance factor, this favors faster settings
    performance = (fet / counter) * instance.performance_weight

    # variety factor, this favors more general settings
    if het == 0:
        variety = 0
    else:
        variety = (het / 16) + len(devices) / len(filtered_devices)
        variety = variety * instance.variety_weight

    fitness = -(performance + variety)
    return fitness


class BadSolutionError(Exception):
    pass


def create_result_enumeration(scores: List[Tuple[List[bool], float]], duration, instance: ProblemInstance, settings,
                              algorithm_params):
    best_solution = min(scores, key=lambda l: l[1])
    device_types = []
    for included, device_type in zip(best_solution[0], instance.filtered_devices):
        if included:
            device_types.append(device_type.id)
    devices = []
    for device in instance.state.devices:
        if device.id in device_types:
            devices.append(device)

    try:
        fet, counter = sum_fet(devices, instance)
        if counter == 0:
            fet = -1
        else:
            fet /= counter
        reqs = calculate_requirements(devices)
        successful = fet != -1
        return Result(instance, duration, reqs, devices, fet, successful, {'function': best_solution[1]}, [], {}, {})
    except ValueError:
        # shouldn't happen, because GA uses the filtered devices
        # signals that a device is in the solution that can't execute the function, i.e. has no FET assigned
        raise BadSolutionError


def create_result(model: geneticalgorithm2, duration: float, instance: ProblemInstance,
                  settings: GaRunSettings, algorithm_params: GaAlgorithmParams) -> Result:
    devices = []
    for index, device in enumerate(instance.filtered_devices):
        if model.best_variable[index] == 1:
            devices.append(device)

    try:
        fet, counter = sum_fet(devices, instance)
        if counter == 0:
            fet = -1
        else:
            fet /= counter
        reqs = calculate_requirements(devices)
        successful = fet != -1
        return Result(instance, duration, reqs, devices, fet, successful, model.output_dict, model.report,
                      asdict(settings),
                      asdict(algorithm_params))
    except ValueError:
        # shouldn't happen, because GA uses the filtered devices
        # signals that a device is in the solution that can't execute the function, i.e. has no FET assigned
        raise BadSolutionError


def can_run(device: Device, function: FunctionDefinition) -> bool:
    sampled_fet = function.sample_fet(device.id)
    return sampled_fet != None


def filter_devices(instance: ProblemInstance) -> List[Device]:
    """
    Filter out all devices that are not able to run at least one function of the cluster
    """
    devices = []
    for device in instance.state.devices:
        can_run = True
        for function in instance.cluster.functions.values():
            if function.sample_fet(device.id) is None:
                can_run = False
        if can_run:
            devices.append(device)
    return devices


def execute_ga_parallel(args):
    return execute_ga(args[0], args[1], args[2])


def get_unique_devices(devices):
    filtered = {}
    for d in devices:
        filtered[d.id] = d
    return list(filtered.values())


def permutations(devices: int):
    for i in range(devices):
        yield from place_ones(devices, i)


def place_ones(size, count):
    # https://stackoverflow.com/a/43817007
    for positions in itertools.combinations(range(size), count):
        p = [False] * size

        for i in positions:
            p[i] = True

        yield p


def execute_ga(instance: ProblemInstance, settings: GaRunSettings = None,
               algorithm_params: GaAlgorithmParams = None) -> Result:
    start = time.perf_counter()
    if settings is None:
        settings = GaRunSettings()

    if algorithm_params is None:
        algorithm_params = GaAlgorithmParams()

    if settings.start_generation is None:
        settings.start_generation = {'variables': None, 'scores': None}

    if instance.consider_all:
        def f(X):
            return fitness_score_device_type(X, instance)

        instance.filtered_devices = get_unique_devices(filter_devices(instance))
        scores = []
        for x in permutations(len(instance.filtered_devices)):
            X = []
            for index, include in enumerate(x):
                X.append(include)
            scores.append((X, f(X)))
        X = [True] * len(instance.filtered_devices)
        scores.append((X, f(X)))
        end = time.perf_counter()
        return create_result_enumeration(scores, end - start, instance, settings, algorithm_params)
    else:
        def f(X):
            return fitness_score_individual(X, instance)

        instance.filtered_devices = filter_devices(instance)
    algorithm_param = asdict(algorithm_params)

    model = ga(function=f, dimension=len(instance.filtered_devices),
               variable_type='bool',
               algorithm_parameters=algorithm_param, function_timeout=100000)

    model.run(
        **asdict(settings)
    )
    end = time.perf_counter()
    return create_result(model, end - start, instance, settings, algorithm_params)
