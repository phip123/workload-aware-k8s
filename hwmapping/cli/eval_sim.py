import datetime
import glob
import json
import logging
import os
import pickle
import random
import resource
import sys
import time
import uuid
from pathlib import Path
from typing import Generator, Tuple, List

import numpy as np
from sim.net import LowBandwidthException

from hwmapping.cli.eval_ga import create_pool_executor
from hwmapping.cli.simparams import vanilla_constant, ga_constant, \
    ga_prerecorded, skippy_constant, skippy_prerecorded, all_constant, all_prerecorded, vanilla_prerecorded
from hwmapping.etheradapter import convert_to_ether_nodes
from hwmapping.evaluation.fetdistributions import execution_time_distributions
from hwmapping.evaluation.oracle import FetOracle, ResourceOracle
from hwmapping.evaluation.resources import resources_per_node_image
from hwmapping.evaluation.results import SingleRunSimResult, SingleRunGaResult
from hwmapping.evaluation.run import run_single
from hwmapping.faas.system import FunctionDeployment
from hwmapping.notebook.setup import setup_topology

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'DEBUG'))


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def create_weight_text(result: SingleRunSimResult):
    if result.type_run == 'ga':
        capability_weight = result.settings['capability_weight']
        contention_weight = result.settings['contention_weight']
        fet_weight = result.settings['fet_weight']
        text = f'cap_{capability_weight}-co_{contention_weight}-fet_{fet_weight}'
        return text
    elif result.type_run == 'contention':
        contention_weight = result.settings['contention_weight']
        fet_weight = result.settings['fet_weight']
        text = f'co_{contention_weight}-fet_{fet_weight}'
        return text
    elif result.type_run == 'vanilla':
        return "vanilla"
    elif result.type_run == 'skippy':
        data_weight = result.settings['data_weight']
        latency_weight = result.settings['latency_weight']
        text = f'dat_{data_weight}-lat_{latency_weight}'
        return text
    elif result.type_run == 'all':
        capability_weight = result.settings['capability_weight']
        contention_weight = result.settings['contention_weight']
        fet_weight = result.settings['fet_weight']
        data_weight = result.settings['data_weight']
        latency_weight = result.settings['latency_weight']
        text = f'cap_{capability_weight}-co_{contention_weight}-fet_{fet_weight}-dat_{data_weight}-lat_{latency_weight}'
        return text
    else:
        raise ValueError(f'unknown type run: {result.type_run}')


def save_sim_result(sim_results_folder: str, result: SingleRunSimResult):
    # save under /type_bench/type_run/clustering_uuid.pkl
    clustering_file = os.path.basename(result.single_run_ga_result.clustering_file[:-5])
    random_postfix = str(uuid.uuid1())[:4]
    weights = create_weight_text(result)
    file_name = f'{clustering_file}_{weights}_{random_postfix}.pkl'
    file = os.path.join(sim_results_folder, result.type_run, result.type_bench, file_name)
    mkdir(os.path.join(sim_results_folder, result.type_run, result.type_bench))
    with open(file, 'wb') as fd:
        pickle.dump(result, fd)


def create_sim_params(single_run_result: SingleRunGaResult, profile: str, model_folder: str, fet_oracle,
                      resource_oracle, settings) -> Generator[
    Tuple, None, None]:
    percentage_of_nodes_to_scores = settings['percentage_of_nodes_to_scores']
    capability_weights = np.linspace(*settings['capability_weights'])
    contention_weights = np.linspace(*settings['contention_weights'])
    fet_weights = np.linspace(*settings['fet_weights'])
    latency_weights = np.linspace(*settings['latency_weights'])
    data_weights = np.linspace(*settings['data_weights'])
    durations = np.linspace(*settings['durations'])
    constant_rps = np.linspace(*settings['constant_rps'])
    benchmark = settings['benchmark']
    arrival_patterns = settings['arrival_patterns']

    yield from ga_prerecorded(capability_weights, contention_weights, durations, fet_oracle, fet_weights, model_folder,
                              percentage_of_nodes_to_scores, profile, resource_oracle, arrival_patterns,
                              single_run_result)

    yield from skippy_prerecorded(latency_weights, durations, fet_oracle, data_weights, model_folder,
                                  percentage_of_nodes_to_scores, profile, resource_oracle, arrival_patterns,
                                  single_run_result)

    yield from vanilla_prerecorded(durations, fet_oracle, model_folder, percentage_of_nodes_to_scores, profile,
                                   resource_oracle, arrival_patterns, single_run_result)

    yield from all_prerecorded(capability_weights, contention_weights, fet_weights, latency_weights, data_weights,
                               durations,
                               fet_oracle, model_folder, percentage_of_nodes_to_scores, profile, resource_oracle,
                               arrival_patterns, single_run_result)

    yield from ga_constant(capability_weights, constant_rps, contention_weights, durations, fet_oracle, fet_weights,
                           model_folder, percentage_of_nodes_to_scores, profile, resource_oracle, single_run_result,
                           benchmark)

    yield from skippy_constant(latency_weights, constant_rps, data_weights, durations, fet_oracle, model_folder,
                               percentage_of_nodes_to_scores, profile, resource_oracle, single_run_result, benchmark)

    yield from all_constant(latency_weights, data_weights, capability_weights, contention_weights, fet_weights,
                            constant_rps, durations, fet_oracle, model_folder, percentage_of_nodes_to_scores, profile,
                            resource_oracle, single_run_result, benchmark)

    yield from vanilla_constant(constant_rps, durations, fet_oracle, model_folder, percentage_of_nodes_to_scores,
                                profile, resource_oracle, single_run_result, benchmark)


def run_sim(args) -> SingleRunSimResult:
    benchmark = args[0]
    type_run = args[1]
    sched_params = args[2]
    single_run_ga_result: SingleRunGaResult = args[3]
    settings = args[4]

    # need to create here, because we change the labels of the nodes, would lead to inconsistency in
    # concurrent scenarios
    ether_nodes = convert_to_ether_nodes(single_run_ga_result.devices)

    env_settings, topology = setup_topology(sched_params, ether_nodes)
    start = time.perf_counter()
    try:
        sim = run_single(benchmark, env_settings, topology)
        settings['failed'] = False
        dfs = {
            "invocations_df": sim.env.metrics.extract_dataframe('invocations'),
            "scale_df": sim.env.metrics.extract_dataframe('scale'),
            "schedule_df": sim.env.metrics.extract_dataframe('schedule'),
            "replica_deployment_df": sim.env.metrics.extract_dataframe('replica_deployment'),
            "function_deployments_df": sim.env.metrics.extract_dataframe('function_deployments'),
            "function_deployment_df": sim.env.metrics.extract_dataframe('function_deployment'),
            "function_deployment_lifecycle_df": sim.env.metrics.extract_dataframe('function_deployment_lifecycle'),
            "functions_df": sim.env.metrics.extract_dataframe('functions'),
            "flow_df": sim.env.metrics.extract_dataframe('flow'),
            "network_df": sim.env.metrics.extract_dataframe('network'),
            "utilization_df": sim.env.metrics.extract_dataframe('utilization'),
            'fets_df': sim.env.metrics.extract_dataframe('fets')
        }

    except LowBandwidthException as ex:
        logging.error(f'Error executing sim {type_run} {settings["optimization"]}', ex)
        settings['failed'] = True
        dfs = {}
    end = time.perf_counter()
    duration = end - start
    settings['duration'] = duration

    return SingleRunSimResult(
        type_bench=benchmark.type,
        type_run=type_run,
        data=dfs,
        settings=settings,
        sched_params=sched_params,
        single_run_ga_result=single_run_ga_result
    )


def run_all_sim(single_run_result: SingleRunGaResult, profile: str, model_folder: str, fet_oracle, resource_oracle,
                settings):
    with create_pool_executor(settings) as executor:
        for result in executor.map(run_sim, create_sim_params(single_run_result, profile, model_folder, fet_oracle,
                                                              resource_oracle, settings)):
            yield result


def check_ga_results_available(root: str, solutions_folder: str, ga_results_folder: str, profiles: List[str]) -> bool:
    for profile in profiles:
        folder = os.path.join(root, solutions_folder, ga_results_folder, profile, 'ga_results')
        if not os.path.isdir(folder):
            return False

    return True


def read_ga_results(root: str, solutions_folder: str, ga_results_folder: str, profiles: List[str]):
    for profile in profiles:
        folder = os.path.join(root, solutions_folder, ga_results_folder, profile, 'ga_results')
        for fname in glob.glob(f'{folder}/*'):
            with open(fname, 'rb') as fd:
                a = pickle.load(fd), profile
                yield a


def main():
    """
    Example of settings.json:
    {
        "cores": int,
        "seed": int,
        "model_folder": "path/to/models",
        "device_folder": "path/to/devices",
        "clustering_folder": "path/to/clustering_jsons",
        "solutions_folder": "path/where/solutions/are/saved",
        "ga_results|folder": "path/to/ga_solution_folder

        Scheduler settings
        "percentage_of_nodes_to_scores": [int],

        PriorityWeights
        "capability_weights": [start, stop, num],
        "contention_weights": [start, stop, num],
        "fet_weights": [start, stop, num],

        "profiles": selection of ["ai", "service", "mixed"],
        "durations": [start, stop, num],

        Sine scenario settings
        "sine_max_rps": [start, stop, num],
        "sine_periods": [start, stop, num],

        Constant scenario settings
        "constant_rps": [start,stop,num]
    }

    * folder paths are relative to path of settings.json

    * for definition of [start, stop, num] lookup up doc of np.linspace

    Hints on setting GA parameters:
    https://github.com/PasaOpasen/geneticalgorithm2#hints-on-how-to-adjust-genetic-algorithms-parameters

    * max_num_iterations should be large, only downside is longer runtime
    * population size is shorter recommended, may get stuck in local optima (100 recommended)
    * elit_ratio: recommended to select at most one from population, sometimes 0 is best
    * mutation_probability: probably has to be adjusted more than others; can range from 0.01 to 0.5 or even larger
    * parents portion: anything between 1 and 0 may work
    * crossover_type: uniform crossover recommended

    :return:
    """
    if len(sys.argv) != 2:
        raise ValueError(
            'Expected usage:eval.py settings.json')
    settings_file_path = sys.argv[1]
    settings_folder = os.path.dirname(settings_file_path)
    with open(settings_file_path, 'r') as fd:
        settings = json.load(fd)

    seed = settings['seed']
    set_seed(seed)

    model_folder = os.path.join(settings_folder, settings['model_folder'])
    solutions_folder = settings['solutions_folder']
    ga_results_folder = settings['ga_results_folder']
    profiles = settings['profiles']

    if not check_ga_results_available(settings_folder, solutions_folder, ga_results_folder, profiles):
        raise ValueError(f"Passed ga_results_folder were not availabel: {ga_results_folder}")

    fet_oracle = FetOracle(execution_time_distributions)
    resource_oracle = ResourceOracle(resources_per_node_image)

    now = datetime.datetime.now()
    now = now.strftime('%m_%d_%Y_%H_%M_%S')
    destination_folder = os.path.join(settings_folder, solutions_folder,
                                      f'{os.path.basename(settings_file_path[:-5])}_{now}')
    mkdir(destination_folder)
    for profile in profiles:
        single_run_result: SingleRunGaResult
        results_folder = os.path.join(destination_folder, profile)
        sim_results_folder = os.path.join(results_folder, 'sim_results')

        mkdir(results_folder)
        mkdir(sim_results_folder)

    logging.info('start sims')
    scaling_settings = {
        'alert_window': FunctionDeployment.alert_window,
        'target_queue_length': FunctionDeployment.target_queue_length,
        'target_average_rps_threshold': FunctionDeployment.target_average_rps_threshold
    }
    for single_run_result, profile in read_ga_results(settings_folder, solutions_folder, ga_results_folder, profiles):
        for result in run_all_sim(single_run_result, profile, model_folder, fet_oracle, resource_oracle,
                                  settings):
            results_folder = os.path.join(destination_folder, profile)
            sim_results_folder = os.path.join(results_folder, 'sim_results')
            result.settings.update(settings)
            result.settings.update(scaling_settings)
            save_sim_result(sim_results_folder, result)


def mkdir(destination_folder):
    Path(destination_folder).mkdir(exist_ok=True, parents=True)


def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 4, hard))


def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


if __name__ == '__main__':
    memory_limit()
    try:
        main()
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)
