import concurrent.futures
import glob
import os
import pickle
from collections import defaultdict
from typing import List, Dict, Tuple

import pandas as pd

from hwmapping.evaluation.results import SingleRunSimResult
from hwmapping.notebook.results import preprocess_sim_result, scale_col


def read_sim_result(file: str, duration=None) -> SingleRunSimResult:
    with open(file, 'rb') as fd:
        result = pickle.load(fd)
        result = preprocess_sim_result(result, duration)
        return result


def read_sim_results(folder: str, profile: str, type_run: str, type_bench: str) -> List[SingleRunSimResult]:
    path = os.path.join(folder, profile, 'sim_results', type_run, type_bench)
    results = []
    for fname in glob.glob(f'{path}/*'):
        results.append(read_sim_result(fname))
    return results


def preprocess_topology_results(results: List[Dict[str, Dict[str, SingleRunSimResult]]]) -> List[
    Dict[str, Dict[str, SingleRunSimResult]]]:
    """
    FET and wait durations are scaled over all scenarios, workloads and type_runs
    :param results: for each device setting an entry, contains all workloads and type_runs (approaches)
    """
    min_maxs = min_max(results, 't_fet')
    wait_min_maxs = min_max(results, 't_wait_duration')
    for result in results:
        for workload, type_runs in result.items():
            for type_run, run_result in type_runs.items():
                fets = run_result.data['fets_df']
                fets = scale_col(fets, min_maxs, 't_fet')
                run_result.data['fets_df'] = scale_col(fets, wait_min_maxs, 't_wait_duration')
    return results


def read_topology_result(folder: str) -> Dict[str, Dict[str, SingleRunSimResult]]:
    profile = 'mixed'
    results_folder = 'sim_results'
    type_runs = ['all', 'ga', 'skippy', 'vanilla']
    workloads = ['constant', 'sine']
    data = defaultdict(dict)
    for workload in workloads:
        for type_run in type_runs:
            path = os.path.join(folder, profile, results_folder, type_run, workload)
            file = glob.glob(f'{path}/*.pkl')[0]
            data[workload][type_run] = read_sim_result(file)

    return data


def read_topology_results(folders: List[str], duration=None) -> Dict[str, Dict[str, List[SingleRunSimResult]]]:
    profile = 'mixed'
    results_folder = 'sim_results'
    type_runs = ['all', 'ga', 'skippy', 'vanilla']
    workloads = ['constant', 'sine']
    data = defaultdict(lambda: defaultdict(list))
    for folder in folders:
        for workload in workloads:
            for type_run in type_runs:
                path = os.path.join(folder, profile, results_folder, type_run, workload)
                try:
                    file = glob.glob(f'{path}/*.pkl')[0]
                except IndexError:
                    raise ValueError(f'No topology files under {path}')
                data[workload][type_run].append(read_sim_result(file, duration))

    return data


df_names = [
    "invocations_df",
    "scale_df",
    "schedule_df",
    "replica_deployment_df",
    "function_deployments_df",
    "function_deployment_df",
    "function_deployment_lifecycle_df",
    "functions_df",
    "flow_df",
    "network_df",
    "utilization_df",
    "fets_df",
]


def read_topology_results_concatenated(folders: List[str], duration=None) -> Dict[str, Dict[str, SingleRunSimResult]]:
    results_by_workload = read_topology_results(folders, duration)
    data = defaultdict(dict)

    for workload, results_by_type in results_by_workload.items():
        for type_run, results in results_by_type.items():
            dfs = defaultdict(list)
            for result in results:
                for df_name in df_names:
                    dfs[df_name].append(result.data[df_name])
            concat_dfs = {}
            for df_name in df_names:
                concat_dfs[df_name] = pd.concat(dfs[df_name])
            run_result = results[0]
            data[workload][type_run] = SingleRunSimResult(
                workload,
                type_run,
                concat_dfs,
                run_result.settings,
                run_result.sched_params,
                run_result.single_run_ga_result
            )
    return data


def check_if_failed(folder: str) -> bool:
    results_by_workload = read_topology_result(folder)
    for workload, results in results_by_workload.items():
        for _, result in results.items():
            if result.settings.get('failed', None) is None:
                continue
            elif result.settings['failed'] is True:
                return True
    return False


def read_topology(folder):
    return read_topology_results_concatenated([folder], 2000)


def read_topologies_par(folders):
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        for result in executor.map(read_topology, folders):
            results.append(result)
    return results


def get_topology_dfs(results: Dict[str, Dict[str, SingleRunSimResult]]) -> Dict[str, pd.DataFrame]:
    data = defaultdict(list)
    for workload, type_runs in results.items():
        for type_run, run_result in type_runs.items():
            for df_name in df_names:
                df = run_result.data[df_name]
                df['workload'] = workload
                df['type_run'] = type_run
                data[df_name].append(df)

    concatenated = {}
    for df_name in df_names:
        concatenated[df_name] = pd.concat(data[df_name])

    return concatenated


def min_max(results: List[Dict[str, Dict[str, SingleRunSimResult]]], col) -> Dict[str, Tuple[float, float]]:
    maxs = {}
    mins = {}
    for result in results:
        for workload, type_runs in result.items():
            for type_run, run_result in type_runs.items():
                fets = run_result.data['fets_df']
                for function in fets['function_deployment'].unique():
                    data: pd.Series = fets[fets['function_deployment'] == function][col]
                    amin = data.min()
                    amax = data.max()
                    if maxs.get(function, 0) < amax:
                        maxs[function] = amax
                    if mins.get(function, None) is None or mins[function] > amin:
                        mins[function] = amin
    min_maxs = {}
    for function in maxs.keys():
        min_maxs[function] = (mins[function], maxs[function])
    return min_maxs
