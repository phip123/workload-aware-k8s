import concurrent.futures
import concurrent.futures
from typing import Dict, List

import pandas as pd

from hwmapping.evaluation.results import SingleRunSimResult
from hwmapping.notebook.reader.simreader import read_topology_results_concatenated, preprocess_topology_results, \
    get_topology_dfs
from hwmapping.notebook.throughput import calculate_throughput_of_topology
from hwmapping.notebook.usage import pretty_string_device_file
from hwmapping.notebook.utils import scale


def read_topology(folder):
    return read_topology_results_concatenated([folder], 2000)


def read(folders):
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        for result in executor.map(read_topology, folders):
            results.append(result)
    return results


def agg_fet_per_function(cloud_topology_folders, hybrid_folders, edge_cloudlet_topology_folders):
    data = {
        'mean_count': [],
        'mean_fet': [],
        'std_fet': [],
        'mean_wait': [],
        'std_wait': [],
        'run': [],
        'workload': [],
        'devices': [],
        'type_run': [],
        'function': []
    }
    for index, (cloud_folder, hybrid_folder, edge_folder) in enumerate(
            zip(cloud_topology_folders, hybrid_folders, edge_cloudlet_topology_folders)):
        results = read([cloud_folder, hybrid_folder, edge_folder])
        scaled = preprocess_topology_results([results[0], results[1], results[2]])
        scaled_cloud = scaled[0]
        scaled_hybrid = scaled[1]
        scaled_edge = scaled[2]

        cloud_dfs = get_topology_dfs(scaled_cloud)
        hybrid_dfs = get_topology_dfs(scaled_hybrid)
        edge_dfs = get_topology_dfs(scaled_edge)

        edge_throughputs = calculate_throughput_of_topology(edge_dfs['invocations_df'])
        hybrid_throughputs = calculate_throughput_of_topology(hybrid_dfs['invocations_df'])
        cloud_throughputs = calculate_throughput_of_topology(cloud_dfs['invocations_df'])

        workloads = ['constant', 'sine']
        type_runs = ['all', 'ga', 'skippy', 'vanilla']
        devices = ['Cloud', 'Hybrid', 'Edge Cloudlet']
        dfs_throughputs = list(zip([cloud_dfs, hybrid_dfs, edge_dfs],
                                   [cloud_throughputs, hybrid_throughputs, edge_throughputs]))
        for workload in workloads:
            for device_index, (dfs, throughput) in enumerate(dfs_throughputs):
                for type_run in type_runs:
                    df = dfs['fets_df']
                    df = df.loc[(df['workload'] == workload) & (df['type_run'] == type_run)]
                    for function in df['fets_df']['function_deployment'].unique():
                        df = df[df['function_deployment'] == function]
                        data['mean_fet'].append(df['t_fet_scaled'].mean())
                        data['std_fet'].append(df['t_fet_scaled'].std())
                        data['mean_wait'].append(df['t_wait_duration_scaled'].mean())
                        data['std_wait'].append(df['t_wait_duration_scaled'].std())
                        data['workload'].append(workload)
                        data['type_run'].append(type_run)
                        data['run'].append(index)
                        data['function'].append(function)
                        data['devices'].append(devices[device_index])
        del scaled
        del cloud_dfs
        del edge_dfs
        del hybrid_dfs
        del edge_throughputs
        del hybrid_throughputs
        del cloud_throughputs

    return pd.DataFrame(data=data)


def agg_fet(cloud_topology_folders, hybrid_folders, edge_cloudlet_topology_folders):
    data = {
        'mean_count': [],
        'mean_fet': [],
        'std_fet': [],
        'mean_wait': [],
        'std_wait': [],
        'run': [],
        'workload': [],
        'devices': [],
        'type_run': []
    }
    for index, (cloud_folder, hybrid_folder, edge_folder) in enumerate(
            zip(cloud_topology_folders, hybrid_folders, edge_cloudlet_topology_folders)):
        results = read([cloud_folder, hybrid_folder, edge_folder])
        scaled = preprocess_topology_results([results[0], results[1], results[2]])
        scaled_cloud = scaled[0]
        scaled_hybrid = scaled[1]
        scaled_edge = scaled[2]

        cloud_dfs = get_topology_dfs(scaled_cloud)
        hybrid_dfs = get_topology_dfs(scaled_hybrid)
        edge_dfs = get_topology_dfs(scaled_edge)

        edge_throughputs = calculate_throughput_of_topology(edge_dfs['invocations_df'])
        hybrid_throughputs = calculate_throughput_of_topology(hybrid_dfs['invocations_df'])
        cloud_throughputs = calculate_throughput_of_topology(cloud_dfs['invocations_df'])

        workloads = ['constant', 'sine']
        type_runs = ['all', 'ga', 'skippy', 'vanilla']
        devices = ['Cloud', 'Hybrid', 'Edge Cloudlet']
        functions = []
        dfs_throughputs = list(zip([cloud_dfs, hybrid_dfs, edge_dfs],
                                   [cloud_throughputs, hybrid_throughputs, edge_throughputs]))
        for workload in workloads:
            for device_index, (dfs, throughput) in enumerate(dfs_throughputs):
                for type_run in type_runs:
                    filtered = throughput.loc[(throughput['workload'] == workload)]
                    count_scaled_sum = filtered.groupby('type_run')['count_scaled'].sum() / len(
                        filtered['function_deployment'].unique())

                    df = dfs['fets_df']
                    df = df.loc[(df['workload'] == workload) & (df['type_run'] == type_run)]
                    data['mean_fet'].append(df['t_fet_scaled'].mean())
                    data['std_fet'].append(df['t_fet_scaled'].std())
                    data['mean_wait'].append(df['t_wait_duration_scaled'].mean())
                    data['std_wait'].append(df['t_wait_duration_scaled'].std())
                    data['workload'].append(workload)
                    data['type_run'].append(type_run)
                    data['run'].append(index)
                    data['mean_count'].append(count_scaled_sum[type_run])
                    data['devices'].append(devices[device_index])
        del scaled
        del cloud_dfs
        del edge_dfs
        del hybrid_dfs
        del edge_throughputs
        del hybrid_throughputs
        del cloud_throughputs

    return pd.DataFrame(data=data)


def normalize_deg_per_node_type(df: pd.DataFrame) -> pd.DataFrame:
    copy = df.copy()
    min_maxs = {}
    for node_type in copy['node_type'].unique():
        data = copy[copy['node_type'] == node_type]
        min_maxs[node_type] = (data['degradation'].min(), data['degradation'].max())

    copy['deg_min'] = copy['node_type'].apply(lambda l: min_maxs[l][0])
    copy['deg_max'] = copy['node_type'].apply(lambda l: min_maxs[l][1])

    copy['deg_scaled'] = scale(copy['degradation'], copy['deg_min'], copy['deg_max'])
    return copy


def calculate_statistic_degradation(interval: int, deg_col: str, df: pd.DataFrame) -> pd.DataFrame:
    copy = df.copy()
    nodes_degradation_mean = copy[deg_col].resample(f'{interval}s').mean()
    series = pd.Series(data=range(0, interval * len(nodes_degradation_mean), interval),
                       index=nodes_degradation_mean.index)
    nodes_degradation_std = copy[deg_col].resample(f'{interval}s').std()
    lower_bound = nodes_degradation_mean - nodes_degradation_std
    upper_bound = nodes_degradation_mean + nodes_degradation_std
    frame = {
        'degradation_mean': nodes_degradation_mean,
        'degradation_std': nodes_degradation_std,
        'degradation_lower_bound': lower_bound,
        'degradation_upper_bound': upper_bound,
        'workload': df['workload'].iloc[0],
        'type_run': df['type_run'].iloc[0],
        'devices': df['devices'].iloc[0],
        'ts_scaled': series
    }
    return pd.DataFrame(frame)


def calculate_statistic_degradation_topology_par_bound(results_by_workload):
    return calculate_statistic_degradation_topology(10, results_by_workload)


def calculate_statistic_degradation_topology_par(interval: int, workloads: List[
    Dict[str, Dict[str, SingleRunSimResult]]], normalize: bool = False) -> List[pd.DataFrame]:
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        for result in executor.map(calculate_statistic_degradation_topology_par_bound,
                                   workloads):
            results.append(result)
    return results


def calculate_statistic_degradation_topology(interval: int, results_by_workload:
Dict[str, Dict[str, SingleRunSimResult]], normalize: bool = False) -> pd.DataFrame:
    dfs = []
    for workload, results_by_type in results_by_workload.items():
        for type_run, result in results_by_type.items():
            fets = result.data['fets_df'].copy()
            if normalize:
                fets = normalize_deg_per_node_type(fets)
                df = calculate_statistic_degradation(interval, 'deg_scaled', fets)
            else:
                df = calculate_statistic_degradation(interval, 'degradation', fets)
            dfs.append(df)
    return pd.concat(dfs)


def calculate_statistic_degradation_multi_topology(interval: int, results_by_workload_list: List[
    Dict[str, Dict[str, SingleRunSimResult]]]) -> pd.DataFrame:
    dfs = []
    for results_by_workload in results_by_workload_list:
        for workload, results_by_type in results_by_workload.items():
            for type_run, result in results_by_type.items():
                fets = result.data['fets_df'].copy()
                fets['workload'] = workload
                fets['type_run'] = type_run
                fets['devices'] = pretty_string_device_file(result.single_run_ga_result.device_file)
                dfs.append(fets)
    concat = pd.concat(dfs)
    return calculate_statistic_degradation(interval, 'degradation', concat)
