import concurrent.futures
import datetime
from collections import defaultdict
from typing import Dict, Tuple, List

import pandas as pd

from hwmapping.evaluation.resources import resources_per_node_image as measured_resources_per_node_image
from hwmapping.evaluation.results import SingleRunSimResult
from hwmapping.faas.system import FunctionResourceCharacterization
from hwmapping.notebook.results import scale
from hwmapping.notebook.utils import pretty_string_device_file

resource_columns = ['cpu', 'gpu', 'blkio', 'net', 'ram']


def add_resources(row, resources_per_node_image: Dict[Tuple[str, str], FunctionResourceCharacterization]):
    node_type = row['node_type']
    image = row['image']
    key = (node_type, image)
    characterization = resources_per_node_image[key]

    return pd.Series([characterization[x] for x in resource_columns
                      ])


def sum_utils(row):
    sum = 0
    for resource in resource_columns:
        sum += row[resource]
    return sum


def scale_and_total_util(df: pd.DataFrame) -> pd.DataFrame:
    copy = df.copy()
    for resource in resource_columns:
        amin = df[resource].min()
        amax = df[resource].max()
        min_col = f'{resource}_min'
        max_col = f'{resource}_max'
        copy[min_col] = amin
        copy[max_col] = amax
        copy[f'{resource}_scaled'] = scale(copy[resource], copy[min_col], copy[max_col])

    copy['total_util'] = copy.apply(sum_utils, axis=1)
    min_total = copy['total_util'].min()
    max_total = copy['total_util'].max()
    copy['total_util_min'] = min_total
    copy['total_util_max'] = max_total
    copy['total_util_scaled'] = scale(copy['total_util'], copy['total_util_min'], copy['total_util_max'])
    return copy


def read_topology_utils_par(interval, resources, workloads):
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        for result in executor.map(lambda f: create_topology_util(interval, f, resources), workloads):
            results.append(result)
    return results


def create_all_topology_util_par(interval: int, results_by_workload_list: List[
    Dict[str, Dict[str, SingleRunSimResult]]], resources_per_node_image=None) -> pd.DataFrame:
    if resources_per_node_image is None:
        resources_per_node_image = measured_resources_per_node_image
    concat = pd.concat(read_topology_utils_par(interval, resources_per_node_image, results_by_workload_list))
    return scale_and_total_util(concat)


def create_all_topology_util(interval: int, results_by_workload_list: List[
    Dict[str, Dict[str, SingleRunSimResult]]], resources_per_node_image=None) -> pd.DataFrame:
    dfs = []
    if resources_per_node_image is None:
        resources_per_node_image = measured_resources_per_node_image
    for results_by_workload in results_by_workload_list:
        dfs.append(create_topology_util(interval, results_by_workload, resources_per_node_image))
    concat = pd.concat(dfs)
    return scale_and_total_util(concat)


def create_topology_util(interval: int, results_by_workload: Dict[str, Dict[str, SingleRunSimResult]],
                         resources_per_node_image=None) -> pd.DataFrame:
    if resources_per_node_image is None:
        resources_per_node_image = measured_resources_per_node_image
    dfs = []
    for workload, results_by_type in results_by_workload.items():
        for type_run, result in results_by_type.items():
            df = create_utilization_df(interval, result, resources_per_node_image)
            dfs.append(df)
    return scale_and_total_util(pd.concat(dfs))


def create_utilization_df(interval: int, result: SingleRunSimResult,
                          resources_per_node_image: Dict[
                              Tuple[str, str], FunctionResourceCharacterization] = None) -> pd.DataFrame:
    if resources_per_node_image is None:
        resources_per_node_image = measured_resources_per_node_image
    replica_deployment_df = result.data['replica_deployment_df']
    # replica_deployment_df = replica_deployment_df[replica_deployment_df['value'].isin(['finish', 'teardown'])]
    columns = ['cpu', 'gpu', 'blkio', 'net', 'ram']
    replica_deployment_df[columns] = replica_deployment_df.apply(
        lambda row: add_resources(row, resources_per_node_image), axis=1)
    start = result.data['function_deployments_df'].iloc[0].name
    end = result.data['invocations_df']['ts_end'].max()
    data = defaultdict(list)
    index = 0

    for node in replica_deployment_df['node_name'].unique():
        df = replica_deployment_df[replica_deployment_df['node_name'] == node]
        df = df[df['value'].isin(['deploy', 'teardown'])]
        node_type = df['node_type'].iloc[0]
        i = start
        next_i = i
        ts_index = 0
        data['cpu'].append(0)
        data['gpu'].append(0)
        data['blkio'].append(0)
        data['net'].append(0)
        data['ram'].append(0)
        data['containers'].append(0)
        data['ts'].append(i)
        data['ts_scaled'].append(ts_index * interval)
        data['node'].append(node)
        data['node_type'].append(node_type)
        index += 1
        ts_index += 1
        while i < end:
            next_i += datetime.timedelta(seconds=interval)
            # the extra 100ms prevent that one event gets chosen twice, in case the time == i
            between = df.loc[i + datetime.timedelta(microseconds=100):next_i]
            deploys = between[between['value'] == 'deploy']
            teardowns = between[between['value'] == 'teardown']
            cpu = data['cpu'][index - 1]
            gpu = data['gpu'][index - 1]
            blkio = data['blkio'][index - 1]
            net = data['net'][index - 1]
            ram = data['ram'][index - 1]
            containers = data['containers'][index - 1]
            for ts, row in deploys.iterrows():
                cpu += row['cpu']
                gpu += row['gpu']
                blkio += row['blkio']
                net += row['net']
                ram += row['ram']
                containers += 1
            for ts, row in teardowns.iterrows():
                cpu -= row['cpu']
                gpu -= row['gpu']
                blkio -= row['blkio']
                net -= row['net']
                ram -= row['ram']
                containers -= 1
            data['cpu'].append(cpu)
            data['gpu'].append(gpu)
            data['blkio'].append(blkio)
            data['net'].append(net)
            data['ram'].append(ram)
            data['containers'].append(containers)
            data['ts'].append(next_i)
            data['ts_scaled'].append(ts_index * interval)
            data['node'].append(node)
            data['node_type'].append(node_type)
            i = next_i
            index += 1
            ts_index += 1

    df = pd.DataFrame(data=data)
    df['type_run'] = result.type_run
    df['workload'] = result.type_bench
    df['devices'] = pretty_string_device_file(result.single_run_ga_result.device_file)
    return df
