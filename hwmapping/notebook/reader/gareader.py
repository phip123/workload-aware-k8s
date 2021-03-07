import glob
import os
import pickle
from collections import defaultdict
from typing import List, Tuple

import pandas as pd

from hwmapping.calculations import calculate_diff_entropy, calculate_requirements
from hwmapping.evaluation.results import SingleRunGaResult
from hwmapping.problem.ga import base_requirements


def format_files(df) -> pd.DataFrame:
    file = os.path.basename(df['device_file'])
    index = file.rindex('_score')
    pkl_index = file.rindex('.pkl')
    df['devices'] = file[:index]
    df['devices_score'] = file[index + len('_score_'):pkl_index]

    file = os.path.basename(df['cluster_file'])
    index = file.rindex('.json')
    df['clustering'] = file[:index]
    split = file.split('_')
    df['no_clusters'] = int(split[1])

    return df


def scale_het(row):
    return (row - 0) / (16 - 0)


def add_scores(r):
    r['sum_scores'] = r['scaled_het'] + r['mean_fet']
    return r


def preprocess_single_run(run) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_result = run
    mean_fets = [x.mean_fet for x in run_result.results]
    clusters = list(run_result.clustering.values())
    cluster_ids = [x.id for x in clusters]
    result_data = defaultdict(list)

    for result, cluster in zip(run_result.results, clusters):
        result_data['function'].append(result.output_dict['function'])
        result_data['mean_fet'].append(result.mean_fet)
        result_data['cluster'].append(cluster.id)
        result_data['duration'].append(result.duration)
        result_data['het_score'].append(calculate_heterogeneity(result.devices))
        device_ids = set([x.id for x in result.devices])
        result_data['het_score_filtered'].append(
            calculate_heterogeneity([x for x in result.instance.filtered_devices if x.id in device_ids]))
        result_data['het_score_orig'].append(calculate_heterogeneity(result.instance.filtered_devices))
        result_data['cluster_file'].append(run.clustering_file)
        result_data['device_file'].append(run.device_file)
        result_data['performance_weight'].append(result.instance.performance_weight)
        result_data['variety_weight'].append(result.instance.variety_weight)

    result_df = pd.DataFrame(data=result_data)

    meta_clustering_data = defaultdict(list)
    for cluster in clusters:
        for function_def in cluster.function_definitions:
            meta_clustering_data['cluster'].append(cluster.id)
            meta_clustering_data['image'].append(function_def.image)

    meta_clustering_df = pd.DataFrame(data=meta_clustering_data)
    meta_clustering_df = meta_clustering_df.sort_values(by='cluster')
    return meta_clustering_df, result_df


def preprocess_ga_result(result: SingleRunGaResult) -> Tuple[SingleRunGaResult, pd.DataFrame, pd.DataFrame]:
    meta, result_df = preprocess_single_run(result)
    return result, meta, result_df


def round_values(df):
    df['duration'] = round(df['duration'], 2)
    df['function'] = round(df['function'], 2)
    df['mean_fet'] = round(df['mean_fet'], 3)
    df['het_score'] = round(df['het_score'], 2)
    df['sum_scores'] = round(df['sum_scores'], 3)
    return df


def preprocess_result_df(result_df: pd.DataFrame) -> pd.DataFrame:
    copy = result_df.copy()
    copy = copy.apply(lambda l: format_files(l), axis=1)
    copy['scaled_het'] = copy['het_score'].apply(lambda r: scale_het(r))
    copy = copy.apply(lambda r: add_scores(r), axis=1)
    copy = copy.sort_values(by=['sum_scores'])
    copy = round_values(copy)
    return copy


def read_ga_results(folder: str, profile: str) -> List[Tuple[str, SingleRunGaResult, pd.DataFrame, pd.DataFrame]]:
    path = os.path.join(folder, profile, 'ga_results')
    results = []
    for fname in glob.glob(f'{path}/*'):
        with open(fname, 'rb') as fd:
            result = pickle.load(fd)
            result, meta, result_df = preprocess_ga_result(result)
            result_df = preprocess_result_df(result_df)
            results.append((fname, result, meta, result_df))
    return results


def calculate_heterogeneity(devices):
    requirements = calculate_requirements(devices)
    return calculate_diff_entropy(base_requirements(), requirements)
