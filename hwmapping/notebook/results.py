import datetime
from typing import Tuple, Dict

import pandas as pd

from hwmapping.evaluation.results import SingleRunSimResult, SingleRunGaResult
from hwmapping.notebook.invocations import preprocess_invocations, preprocess_schedules, strip_node_id
from hwmapping.notebook.utils import scale, pretty_string_device_file


def calc_t_fet(df: pd.DataFrame) -> pd.DataFrame:
    copy = df.copy()
    copy['t_fet'] = copy['t_fet_end'] - copy['t_fet_start']
    return copy


def scale_col(df: pd.DataFrame, min_maxs: Dict[str, Tuple[float, float]], col) -> pd.DataFrame:
    copy = df.copy()
    # copy = copy.apply(lambda l: scale(l, min_maxs), axis=1)
    max_col = f'max_{col}'
    min_col = f'min_{col}'
    # copy[min_col] = copy['function_deployment'].apply(lambda l: min_maxs[l][0])
    copy[min_col] = 0
    copy[max_col] = copy['function_deployment'].apply(lambda l: min_maxs[l][1])
    copy[f'{col}_scaled'] = scale(copy[col], copy[min_col], copy[max_col])
    return copy


def calc_wait_duration(df: pd.DataFrame) -> pd.DataFrame:
    if 't_wait_start' not in df.columns:
        return df
    copy = df.copy()
    copy['t_wait_duration'] = copy['t_wait_end'] - copy['t_wait_start']
    return copy


def preprocess_fets(df: pd.DataFrame, run_result: SingleRunSimResult) -> pd.DataFrame:
    updated = calc_t_fet(df)
    updated = calc_wait_duration(updated)
    updated = strip_node_id(updated, 'node')
    updated['degradation_orig'] = updated['degradation'].copy()
    updated['degradation'] = updated['degradation'].apply(lambda x: x if x >= 1 else 1)
    updated['workload'] = run_result.type_bench
    updated['type_run'] = run_result.type_run
    updated['devices'] = pretty_string_device_file(run_result.single_run_ga_result.device_file)
    return updated


def append_image(replica_id, schedule_df):
    filtered = schedule_df[schedule_df['successful'] == True]
    return filtered[filtered['replica_id'] == replica_id].iloc[0]['image']


def append_fn_image(df: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    copy = df.copy()
    copy['image'] = copy['replica_id'].apply(lambda row: append_image(row, schedule_df))
    return copy


def preprocess_replica_deployment(df: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    updated = strip_node_id(df, 'node_name')
    updated = append_fn_image(updated, schedule_df)
    return updated


def preprocess_sim_result(result: SingleRunSimResult, duration: int = None) -> SingleRunSimResult:
    start = result.data['function_deployments_df'].iloc[0].name

    result.data['invocations_df'] = preprocess_invocations(result.data['invocations_df'], start)
    result.data['invocations_df']['workload'] = result.type_bench
    result.data['invocations_df']['type_run'] = result.type_run
    result.data['invocations_df']['devices'] = pretty_string_device_file(result.single_run_ga_result.device_file)

    result.data['schedule_df']['workload'] = result.type_bench
    result.data['schedule_df']['type_run'] = result.type_run
    result.data['schedule_df']['devices'] = pretty_string_device_file(result.single_run_ga_result.device_file)


    result.data['schedule_df'] = preprocess_schedules(result.data['schedule_df'])
    result.data['replica_deployment_df'] = preprocess_replica_deployment(result.data['replica_deployment_df'],
                                                                         result.data['schedule_df'])
    if result.data.get('fets_df', None) is not None:
        result.data['fets_df'] = preprocess_fets(result.data['fets_df'], result)

    if duration is not None:
        first_invocation = result.data['invocations_df']['t_start'].iloc[0]
        stop = first_invocation + duration
        stop_ts = start + datetime.timedelta(seconds=stop)
        df_names = list(result.data.keys())
        for df_name in df_names:
            result.data[df_name] = result.data[df_name].loc[:stop_ts]

    return result


def preprocess_ga_result(result: SingleRunGaResult) -> SingleRunGaResult:
    return result
