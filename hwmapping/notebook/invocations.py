import datetime

import pandas as pd

from hwmapping.util import extract_model_type


def strip_node_id(df: pd.DataFrame, node_col: str) -> pd.DataFrame:
    copy = df.copy()

    copy['node_type'] = copy[node_col].apply(lambda n: extract_model_type(n))
    return copy


def calc_t_end(df: pd.DataFrame) -> pd.DataFrame:
    copy = df.copy()
    copy['t_end'] = copy['t_start'] + copy['t_exec']
    return copy


def add_time(row, start):
    return start + datetime.timedelta(seconds=row['t_end'])


def calc_ts_end(df: pd.DataFrame, start) -> pd.DataFrame:
    copy = df.copy()
    copy['ts_end'] = copy.apply(lambda row: add_time(row, start), axis=1)
    return copy


def preprocess_invocations(df: pd.DataFrame, start, stop=None) -> pd.DataFrame:
    updated = strip_node_id(df, 'node')
    updated = calc_t_end(updated)
    updated = calc_ts_end(updated, start)
    if stop is not None:
        updated = updated[updated['t_end'] <= stop]
    return updated


def preprocess_schedules(df: pd.DataFrame, stop=None) -> pd.DataFrame:
    updated = strip_node_id(df, 'node_name')
    if stop is not None:
        updated = updated.loc[:stop]
    return updated
