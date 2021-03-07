import pandas as pd


def get_total_and_failed_schedules(df: pd.DataFrame) -> pd.DataFrame:
    queued = get_total_failed_schedules_per_fn(df)
    summed = queued.groupby(['workload', 'type_run', 'devices']).sum()
    diff_perc = summed[['value', 'failed']].apply(lambda row: ((row['failed']) / row['value']),
                                                  axis=1)
    summed['failed_perc'] = diff_perc
    cols = ['workload', 'type_run', 'value', 'failed', 'devices', 'failed_perc']
    return summed.reset_index()[cols].sort_values(by=['workload', 'type_run', 'devices'])


def get_total_failed_schedules_per_fn(df: pd.DataFrame) -> pd.DataFrame:
    copy = df.copy()
    successful = copy[copy['successful'] == True]
    successful = successful[successful['value'] == 'finish'].groupby(
        ['workload', 'type_run', 'function_name', 'devices']).count()
    queued = copy[copy['value'] == 'queue'].groupby(['workload', 'type_run', 'function_name', 'devices']).count()
    joined_s_q = successful.join(queued, lsuffix='_s')
    # value_s == successful, value == all => value_s <= value
    diff = joined_s_q[['value_s', 'value']].apply(lambda row: (row['value'] - row['value_s']), axis=1)
    diff_perc = joined_s_q[['value_s', 'value']].apply(lambda row: (1 - (row['value_s']) / row['value']),
                                                       axis=1)
    queued['failed'] = diff
    queued['failed_perc'] = diff_perc
    return queued


def get_total_and_failed_schedules_per_node_type(df: pd.DataFrame) -> pd.DataFrame:
    queued = get_total_failed_schedules_per_fn(df)
    summed = queued.groupby(['workload', 'type_run', 'devices']).sum()
    cols = ['workload', 'type_run', 'value', 'failed', 'devices', 'failed_perc']
    return summed.reset_index()[cols].sort_values(by=['workload', 'type_run', 'devices'])


def get_total_failed_schedules_per_node(df: pd.DataFrame) -> pd.DataFrame:
    copy = df.copy()
    successful = copy[copy['successful'] == True]
    successful = successful[successful['value'] == 'finish'].groupby(
        ['workload', 'type_run', 'function_name', 'devices']).count()
    queued = copy[copy['value'] == 'queue'].groupby(['workload', 'type_run', 'function_name', 'devices']).count()
    joined_s_q = successful.join(queued, lsuffix='_s')
    # value_s == successful, value == all => value_s <= value
    diff = joined_s_q[['value_s', 'value']].apply(lambda row: (row['value'] - row['value_s']), axis=1)
    diff_perc = joined_s_q[['value_s', 'value']].apply(lambda row: 1 - (row['value_s'] / row['value']),
                                                       axis=1)
    queued['failed'] = diff
    queued['failed_perc'] = diff_perc / len(df['function_name'].unique())
    return queued
