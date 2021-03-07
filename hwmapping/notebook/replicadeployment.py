import pandas as pd


def get_total_and_unfinished_deployments_per_node_type(df: pd.DataFrame) -> pd.DataFrame:
    queued = get_total_deployments_per_node(df)
    summed = queued.groupby(['workload', 'type_run', 'node_type']).sum()
    diff_perc = summed[['value_d', 'unfinished']].apply(lambda row: ((row['unfinished']) / row['value_d']), axis=1)
    summed['unfinished_perc'] = diff_perc
    cols = ['workload', 'type_run', 'node_type', 'value_d', 'value', 'unfinished', 'unfinished_perc']
    return summed.reset_index()[cols].sort_values(by=['workload', 'type_run'])


def get_total_deployments_per_node(df: pd.DataFrame) -> pd.DataFrame:
    copy = df.copy()
    finish = copy[copy['value'] == 'finish']
    deploy = copy[copy['value'] == 'deploy']
    finish = finish.groupby(
        ['workload', 'type_run', 'node_type', 'node_name', 'function_name']).count()
    deploy = deploy.groupby(
        ['workload', 'type_run', 'node_type', 'node_name', 'function_name']).count()
    merged = deploy.join(finish, lsuffix='_d').fillna(value=0)

    diff = merged[['value_d', 'value']].apply(lambda row: (row['value_d'] - row['value']), axis=1)
    diff_perc = merged[['value_d', 'value']].apply(lambda row: 1 - (row['value'] / row['value_d']),
                                                   axis=1)
    merged['unfinished'] = diff
    merged['unfinished_perc'] = diff_perc

    return merged.sort_values(by=['node_type', 'node_name', 'function_name'])


def get_total_deployments_per_run(df: pd.DataFrame) -> pd.DataFrame:
    total_replica_lifecycle = get_total_deployments_per_node(df) \
        .groupby(['workload', 'type_run', 'function_name']).sum() \
        .groupby(['workload', 'type_run']).sum()
    total_replica_lifecycle['unfinished_perc'] = total_replica_lifecycle[['value_d', 'unfinished']] \
        .apply(lambda row: row['unfinished'] / row['value_d'], axis=1)
    return total_replica_lifecycle


def get_total_deployments_per_fn_run(df: pd.DataFrame) -> pd.DataFrame:
    total_replica_lifecycle = get_total_deployments_per_node(df) \
        .groupby(['workload', 'type_run', 'function_name']).sum()
    total_replica_lifecycle['unfinished_perc'] = total_replica_lifecycle[['value_d', 'unfinished']] \
        .apply(lambda row: row['unfinished'] / row['value_d'], axis=1)
    return total_replica_lifecycle
