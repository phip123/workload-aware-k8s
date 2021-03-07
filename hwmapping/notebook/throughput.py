from typing import List

import pandas as pd

from hwmapping.notebook.utils import scale


def calculate_scaled_throughput(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    counted = []
    for index, df in enumerate(dfs):
        count = df.groupby('function_deployment').count()
        count['count'] = count.apply(lambda r: r['function_name'], axis=1)
        count = count.drop(labels=[x for x in df.columns if x != 'function_deployment'], axis=1)
        if 'id' in df.columns:
            count['id'] = df['id'].iloc[0]
        else:
            count['id'] = index
        count['workload'] = df['workload'].iloc[0]
        count['devices'] = df['devices'].iloc[0]
        count['type_run'] = df['type_run'].iloc[0]
        counted.append(count)

    concat = pd.concat(counted)
    maxs = {}
    for service in dfs[0]['function_deployment'].unique():
        maxs[service] = concat.loc[service]['count'].max()
    concat = concat.reset_index()

    concat['min'] = 0
    concat['max'] = concat.apply(lambda r: maxs[r['function_deployment']], axis=1)
    concat['count_scaled'] = scale(concat['count'], concat['min'], concat['max'])
    return concat


def calculate_throughput_of_topology(df: pd.DataFrame) -> pd.DataFrame:
    throughputs = []
    for workload in ['constant', 'sine']:
        dfs = []
        for type_run in ['ga', 'vanilla', 'skippy', 'all']:
            data = df.loc[(df['workload'] == workload) & (df['type_run'] == type_run)]
            dfs.append(data)
        throughput = calculate_scaled_throughput(dfs)
        throughputs.append(throughput)
    return pd.concat(throughputs)
