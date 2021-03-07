from collections import defaultdict

import pandas as pd


def count_mean_containers(util_df, maximum):
    data_dict = defaultdict(list)
    a = util_df.groupby(['type_run', 'node']) \
        .mean() \
        .sort_values(by='containers', ascending=False)
    for type_run in ['all', 'ga', 'skippy', 'vanilla']:
        data = a.loc[type_run]
        for minimum in range(2, maximum + 1):
            data_dict['type_run'].append(type_run)
            data_dict['minimum'].append(minimum)
            data_dict['containers'].append(len(data[data['containers'] >= minimum]))
    return pd.DataFrame(data=data_dict)


def count_mean_containers_per_node_type(util_df):
    data_dict = defaultdict(list)
    df = util_df[util_df['containers'] != 0]
    a = df.groupby(['type_run', 'node_type']) \
        .median()
    for type_run in ['all', 'ga', 'skippy', 'vanilla']:
        data = a.loc[type_run]
        data = data.reset_index()
        for row in data.iterrows():
            data_dict['type_run'].append(type_run)
            data_dict['median'].append(row[1]['containers'])
            data_dict['node_type'].append(row[1]['node_type'])
    return pd.DataFrame(data=data_dict)
