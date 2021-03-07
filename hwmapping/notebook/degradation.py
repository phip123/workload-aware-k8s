from typing import List

import pandas as pd

from hwmapping.notebook.fet import calculate_statistic_degradation_topology_par
from hwmapping.notebook.reader.simreader import read_topologies_par


def agg_degradation_utilization(folders: List[str]) -> pd.DataFrame:
    read_folders = read_topologies_par([
        folders[0],
        folders[1],
        folders[2]
    ])
    cloud_topology_results = read_folders[0]

    hybrid_topology_results = read_folders[1]

    edge_cloudlet_topology_results = read_folders[2]

    degs = calculate_statistic_degradation_topology_par(
        10,
        [
            cloud_topology_results,
            hybrid_topology_results,
            edge_cloudlet_topology_results
        ]
    )
    cloud_deg = degs[0]

    hybrid_deg = degs[1]

    edge_cloudlet_deg = degs[2]

    dfs = []
    dfs.append(cloud_deg.groupby(['type_run', 'workload', 'devices']).mean()[['degradation_mean', 'degradation_std']])
    dfs.append(hybrid_deg.groupby(['type_run', 'workload', 'devices']).mean()[['degradation_mean', 'degradation_std']])
    dfs.append(
        edge_cloudlet_deg.groupby(['type_run', 'workload', 'devices']).mean()[['degradation_mean', 'degradation_std']])
    del cloud_deg
    del hybrid_deg
    del edge_cloudlet_deg
    del cloud_topology_results
    del hybrid_topology_results
    del edge_cloudlet_topology_results
    return pd.concat(dfs)


