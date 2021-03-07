import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sim.faassim import Simulation

from hwmapping.notebook.invocations import preprocess_invocations
from hwmapping.notebook.palettes import fn_palette, fn_deployment_palette, node_palette

def plot_invocations(invocations_df: pd.DataFrame,axs=None):
    plot_fns = [
        plot_line_start_exec,
        plot_scatter_exec_per_node_type,
        plot_ecdf_exec_per_node_type,
        plot_ecdf_exec_per_node,
        plot_scatter_exec_per_function,
        plot_ecdf_exec_fn_deployment,
        plot_kde_exec_fn_deployment,
        plot_ecdf_exec,
        plot_rps,
    ]

    n_plots = len(plot_fns)
    if axs is None:
        fig, axs = plt.subplots(n_plots)
        fig.set_figwidth(7)
        fig.set_figheight(45)

    for i in range(n_plots):
        plot_fn = plot_fns[i]
        plot_fn(invocations_df, ax=axs[i])
        # axs[i].legend(loc=2)


def plot_results(sim: Simulation):
    invocations_df = sim.env.metrics.extract_dataframe('invocations')
    scale_df = sim.env.metrics.extract_dataframe('scale')
    schedule_df = sim.env.metrics.extract_dataframe('schedule')
    replica_deployment_df = sim.env.metrics.extract_dataframe('replica_deployment')
    function_deployments_df = sim.env.metrics.extract_dataframe('function_deployments')
    function_deployment_df = sim.env.metrics.extract_dataframe('function_deployment')
    function_deployment_lifecycle_df = sim.env.metrics.extract_dataframe('function_deployment_lifecycle')
    functions_df = sim.env.metrics.extract_dataframe('functions')
    flow_df = sim.env.metrics.extract_dataframe('flow')
    network_df = sim.env.metrics.extract_dataframe('network')
    utilization_df = sim.env.metrics.extract_dataframe('utilization')
    invocations_df = preprocess_invocations(invocations_df)

    plot_invocations(invocations_df)

def plot_scatter_exec_per_function(invocations_df, **kwargs):
    sns.scatterplot(data=invocations_df, y='t_exec', x='t_start', hue='function_name', palette=fn_palette, **kwargs)


def plot_scatter_exec_per_node_type(invocations_df, **kwargs):
    sns.scatterplot(data=invocations_df, y='t_exec', x='t_start', hue='node_type', palette=node_palette, **kwargs)



def plot_line_start_exec(invocations_df, **kwargs):
    x = invocations_df[['t_start', 't_exec']]
    x.index = x.index - pd.to_timedelta(x['t_exec'], unit='s')
    y = x.t_start.resample('1s').count()
    ax = kwargs.get('ax',plt)
    ax.plot(y)


def plot_ecdf_exec_fn_deployment(invocations_df, **kwargs):
    sns.ecdfplot(data=invocations_df, x='t_exec', hue='function_deployment', palette=fn_deployment_palette, **kwargs)


def plot_kde_exec_fn_deployment(invocations_df, **kwargs):
    sns.kdeplot(data=invocations_df, x='t_exec', hue='function_deployment', palette=fn_deployment_palette, **kwargs)


def plot_ecdf_exec(invocations_df, **kwargs):
    sns.ecdfplot(data=invocations_df, x='t_exec', **kwargs)


def plot_rps(invocations_df, **kwargs):
    invocations_df['rounded_t_start'] = invocations_df['t_start'].apply(lambda v: int(v))
    rps_per_second_per_function = invocations_df.groupby(['function_deployment', 'rounded_t_start']).count()
    sns.scatterplot(data=rps_per_second_per_function, y='t_start', hue='function_deployment',
                    x=list(range(len(rps_per_second_per_function))), palette=fn_deployment_palette, **kwargs)

    plt.ylabel('Requests per second')
    plt.xlabel('Time in seconds')


def plot_ecdf_exec_per_node(invocations_df, **kwargs):
    sns.ecdfplot(data=invocations_df, x='t_exec', hue='node', **kwargs)


def plot_ecdf_exec_per_node_type(invocations_df, **kwargs):
    sns.ecdfplot(data=invocations_df, x='t_exec', hue='node_type', palette=node_palette, **kwargs)
