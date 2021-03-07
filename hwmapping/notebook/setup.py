from skippy.core.storage import StorageIndex

from hwmapping.evaluation.functionsim import InterferenceAwarePythonHttpSimulatorFactory, PythonHttpSimulatorFactory
from hwmapping.evaluation.run import EnvSettings
from hwmapping.evaluation.topology import urban_sensing_topology


def setup_topology(sched_params, ether_nodes, topology=None):
    storage_index = StorageIndex()
    if topology is None:
        topology = urban_sensing_topology(ether_nodes, storage_index)

    simulator_factory = InterferenceAwarePythonHttpSimulatorFactory()
    # simulator_factory = PythonHttpSimulatorFactory()

    env_settings = EnvSettings(
        simulator_factory=simulator_factory,
        sched_params=sched_params,
        label_problem_solver_settings=None,
        storage_index=storage_index,
        null_logger=False,
        scale_by_requests=False,
        scale_by_resources=False,
        scale_by_requests_per_replica=False,
        scale_by_queue_requests_per_replica=True
    )

    return env_settings, topology
