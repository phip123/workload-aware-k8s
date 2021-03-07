from sim.faas import FunctionReplica, FunctionRequest
from sim.metrics import Metrics
from skippy.core.model import SchedulingResult

from hwmapping.faas.system import FunctionDeployment


class OurMetrics(Metrics):

    def log_function_deployment(self, fn: FunctionDeployment):
        """
        Logs the functions name, related container images and their metadata
        """
        record = {'name': fn.name, 'image': fn.image}
        self.log('function_deployments', record, type='deploy')

    def log_invocation(self, function_deployment, function_name, node_name, t_wait, t_start, t_exec, replica_id):
        function = self.env.faas.functions[function_name]
        mem = function.get_resource_requirements().get('memory')

        self.log('invocations', {'t_wait': t_wait, 't_exec': t_exec, 't_start': t_start, 'memory': mem},
                 function_deployment=function_deployment,
                 function_name=function_name, node=node_name, replica_id=replica_id)

    def log_fet(self, function_deployment, function_name, node_name, t_fet_start, t_fet_end, t_wait_start, t_wait_end,
                degradation,
                replica_id):
        self.log('fets', {'t_fet_start': t_fet_start, 't_fet_end': t_fet_end, 't_wait_start': t_wait_start,
                          't_wait_end': t_wait_end, 'degradation': degradation},
                 function_deployment=function_deployment,
                 function_name=function_name, node=node_name, replica_id=replica_id)

    def log_start_exec(self, request: FunctionRequest, replica: FunctionReplica):
        self.invocations[replica.function.name] += 1
        self.total_invocations += 1
        self.last_invocation[replica.function.name] = self.env.now

        node = replica.node
        function = replica.function

        for resource, value in function.get_resource_requirements().items():
            self.utilization[node.name][resource] += value

        self.log('utilization', {
            'cpu': self.utilization[node.name]['cpu'] / node.ether_node.capacity.cpu_millis,
            'mem': self.utilization[node.name]['memory'] / node.ether_node.capacity.memory
        }, node=node.name)

    def log_function_deployment_lifecycle(self, fn: FunctionDeployment, event: str):
        self.log('function_deployment_lifecycle', event, name=fn.name, function_id=id(fn))

    def log_queue_schedule(self, replica: FunctionReplica):
        self.log('schedule', 'queue', function_name=replica.function.name, image=replica.function.image,
                 replica_id=id(replica))

    def log_start_schedule(self, replica: FunctionReplica):
        self.log('schedule', 'start', function_name=replica.function.name, image=replica.function.image,
                 replica_id=id(replica))

    def log_finish_schedule(self, replica: FunctionReplica, result: SchedulingResult):
        if not result.suggested_host:
            node_name = 'None'
        else:
            node_name = result.suggested_host.name

        self.log('schedule', 'finish', function_name=replica.function.name, image=replica.function.image,
                 node_name=node_name,
                 successful=node_name != 'None', replica_id=id(replica))

    def log_function_deploy(self, replica: FunctionReplica):
        fn = replica.function
        self.log('function_deployment', 'deploy', name=fn.name, image=fn.image, function_id=id(fn),
                 node=replica.node.name)

    def log_function_suspend(self, replica: FunctionReplica):
        fn = replica.function
        self.log('function_deployment', 'suspend', name=fn.name, image=fn.image, function_id=id(fn),
                 node=replica.node.name)

    def log_function_remove(self, replica: FunctionReplica):
        fn = replica.function
        self.log('function_deployment', 'remove', name=fn.name, image=fn.image, function_id=id(fn),
                 node=replica.node.name)
