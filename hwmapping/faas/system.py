import logging
import math
import time
from collections import Counter
from typing import List, Dict, Optional
import numpy as np
from sim.core import Environment
from sim.faas import Resources, FaasSystem, FunctionRequest, FunctionState, simulate_function_invocation, \
    FunctionReplica, simulate_function_start

from hwmapping.evaluation.oracle import ResourceOracle, FetOracle

logger = logging.getLogger(__name__)


class FunctionResourceCharacterization:
    cpu: float
    blkio: float
    gpu: float
    net: float
    ram: float

    def __init__(self, cpu: float, blkio: float, gpu: float, net: float, ram: float):
        self.cpu = cpu
        self.blkio = blkio
        self.gpu = gpu
        self.net = net
        self.ram = ram

    def __len__(self):
        return 5

    def __delitem__(self, key):
        self.__delattr__(key)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)


class FunctionCharacterization:

    def __init__(self, image: str, fet_oracle: FetOracle, resource_oracle: ResourceOracle):
        self.image = image
        self.fet_oracle = fet_oracle
        self.resource_oracle = resource_oracle

    def sample_fet(self, host: str) -> Optional[float]:
        return self.fet_oracle.sample(host, self.image)

    def get_resources_for_node(self, host: str) -> FunctionResourceCharacterization:
        return self.resource_oracle.get_resources(host, self.image)


class DeploymentRanking:
    images: List[str] = ['tpu', 'gpu', 'cpu']

    def __init__(self, images: List[str]):
        self.images = images

    def set_first(self, image: str):
        index = self.images.index(image)
        updated = self.images[:index] + self.images[index + 1:]
        self.images = [image] + updated

    def get_first(self):
        return self.images[0]


class FunctionDefinition:
    # TODO is this useful on this level? can we say something about the other architectures?
    # or would it be too cumbersome to provide this for each image?
    requests: Resources = Resources()

    name: str

    # the manifest list name
    image: str

    # characterization per image
    characterization: FunctionCharacterization

    labels: Dict[str, str]

    def __init__(self, name: str, image: str, characterization: FunctionCharacterization,
                 labels: Dict[str, str]):
        self.name = name
        self.image = image
        self.characterization = characterization
        self.labels = labels

    def get_resource_requirements(self) -> Dict:
        return {
            'cpu': self.requests.cpu,
            'memory': self.requests.memory
        }

    def sample_fet(self, host: str) -> Optional[float]:
        return self.characterization.sample_fet(host)

    def get_resources_for_node(self, host: str):
        return self.characterization.get_resources_for_node(host)


class FunctionDeployment:
    name: str
    function_definitions: Dict[str, FunctionDefinition]

    # used to determine which function to take when scaling
    ranking: DeploymentRanking

    scale_min: int = 1
    scale_max: int = 20
    scale_factor: int = 1
    scale_zero: bool = False

    # percentages of scaling per image, can be used to hinder scheduler to overuse expensive resources (i.e. tpu)
    function_factor: Dict[str, float]

    # average requests per second threshold for scaling
    rps_threshold: int = 20

    # window over which to track the average rps
    alert_window: int = 50  # TODO currently not supported by FaasRequestScaler

    # seconds the rps threshold must be violated to trigger scale up
    rps_threshold_duration: int = 10

    # target average cpu utilization of all replicas, used by HPA
    target_average_utilization: float = 0.5

    # target average rps over all replicas, used by AverageFaasRequestScaler
    target_average_rps: int = 200

    # target of maximum requests in queue
    target_queue_length: int = 75

    target_average_rps_threshold = 0.1

    def __init__(self, name: str, function_definitions: Dict[str, FunctionDefinition], ranking: DeploymentRanking):
        self.name = name
        self.image = name  #
        self.function_definitions = function_definitions
        self.ranking = ranking

    def get_selected_service(self):
        return self.function_definitions[self.ranking.get_first()]

    def get_services(self):
        return list(map(lambda i: self.function_definitions[i], self.ranking.images))


class OurFaas(FaasSystem):

    def __init__(self, env: Environment, scale_by_requests: bool = False,
                 scale_by_average_requests: bool = False, scale_by_queue_requests_per_replica: bool = False) -> None:
        super().__init__(env)
        self.scale_by_requests = scale_by_requests
        self.scale_by_average_requests_per_replica = scale_by_average_requests
        self.scale_by_queue_requests_per_replica = scale_by_queue_requests_per_replica
        self.functions_deployments: Dict[str, FunctionDeployment] = dict()
        self.faas_scalers: Dict[str, FaasRequestScaler] = dict()
        self.avg_faas_scalers: Dict[str, AverageFaasRequestScaler] = dict()
        self.queue_faas_scalers: Dict[str, AverageQueueFaasRequestScaler] = dict()
        self.replica_count: Dict[str, int] = dict()
        self.functions_definitions = Counter()

    def deploy(self, fn: FunctionDeployment):
        if fn.name in self.functions_deployments:
            raise ValueError('function already deployed')

        self.functions_deployments[fn.name] = fn
        self.faas_scalers[fn.name] = FaasRequestScaler(fn, self.env)
        self.avg_faas_scalers[fn.name] = AverageFaasRequestScaler(fn, self.env)
        self.queue_faas_scalers[fn.name] = AverageQueueFaasRequestScaler(fn, self.env)

        if self.scale_by_requests:
            self.env.process(self.faas_scalers[fn.name].run())
        if self.scale_by_average_requests_per_replica:
            self.env.process(self.avg_faas_scalers[fn.name].run())
        if self.scale_by_queue_requests_per_replica:
            self.env.process(self.queue_faas_scalers[fn.name].run())

        for name, f in fn.function_definitions.items():
            self.functions[name] = f

        self.env.metrics.log_function_deployment(fn)
        self.env.metrics.log_function_deployment_lifecycle(fn, 'deploy')
        logger.info('deploying function %s with scale_min=%d', fn.name, fn.scale_min)
        yield from self.scale_up(fn.name, fn.scale_min)

    def suspend(self, function_name: str):
        if function_name not in self.functions:
            raise ValueError

        function: FunctionDefinition = self.functions[function_name]
        replicas: List[FunctionReplica] = self.discover(function)

        for replica in replicas:
            yield from self._remove_replica(replica)

        self.env.metrics.log_function_deployment_lifecycle(self.functions_deployments[function_name], 'suspend')

    def invoke(self, request: FunctionRequest):
        # TODO: how to return a FunctionResponse?
        logger.debug('invoking function %s', request.name)

        if request.name not in self.functions_deployments.keys():
            logger.warning('invoking non-existing function %s', request.name)
            return

        t_received = self.env.now

        replicas = self.get_replicas(request.name, FunctionState.RUNNING)
        if not replicas:
            '''
            https://docs.openfaas.com/architecture/autoscaling/#scaling-up-from-zero-replicas

            When scale_from_zero is enabled a cache is maintained in memory indicating the readiness of each function.
            If when a request is received a function is not ready, then the HTTP connection is blocked, the function is
            scaled to min replicas, and as soon as a replica is available the request is proxied through as per normal.
            You will see this process taking place in the logs of the gateway component.
            '''
            yield from self.poll_available_replica(request.name)

        if len(replicas) < 1:
            raise ValueError
        elif len(replicas) > 1:
            logger.debug('asking load balancer for replica for request %s:%d', request.name, request.request_id)
            replica = self.next_replica(request)
        else:
            replica = replicas[0]

        logger.debug('dispatching request %s:%d to %s', request.name, request.request_id, replica.node.name)

        t_start = self.env.now

        yield from simulate_function_invocation(self.env, replica, request)

        t_end = self.env.now

        t_wait = t_start - t_received
        t_exec = t_end - t_start

        self.env.metrics.log_invocation(request.name, replica.function.image, replica.node.name, t_wait, t_start,
                                        t_exec,
                                        id(replica))

    def remove(self, fn: FunctionDeployment):
        self.env.metrics.log_function_deployment_lifecycle(fn, 'remove')

        replica_count = self.replica_count[fn.name]
        yield from self.scale_down(fn.name, replica_count)
        self.faas_scalers[fn.name].stop()
        self.avg_faas_scalers[fn.name].stop()
        self.queue_faas_scalers[fn.name].stop()

        del self.functions_deployments[fn.name]
        del self.faas_scalers[fn.name]
        del self.avg_faas_scalers[fn.name]
        del self.queue_faas_scalers[fn.name]
        del self.replica_count[fn.name]

    def scale_down(self, fn_name: str, remove: int):
        replica_count = len(self.get_replicas(fn_name, FunctionState.RUNNING))
        if replica_count == 0:
            return
        replica_count -= remove
        if replica_count <= 0:
            remove = remove + replica_count

        scale_min = self.functions_deployments[fn_name].scale_min
        if self.replica_count.get(fn_name, 0) - remove < scale_min:
            remove = self.replica_count.get(fn_name, 0) - scale_min

        if replica_count - remove <= 0 or remove == 0:
            return

        logger.info(f'scale down {fn_name} by {remove}')
        replicas = self.choose_replicas_to_remove(fn_name, remove)
        self.env.metrics.log_scaling(fn_name, -remove)
        self.replica_count[fn_name] -= remove
        for replica in replicas:
            yield from self._remove_replica(replica)
            replicas.remove(replica)

    def choose_replicas_to_remove(self, fn_name: str, n: int):
        # TODO implement more sophisticated, currently just picks last ones deployed
        running_replicas = self.get_replicas(fn_name, FunctionState.RUNNING)
        return running_replicas[len(running_replicas) - n:]

    def scale_up(self, fn_name: str, replicas: int):
        fn = self.functions_deployments[fn_name]
        scale = replicas
        if self.replica_count.get(fn_name, None) is None:
            self.replica_count[fn_name] = 0
        if self.replica_count[fn_name] + replicas > fn.scale_max:
            if self.replica_count[fn_name] >= fn.scale_max:
                logger.debug('Function %s wanted to scale up, but maximum number of replicas reached', fn_name)
                return
            reduce = self.replica_count[fn_name] + replicas - fn.scale_max
            scale = replicas - reduce
        if scale == 0:
            return
        actually_scaled = 0
        for index, service in enumerate(fn.get_services()):
            # check whether service has capacity, otherwise continue
            # TODO can be possible that devices are left out when scale > rest capacity is
            leftover_scale = scale
            if fn.function_factor[service.image] * fn.scale_max < scale + self.functions_definitions[service.image]:
                max_replicas = int(fn.function_factor[service.image] * fn.scale_max)
                reduce = max_replicas - (self.functions_definitions[fn_name] + replicas)
                if reduce < 0:
                    # all replicas used
                    continue
                leftover_scale = leftover_scale - reduce
            if leftover_scale > 0:
                for _ in range(leftover_scale):
                    yield from self.deploy_replica(service, fn.get_services()[index:])
                    actually_scaled += 1
                    scale -= 1

        self.env.metrics.log_scaling(fn.name, actually_scaled)

        if scale > 0:
            logger.debug("Function %s wanted to scale, but not all requested replicas were deployed: %s", fn_name,
                         str(scale))

    def deploy_replica(self, fn: FunctionDefinition, services: List[FunctionDefinition]):
        replica = self.create_replica(fn)
        self.replicas[fn.name].append(replica)
        self.env.metrics.log_queue_schedule(replica)
        self.env.metrics.log_function_replica(replica)
        yield self.scheduler_queue.put((replica, services))

    def run_scheduler_worker(self):
        env = self.env

        while True:
            replica: FunctionReplica
            replica, services = yield self.scheduler_queue.get()

            logger.debug('scheduling next replica %s', replica.function.name)

            # schedule the required pod
            self.env.metrics.log_start_schedule(replica)
            pod = replica.pod
            then = time.time()
            result = env.scheduler.schedule(pod)
            duration = time.time() - then
            self.env.metrics.log_finish_schedule(replica, result)

            yield env.timeout(duration)  # include scheduling latency in simulation time

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Pod scheduling took %.2f ms, and yielded %s', duration * 1000, result)

            if not result.suggested_host:
                self.replicas[replica.function.name].remove(replica)
                if len(services) > 0:
                    logger.warning('retry scheduling pod %s', pod.name)
                    yield from self.deploy_replica(services[0], services[1:])
                else:
                    logger.error('pod %s cannot be scheduled', pod.name)

                continue

            logger.info('pod %s was scheduled to %s', pod.name, result.suggested_host)

            replica.node = self.env.get_node_state(result.suggested_host.name)

            # TODO decrease when removing replica
            self.functions_definitions[replica.function.image] += 1
            self.replica_count[replica.function.name] += 1

            self.env.metrics.log_function_deploy(replica)
            # start a new process to simulate starting of pod
            env.process(simulate_function_start(env, replica))


class FaasRequestScaler:

    def __init__(self, fn: FunctionDeployment, env: Environment):
        self.env = env
        self.function_invocations = dict()
        self.reconcile_interval = fn.rps_threshold_duration
        self.threshold = fn.rps_threshold
        self.alert_window = fn.alert_window
        self.running = True
        self.fn_name = fn.name
        self.fn = fn

    def run(self):
        env: Environment = self.env
        faas: OurFaas = env.faas
        while self.running:
            yield env.timeout(self.reconcile_interval)
            if self.function_invocations.get(self.fn_name, None) is None:
                self.function_invocations[self.fn_name] = 0
            last_invocations = self.function_invocations.get(self.fn_name, 0)
            current_total_invocations = env.metrics.invocations.get(self.fn_name, 0)
            invocations = current_total_invocations - last_invocations
            self.function_invocations[self.fn_name] += invocations
            # TODO divide by alert window, but needs to store the invocations, such that reconcile_interval != alert_window is possible
            if (invocations / self.reconcile_interval) >= self.threshold:
                scale = (self.fn.scale_factor / 100) * self.fn.scale_max
                yield from faas.scale_up(self.fn_name, int(scale))
                logger.debug(f'scaled up {self.fn_name} by {scale}')
            else:
                scale = (self.fn.scale_factor / 100) * self.fn.scale_max
                yield from faas.scale_down(self.fn_name, int(scale))
                logger.debug(f'scaled down {self.fn_name} by {scale}')

    def stop(self):
        self.running = False


class AverageFaasRequestScaler:
    """
    Scales deployment according to the average number of requests distributed equally over all replicas.
    The distributed property holds as per default the round robin scheduler is used
    """

    def __init__(self, fn: FunctionDeployment, env: Environment):
        self.env = env
        self.function_invocations = dict()
        self.threshold = fn.target_average_rps
        self.alert_window = fn.alert_window
        self.running = True
        self.fn_name = fn.name
        self.fn = fn

    def run(self):
        env: Environment = self.env
        faas: OurFaas = env.faas
        while self.running:
            yield env.timeout(self.alert_window)
            if self.function_invocations.get(self.fn_name, None) is None:
                self.function_invocations[self.fn_name] = 0
            running_replicas = faas.get_replicas(self.fn.name, FunctionState.RUNNING)
            running = len(running_replicas)
            if running == 0:
                continue

            conceived_replicas = faas.get_replicas(self.fn.name, FunctionState.CONCEIVED)
            starting_replicas = faas.get_replicas(self.fn.name, FunctionState.STARTING)

            last_invocations = self.function_invocations.get(self.fn_name, 0)
            current_total_invocations = env.metrics.invocations.get(self.fn_name, 0)
            invocations = current_total_invocations - last_invocations
            self.function_invocations[self.fn_name] += invocations
            average = invocations / running
            desired_replicas = math.ceil(running * (average / self.threshold))

            updated_desired_replicas = desired_replicas
            if len(conceived_replicas) > 0 or len(starting_replicas) > 0:
                if desired_replicas > len(running_replicas):
                    count = len(running_replicas) + len(conceived_replicas) + len(starting_replicas)
                    average = invocations/count
                    updated_desired_replicas = math.ceil(running * (average / self.threshold))

            if desired_replicas > len(running_replicas) and updated_desired_replicas < len(running_replicas):
                # no scaling in case of reversed decision
                continue

            ratio = average / self.threshold
            if 1 > ratio >= 1 - self.fn.target_average_rps_threshold:
                # ratio is sufficiently close to 1.0
                continue

            if 1 < ratio < 1 + self.fn.target_average_rps_threshold:
                continue

            if desired_replicas < len(running_replicas):
                # scale down
                scale = len(running_replicas) - desired_replicas
                yield from faas.scale_down(self.fn.name, scale)
            else:
                # scale up
                scale = desired_replicas - len(running_replicas)
                yield from faas.scale_up(self.fn.name, scale)

    def stop(self):
        self.running = False


class AverageQueueFaasRequestScaler:
    """
    Scales deployment according to the average number of requests distributed equally over all replicas.
    The distributed property holds as per default the round robin scheduler is used
    """

    def __init__(self, fn: FunctionDeployment, env: Environment):
        self.env = env
        self.threshold = fn.target_queue_length
        self.alert_window = fn.alert_window
        self.running = True
        self.fn_name = fn.name
        self.fn = fn

    def run(self):
        env: Environment = self.env
        faas: OurFaas = env.faas
        while self.running:
            yield env.timeout(self.alert_window)
            running_replicas = faas.get_replicas(self.fn.name, FunctionState.RUNNING)
            running = len(running_replicas)
            if running == 0:
                continue

            conceived_replicas = faas.get_replicas(self.fn.name, FunctionState.CONCEIVED)
            starting_replicas = faas.get_replicas(self.fn.name, FunctionState.STARTING)

            in_queue = []
            for replica in running_replicas:
                sim: 'InterferenceAwarePythonHttpSimulator' = replica.simulator
                in_queue.append(len(sim.queue.queue))
            if len(in_queue) == 0:
                average = 0
            else:
                average = int(math.ceil( np.median(np.array(in_queue))))

            desired_replicas = math.ceil(running * (average / self.threshold))

            updated_desired_replicas = desired_replicas
            if len(conceived_replicas) > 0 or len(starting_replicas) > 0:
                if desired_replicas > len(running_replicas):
                    for i in range(len(conceived_replicas) + len(starting_replicas)):
                        in_queue.append(0)

                    average = int(math.ceil( np.median(np.array(in_queue))))
                    updated_desired_replicas = math.ceil(running * (average / self.threshold))

            if desired_replicas > len(running_replicas) and updated_desired_replicas < len(running_replicas):
                # no scaling in case of reversed decision
                continue

            ratio = average / self.threshold
            if 1 > ratio >= 1 - self.fn.target_average_rps_threshold:
                # ratio is sufficiently close to 1.0
                continue

            if 1 < ratio < 1 + self.fn.target_average_rps_threshold:
                continue

            if desired_replicas < len(running_replicas):
                # scale down
                scale = len(running_replicas) - desired_replicas
                yield from faas.scale_down(self.fn.name, scale)
            else:
                # scale up
                scale = desired_replicas - len(running_replicas)
                yield from faas.scale_up(self.fn.name, scale)

    def stop(self):
        self.running = False
