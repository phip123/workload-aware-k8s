import logging
import os
from typing import List, Tuple, Dict, Generator

import joblib
import numpy as np
from ether.util import parse_size_string
from sim import docker
from sim.benchmark import Benchmark
from sim.core import Environment
from sim.docker import ImageProperties
from sim.faas import FunctionRequest

from hwmapping.evaluation import images
from hwmapping.evaluation.deployments import create_all_deployments
from hwmapping.evaluation.fetdistributions import execution_time_distributions
from hwmapping.evaluation.oracle import FetOracle, ResourceOracle
from hwmapping.evaluation.requestgen import function_trigger, expovariate_arrival_profile, sine_rps_profile, \
    constant_rps_profile
from hwmapping.evaluation.resources import resources_per_node_image
from hwmapping.faas.system import FunctionDeployment


class BenchmarkBase(Benchmark):
    def __init__(self, images: List[Tuple[str, str, str]], deployments: List[FunctionDeployment],
                 arrival_profiles: Dict[str, Generator], duration: int = None):
        self.duration = duration  # in seconds
        self.images = images
        self.deployments = deployments
        self.deployments_per_name = self.__create_deployments_per_name()
        self.arrival_profiles = arrival_profiles

    def __create_deployments_per_name(self):
        deployments_per_name = {}
        for deployment in self.deployments:
            deployments_per_name[deployment.name] = deployment
        return deployments_per_name

    def setup(self, env: Environment):
        super().setup(env)
        self.register_images(env)

    def register_images(self, env: Environment):
        containers: docker.ContainerRegistry = env.container_registry
        for image, size, arch in self.images:
            containers.put(ImageProperties(image, parse_size_string(size), arch=arch))

        for name, tag_dict in containers.images.items():
            for tag, images in tag_dict.items():
                logging.info('%s, %s, %s', name, tag, images)

    def run(self, env: Environment):
        for deployment in self.deployments:
            yield from env.faas.deploy(deployment)
        for deployment in self.deployments:
            yield env.process(env.faas.poll_available_replica(deployment.name))

        ps = []
        logging.info('executing requests')
        for deployment in self.deployments:
            try:
                ia_generator = self.arrival_profiles[deployment.name]
                if self.duration is None:
                    p = env.process(function_trigger(env, deployment, ia_generator, max_requests=1000))
                else:
                    p = env.process(function_trigger(env, deployment, ia_generator))
                ps.append(p)
            except KeyError:
                logging.warning('no arrival profile for deployment %s', deployment.name)

        if self.duration is not None:
            env.process(self.wait(env, ps))

        yield from ps

    def wait(self, env, ps):
        yield env.timeout(env.now + self.duration)
        for p in ps:
            p.interrupt('stop')


class DevBenchmark(BenchmarkBase):
    """
    Benchmark used to develop the simulation and evaluation.
    """

    def __init__(self):
        all_images = images.all_images

        fet_oracle = FetOracle(execution_time_distributions)
        resource_oracle = ResourceOracle(resources_per_node_image)

        deployments = create_all_deployments(fet_oracle, resource_oracle)

        super().__init__(all_images, list(deployments.values()), arrival_profiles=dict())

    def setup(self, env: Environment):
        super().setup(env)
        deployments = self.deployments_per_name
        no_of_devices = len(env.topology.get_nodes())
        deployments[images.resnet50_inference_function].rps_threshold = 100
        deployments[images.resnet50_inference_function].scale_max = int(0.7 * no_of_devices)
        deployments[images.resnet50_inference_function].scale_factor = int(0.05 * no_of_devices)
        deployments[images.resnet50_inference_function].rps_threshold_duration = 10

        deployments[images.mobilenet_inference_function].rps_threshold = 70
        deployments[images.mobilenet_inference_function].scale_max = int(0.25 * no_of_devices)
        deployments[images.mobilenet_inference_function].scale_factor = 5
        deployments[images.mobilenet_inference_function].rps_threshold_duration = 10

        deployments[images.speech_inference_function].rps_threshold = 40
        deployments[images.speech_inference_function].scale_max = int(0.25 * no_of_devices)
        deployments[images.speech_inference_function].scale_factor = 5
        deployments[images.speech_inference_function].rps_threshold_duration = 15

        self.arrival_profiles[images.mobilenet_inference_function] = \
            expovariate_arrival_profile(constant_rps_profile(rps=100), max_ia=240)

        self.arrival_profiles[images.speech_inference_function] = \
            expovariate_arrival_profile(sine_rps_profile(env, max_rps=100, period=3600), max_ia=240)

        # set arrival profiles
        self.arrival_profiles[images.resnet50_inference_function] = \
            expovariate_arrival_profile(sine_rps_profile(env, max_rps=300, period=3600), max_ia=240)

        # set arrival profiles
        self.arrival_profiles[images.resnet50_preprocessing_function] = \
            expovariate_arrival_profile(sine_rps_profile(env, max_rps=10, period=3600), max_ia=240)

        # set arrival profiles
        self.arrival_profiles[images.resnet50_training_function] = \
            expovariate_arrival_profile(sine_rps_profile(env, max_rps=1, period=240), max_ia=240)


def get_model_file(folder, node_name):
    if 'xeongpu' in node_name or 'xeoncpu' in node_name:
        file = 'eb-xeongpu.sav'
    elif 'nx' in node_name:
        file = 'eb-jetson-nx-01.sav'
    elif 'nano' in node_name:
        file = 'eb-jetson-nano-01.sav'
    elif 'tx2' in node_name:
        file = 'eb-jetson-tx2-01.sav'
    elif 'tpu' in node_name or 'coral' in node_name:
        file = 'eb-rpi4-01.sav'
    elif 'rpi3' in node_name:
        file = 'eb-rpi3-01.sav'
    elif 'rockpi' in node_name:
        file = 'eb-rockpi.sav'
    elif 'nuc' in node_name:
        file = 'eb-nuc7.sav'
    elif 'rpi4' in node_name:
        file = 'eb-rpi4-01.sav'
    else:
        raise ValueError(f"Can't find model for node: {node_name}")
    return os.path.join(folder, file)


def set_degradation(env: Environment, folder: str):
    models = {}
    for ether_node in env.topology.get_nodes():
        try:
            name = ether_node.name[:ether_node.name.rindex("_")]
            model = models.get(name, None)
            if model is None:
                model_file = get_model_file(folder, name)
                model = joblib.load(model_file)
                models[name] = model

            env.degradation_models[ether_node.name] = model
        except ValueError:
            # happens when an ether_node has no '_', i.e. docker registry -> can be ignored
            pass


class DevDegradationBenchmark(DevBenchmark):

    def __init__(self, model_folder='./data'):
        super().__init__()
        self.model_folder = model_folder

    def setup(self, env: Environment):
        set_degradation(env, self.model_folder)
        super().setup(env)
