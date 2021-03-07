from sim.core import Environment

from hwmapping.evaluation import images
from hwmapping.evaluation.benchmark import BenchmarkBase, set_degradation
from hwmapping.evaluation.benchmarks.util import create_deployments_for_profile
from hwmapping.evaluation.fetdistributions import execution_time_distributions
from hwmapping.evaluation.oracle import FetOracle, ResourceOracle
from hwmapping.evaluation.requestgen import expovariate_arrival_profile, sine_rps_profile
from hwmapping.evaluation.resources import resources_per_node_image


class SineBenchmark(BenchmarkBase):

    def __init__(self, profile: str, duration: int, max_rps: int, period: int, model_folder='./data'):
        all_images = images.all_images
        self.model_folder = model_folder
        self.profile = profile
        self.max_rps = max_rps
        self.period = period
        self.duration = duration
        fet_oracle = FetOracle(execution_time_distributions)
        resource_oracle = ResourceOracle(resources_per_node_image)

        deployments = create_deployments_for_profile(profile, fet_oracle, resource_oracle)

        super().__init__(all_images, list(deployments.values()), arrival_profiles=dict(), duration=duration)

    @property
    def settings(self):
        return {
            'max_rps': self.max_rps,
            'period': self.period,
            'duration': self.duration,
            'profile': self.profile
        }

    @property
    def type(self):
        return 'sine'

    def setup(self, env: Environment):
        self.set_deployments(env)
        self.setup_profile(env)
        set_degradation(env, self.model_folder)
        super().setup(env)

    def setup_profile(self, env: Environment):
        if self.profile == 'service':
            self.set_service_profiles(env)
        elif self.profile == 'ai':
            self.set_ai_profiles(env)
        elif self.profile == 'mixed':
            self.set_mixed_profiles(env)
        else:
            raise AttributeError(f'unknown profile: {self.profile}')

    def set_mixed_profiles(self, env: Environment):

        self.arrival_profiles[images.resnet50_inference_function] = \
            expovariate_arrival_profile(sine_rps_profile(env, self.max_rps, self.period / 2))

        self.arrival_profiles[images.mobilenet_inference_function] = \
            expovariate_arrival_profile(sine_rps_profile(env, self.max_rps, self.period / 2))

        self.arrival_profiles[images.speech_inference_function] = \
            expovariate_arrival_profile(sine_rps_profile(env, self.max_rps, self.period / 2))

        self.arrival_profiles[images.resnet50_training_function] = \
            expovariate_arrival_profile(sine_rps_profile(env, 10, self.period * 2))

        self.arrival_profiles[images.resnet50_preprocessing_function] = \
            expovariate_arrival_profile(sine_rps_profile(env, self.max_rps / 4, self.period))
        #
        # self.arrival_profiles[images.pi_function] = \
        #     expovariate_arrival_profile(sine_rps_profile(env, self.max_rps, self.period / 4))
        # self.arrival_profiles[images.tf_gpu_function] = \
        #     expovariate_arrival_profile(sine_rps_profile(env, self.max_rps, self.period / 3))
        # # set arrival profiles
        # self.arrival_profiles[images.fio_function] = \
        #     expovariate_arrival_profile(sine_rps_profile(env, self.max_rps, self.period / 2))

    def set_deployments(self, env):
        deployments = self.deployments_per_name
        no_of_devices = len(env.topology.get_nodes())
        for deployment in deployments.values():
            deployment.scale_min = 5
            deployment.target_average_utilization = 0.5

        if self.profile == 'ai' or self.profile == 'mixed':
            deployments[images.resnet50_inference_function].rps_threshold = 100
            deployments[images.resnet50_inference_function].scale_max = no_of_devices
            deployments[images.resnet50_inference_function].scale_factor = 2
            deployments[images.resnet50_inference_function].rps_threshold_duration = 10
            deployments[images.mobilenet_inference_function].rps_threshold = 70
            deployments[images.mobilenet_inference_function].scale_max = no_of_devices
            deployments[images.mobilenet_inference_function].scale_factor = 1
            deployments[images.mobilenet_inference_function].rps_threshold_duration = 10
            deployments[images.speech_inference_function].rps_threshold = 40
            deployments[images.speech_inference_function].scale_max = no_of_devices
            deployments[images.speech_inference_function].scale_factor = 1
            deployments[images.speech_inference_function].rps_threshold_duration = 15
            deployments[images.resnet50_preprocessing_function].rps_threshold = 40
            deployments[images.resnet50_preprocessing_function].scale_max = no_of_devices / 4
            deployments[images.resnet50_preprocessing_function].scale_factor = 1
            deployments[images.resnet50_preprocessing_function].rps_threshold_duration = 15
            deployments[images.resnet50_training_function].rps_threshold = 40
            deployments[images.resnet50_training_function].scale_max = no_of_devices / 2
            deployments[images.resnet50_training_function].scale_factor = 1
            deployments[images.resnet50_training_function].rps_threshold_duration = 15

        if self.profile == 'service':
            deployments[images.fio_function].rps_threshold = 100
            deployments[images.fio_function].scale_max = int(0.7 * no_of_devices)
            deployments[images.fio_function].scale_factor = int(0.05 * no_of_devices)
            deployments[images.fio_function].rps_threshold_duration = 10
            deployments[images.pi_function].rps_threshold = 70
            deployments[images.pi_function].scale_max = int(0.25 * no_of_devices)
            deployments[images.pi_function].scale_factor = 5
            deployments[images.pi_function].rps_threshold_duration = 10
            deployments[images.tf_gpu_function].rps_threshold = 40
            deployments[images.tf_gpu_function].scale_max = int(0.25 * no_of_devices)
            deployments[images.tf_gpu_function].scale_factor = 5
            deployments[images.tf_gpu_function].rps_threshold_duration = 15

    def set_ai_profiles(self, env):
        self.arrival_profiles[images.resnet50_inference_function] = \
            expovariate_arrival_profile(sine_rps_profile(env, self.max_rps, self.period / 4))

        self.arrival_profiles[images.mobilenet_inference_function] = \
            expovariate_arrival_profile(sine_rps_profile(env, self.max_rps, self.period / 3))

        self.arrival_profiles[images.speech_inference_function] = \
            expovariate_arrival_profile(sine_rps_profile(env, self.max_rps, self.period / 2))

        self.arrival_profiles[images.resnet50_training_function] = \
            expovariate_arrival_profile(sine_rps_profile(env, 10, self.period * 2))

        self.arrival_profiles[images.resnet50_preprocessing_function] = \
            expovariate_arrival_profile(sine_rps_profile(env, self.max_rps, self.period))

    def set_service_profiles(self, env):
        self.arrival_profiles[images.pi_function] = \
            expovariate_arrival_profile(sine_rps_profile(env, self.max_rps, self.period / 4))
        self.arrival_profiles[images.tf_gpu_function] = \
            expovariate_arrival_profile(sine_rps_profile(env, self.max_rps, self.period / 3))
        # set arrival profiles
        self.arrival_profiles[images.fio_function] = \
            expovariate_arrival_profile(sine_rps_profile(env, self.max_rps, self.period / 2))
