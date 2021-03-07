import glob
import os
import pickle
from collections import defaultdict
from typing import List

from hwmapping.calculations import calculate_requirements
from hwmapping.etheradapter import convert_to_ether_nodes
from hwmapping.generator import GeneratorSettings, generate_devices


def read_generator_settings(folder: str) -> List[GeneratorSettings]:
    files = glob.glob(f'{folder}/*')
    settings = []
    for file in files:
        if os.path.getsize(file) > 0:
            with open(file, 'rb') as fd:
                setting = pickle.load(fd)
                settings.append(setting)
    return settings


def test_generated_settings():
    settings_path = '/mnt/ssd1data/Documents/code/python/hw_mapping_gen_settings'
    settings_folder = '2020_11_24_19_29_13_archsteps-6_steps-5_percentage-0.1_500_random_extract'
    settings_folder = os.path.join(settings_path, settings_folder)
    settings = read_generator_settings(settings_folder)[:10]
    num_devices = 100
    all_devices = []
    ether_nodes = []
    all_devices_by_types = []
    for index, setting in enumerate(settings):
        all_devices.append(generate_devices(num_devices, setting))
        ether_nodes.append(convert_to_ether_nodes(all_devices[index]))

        devices_by_type = defaultdict(list)
        for devices in ether_nodes:
            for device in devices:
                devices_by_type[device.name[:device.name.rindex('_')]].append(device)
            all_devices_by_types.append(devices_by_type)
    scores = []
    for devices in all_devices:
        scores.append(calculate_requirements(devices))

    pass


def test():
    path = "/mnt/ssd1data/Documents/code/python/hw_mapping_gen_settings"
    folder = '2020_11_24_19_29_13_archsteps-6_steps-5_percentage-0.1_500_random_extract'
    settings = read_generator_settings(os.path.join(path, folder))
    # devices = generate_devices_with_settings(100, settings[0][0])
    pass


if __name__ == '__main__':
    test()
