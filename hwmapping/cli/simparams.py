import itertools
import logging
from typing import Dict

from hwmapping.evaluation.benchmarks.constant import ConstantBenchmark as FirstConstantBenchmark
from hwmapping.evaluation.benchmarks.prerecorded import PrerecordedBenchmark
from hwmapping.evaluation.benchmarks.second.constant import ConstantBenchmark as SecondConstantBenchmark
from hwmapping.evaluation.benchmarks.second.sine import SineBenchmark as SecondSineBenchmark
from hwmapping.evaluation.benchmarks.sine import SineBenchmark as FirstSineBenchmark
from hwmapping.evaluation.results import SingleRunGaResult, set_requirements
from hwmapping.notebook import ga, vanilla, skippy


def create_prerecorded_benchmark(profile, duration, arrival_patterns: Dict[str, str], model_folder):
    return PrerecordedBenchmark(profile, arrival_patterns, duration, model_folder)


def create_sine_benchmark(benchmark, profile, duration, max_rps, period, model_folder):
    if benchmark == 'first' or benchmark is None:
        return FirstSineBenchmark(profile,
                                  duration=duration, max_rps=max_rps, period=period, model_folder=model_folder)
    elif benchmark == 'second':
        return SecondSineBenchmark(profile,
                                   duration=duration, max_rps=max_rps, period=period, model_folder=model_folder)
    else:
        raise ValueError(f'Unknown benchmark: {benchmark}')


def create_constant_benchmark(benchmark, profile, duration, rps, model_folder):
    if benchmark == 'first' or benchmark is None:
        return FirstConstantBenchmark(profile, duration=duration, rps=rps, model_folder=model_folder)
    elif benchmark == 'second':
        return SecondConstantBenchmark(profile, duration=duration, rps=rps, model_folder=model_folder)
    else:
        raise ValueError(f'Unknown benchmark: {benchmark}')


def ga_prerecorded(capability_weights, contention_weights, durations, fet_oracle, fet_weights, model_folder,
                   percentage_of_nodes_to_scores, profile, resource_oracle, arrival_patterns,
                   single_run_result: SingleRunGaResult):
    # execute  GA powered Sinebenchmarks
    for t in itertools.product(percentage_of_nodes_to_scores, capability_weights, contention_weights, fet_weights,
                               durations):
        percentage_of_nodes_to_score = t[0]
        capability_weight = t[1]
        contention_weight = t[2]
        fet_weight = t[3]
        duration = t[4]

        predicates = ga.get_predicates(fet_oracle, resource_oracle)

        priorities = ga.get_priorities(
            fet_oracle,
            resource_oracle,
            capability_weight=capability_weight,
            contention_weight=contention_weight,
            fet_weight=fet_weight
        )

        sched_params = {
            'percentage_of_nodes_to_score': percentage_of_nodes_to_score,
            'priorities': priorities,
            'predicates': predicates
        }

        benchmark = create_prerecorded_benchmark(profile, duration, arrival_patterns, model_folder)

        set_requirements(benchmark, single_run_result)

        type_run = 'ga'

        # TODO in case of other prerecorded patterns, type must be updated
        settings = {
            'percentage_nodes_to_score': percentage_of_nodes_to_score,
            'contention_weight': contention_weight,
            'capability_weight': capability_weight,
            'fet_weight': fet_weight,
            'duration': duration,
            'type': 'sine',
            'optimization': 'ga'
        }

        yield benchmark, type_run, sched_params, single_run_result, settings


def vanilla_constant(constant_rps, durations, fet_oracle, model_folder, percentage_of_nodes_to_scores, profile,
                     resource_oracle, single_run_result: SingleRunGaResult, benchmark):
    for t in itertools.product(percentage_of_nodes_to_scores, durations, constant_rps):
        percentage_of_nodes_to_score = t[0]
        duration = t[1]
        rps = t[2]

        predicates = vanilla.get_predicates(fet_oracle, resource_oracle)

        priorities = vanilla.get_priorities()

        sched_params = {
            'percentage_of_nodes_to_score': percentage_of_nodes_to_score,
            'priorities': priorities,
            'predicates': predicates
        }

        benchmark = create_constant_benchmark(benchmark, profile, duration=duration, rps=rps, model_folder=model_folder)

        type_run = 'vanilla'

        settings = {
            'percentage_nodes_to_score': percentage_of_nodes_to_score,
            'duration': duration,
            'rps': rps,
            'type': 'constant',
            'optimization': type_run
        }

        yield benchmark, type_run, sched_params, single_run_result, settings


def vanilla_prerecorded(durations, fet_oracle, model_folder, percentage_of_nodes_to_scores, profile, resource_oracle,
                        arrival_patterns, single_run_result):
    for t in itertools.product(percentage_of_nodes_to_scores, durations):
        percentage_of_nodes_to_score = t[0]
        duration = t[1]
        predicates = vanilla.get_predicates(fet_oracle, resource_oracle)

        priorities = vanilla.get_priorities()

        sched_params = {
            'percentage_of_nodes_to_score': percentage_of_nodes_to_score,
            'priorities': priorities,
            'predicates': predicates
        }

        benchmark = create_prerecorded_benchmark(profile, duration, arrival_patterns, model_folder)
        # TODO in case of other prerecorded patterns, type must be updated

        type_run = 'vanilla'
        settings = {
            'percentage_nodes_to_score': percentage_of_nodes_to_score,
            'duration': duration,
            'type': 'sine',
            'optimization': type_run
        }

        yield benchmark, type_run, sched_params, single_run_result, settings


def ga_constant(capability_weights, constant_rps, contention_weights, durations, fet_oracle, fet_weights, model_folder,
                percentage_of_nodes_to_scores, profile, resource_oracle, single_run_result: SingleRunGaResult,
                benchmark):
    for t in itertools.product(percentage_of_nodes_to_scores, capability_weights, contention_weights, fet_weights,
                               durations, constant_rps):
        try:
            percentage_of_nodes_to_score = t[0]
            capability_weight = t[1]
            contention_weight = t[2]
            fet_weight = t[3]
            duration = t[4]
            rps = t[5]

            predicates = ga.get_predicates(fet_oracle, resource_oracle)

            priorities = ga.get_priorities(
                fet_oracle,
                resource_oracle,
                capability_weight=capability_weight,
                contention_weight=contention_weight,
                fet_weight=fet_weight
            )

            sched_params = {
                'percentage_of_nodes_to_score': percentage_of_nodes_to_score,
                'priorities': priorities,
                'predicates': predicates
            }

            benchmark = create_constant_benchmark(benchmark, profile, duration=duration, rps=rps,
                                                  model_folder=model_folder)

            set_requirements(benchmark, single_run_result)

            type_run = 'ga'

            settings = {
                'percentage_nodes_to_score': percentage_of_nodes_to_score,
                'contention_weight': contention_weight,
                'capability_weight': capability_weight,
                'fet_weight': fet_weight,
                'duration': duration,
                'rps': rps,
                'type': 'constant',
                'optimization': 'ga'
            }

            yield benchmark, type_run, sched_params, single_run_result, settings
        except IndexError as e:
            logging.error(e)


def skippy_constant(latency_weights, constant_rps, data_weights, durations, fet_oracle,
                    model_folder,
                    percentage_of_nodes_to_scores, profile, resource_oracle, single_run_result: SingleRunGaResult,
                    benchmark):
    # execute  GA powered Sinebenchmarks
    for t in itertools.product(percentage_of_nodes_to_scores, latency_weights, data_weights,
                               durations, constant_rps):
        try:
            percentage_of_nodes_to_score = t[0]
            latency_weight = t[1]
            data_weight = t[2]
            duration = t[3]
            rps = t[4]

            predicates = skippy.get_predicates(fet_oracle, resource_oracle)

            priorities = skippy.get_priorities(
                latency_weight=latency_weight,
                data_weight=data_weight
            )

            sched_params = {
                'percentage_of_nodes_to_score': percentage_of_nodes_to_score,
                'priorities': priorities,
                'predicates': predicates
            }

            benchmark = create_constant_benchmark(benchmark, profile, duration=duration, rps=rps,
                                                  model_folder=model_folder)

            type_run = 'skippy'

            settings = {
                'percentage_nodes_to_score': percentage_of_nodes_to_score,
                'latency_weight': latency_weight,
                'data_weight': data_weight,
                'duration': duration,
                'rps': rps,
                'type': 'constant',
                'optimization': 'skippy'
            }

            yield benchmark, type_run, sched_params, single_run_result, settings
        except IndexError as e:
            logging.error(e)


def skippy_prerecorded(latency_weights, durations, fet_oracle, data_weights, model_folder,
                       percentage_of_nodes_to_scores,
                       profile, resource_oracle, arrival_patterns, single_run_result):
    for t in itertools.product(percentage_of_nodes_to_scores, latency_weights, data_weights,
                               durations):
        percentage_of_nodes_to_score = t[0]
        latency_weight = t[1]
        data_weight = t[2]
        duration = t[3]
        predicates = skippy.get_predicates(fet_oracle, resource_oracle)

        priorities = skippy.get_priorities(
            latency_weight=latency_weight,
            data_weight=data_weight
        )

        sched_params = {
            'percentage_of_nodes_to_score': percentage_of_nodes_to_score,
            'priorities': priorities,
            'predicates': predicates
        }

        benchmark = create_prerecorded_benchmark(profile, duration, arrival_patterns, model_folder)
        # TODO in case of other prerecorded patterns, type must be updated

        type_run = 'skippy'
        settings = {
            'percentage_nodes_to_score': percentage_of_nodes_to_score,
            'latency_weight': latency_weight,
            'data_weight': data_weight,
            'duration': duration,
            'type': 'sine',
            'optimization': 'skippy'
        }

        yield benchmark, type_run, sched_params, single_run_result, settings


def all_constant(latency_weights, data_weights, capability_weights, contention_weights, fet_weights, constant_rps,
                 durations, fet_oracle,
                 model_folder,
                 percentage_of_nodes_to_scores, profile, resource_oracle, single_run_result: SingleRunGaResult,
                 benchmark):
    # execute  all powered constantBenchmarks
    for t in itertools.product(percentage_of_nodes_to_scores, latency_weights, data_weights, capability_weights,
                               contention_weights, fet_weights,
                               durations, constant_rps):
        try:
            percentage_of_nodes_to_score = t[0]
            latency_weight = t[1]
            data_weight = t[2]
            capability_weight = t[3]
            contention_weight = t[4]
            fet_weight = t[5]
            duration = t[6]
            rps = t[7]

            predicates = skippy.get_predicates(fet_oracle, resource_oracle)

            skippy_priorities = skippy.get_priorities(
                latency_weight=latency_weight,
                data_weight=data_weight
            )

            ga_priorities = ga.get_priorities(
                fet_oracle,
                resource_oracle,
                capability_weight=capability_weight,
                contention_weight=contention_weight,
                fet_weight=fet_weight
            )

            priorities = []
            priorities.extend(skippy_priorities)
            priorities.extend(ga_priorities)

            sched_params = {
                'percentage_of_nodes_to_score': percentage_of_nodes_to_score,
                'priorities': priorities,
                'predicates': predicates
            }

            benchmark = create_constant_benchmark(benchmark, profile, duration=duration, rps=rps,
                                                  model_folder=model_folder)

            set_requirements(benchmark, single_run_result)

            type_run = 'all'

            settings = {
                'percentage_nodes_to_score': percentage_of_nodes_to_score,
                'latency_weight': latency_weight,
                'data_weight': data_weight,
                'contention_weight': contention_weight,
                'capability_weight': capability_weight,
                'fet_weight': fet_weight,
                'duration': duration,
                'rps': rps,
                'type': 'constant',
                'optimization': type_run
            }

            yield benchmark, type_run, sched_params, single_run_result, settings
        except IndexError as e:
            logging.error(e)


def all_prerecorded(capability_weights, contention_weights, fet_weights, latency_weights, data_weights, durations,
                    fet_oracle,
                    model_folder, percentage_of_nodes_to_scores,
                    profile, resource_oracle, arrival_patterns, single_run_result):
    for t in itertools.product(percentage_of_nodes_to_scores, latency_weights, data_weights, capability_weights,
                               contention_weights, fet_weights,
                               durations):
        percentage_of_nodes_to_score = t[0]
        latency_weight = t[1]
        data_weight = t[2]
        capability_weight = t[3]
        contention_weight = t[4]
        fet_weight = t[5]
        duration = t[6]
        predicates = skippy.get_predicates(fet_oracle, resource_oracle)

        skippy_priorities = skippy.get_priorities(
            latency_weight=latency_weight,
            data_weight=data_weight
        )

        ga_priorities = ga.get_priorities(
            fet_oracle,
            resource_oracle,
            capability_weight=capability_weight,
            contention_weight=contention_weight,
            fet_weight=fet_weight
        )

        priorities = []
        priorities.extend(skippy_priorities)
        priorities.extend(ga_priorities)

        sched_params = {
            'percentage_of_nodes_to_score': percentage_of_nodes_to_score,
            'priorities': priorities,
            'predicates': predicates
        }

        benchmark = create_prerecorded_benchmark(profile, duration, arrival_patterns, model_folder)

        set_requirements(benchmark, single_run_result)
        # TODO in case of other prerecorded patterns, type must be updated

        type_run = 'all'
        settings = {
            'percentage_nodes_to_score': percentage_of_nodes_to_score,
            'latency_weight': latency_weight,
            'data_weight': data_weight,
            'contention_weight': contention_weight,
            'capability_weight': capability_weight,
            'fet_weight': fet_weight,
            'duration': duration,
            'type': 'sine',
            'optimization': type_run
        }

        yield benchmark, type_run, sched_params, single_run_result, settings
