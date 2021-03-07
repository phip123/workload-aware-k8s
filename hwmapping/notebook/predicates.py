from skippy.core.scheduler import Scheduler

from hwmapping.faas.predicates import CanRunPred, HasEnoughRamPredicate, NodeHasAcceleratorPred, NodeHasFreeGpu, \
    NodeHasFreeTpu


def get_predicates(fet_oracle, resource_oracle):
    predicates = []
    predicates.extend(Scheduler.default_predicates)
    predicates.extend([
        CanRunPred(fet_oracle, resource_oracle),
        HasEnoughRamPredicate(resource_oracle),
        NodeHasAcceleratorPred(),
        NodeHasFreeGpu(),
        NodeHasFreeTpu()
    ])
    return predicates
