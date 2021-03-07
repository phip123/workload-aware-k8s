from skippy.core.priorities import ImageLocalityPriority, BalancedResourcePriority

from hwmapping.notebook import predicates


def get_predicates(fet_oracle, resource_oracle):
    return predicates.get_predicates(fet_oracle, resource_oracle)


def get_priorities():
    return [
        (1, BalancedResourcePriority()),
        (1, ImageLocalityPriority()),
    ]


