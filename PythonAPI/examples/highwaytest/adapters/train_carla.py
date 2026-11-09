"""Use the same runner, networks, objective and paired evaluator on CARLA.

Call from the existing CARLA scenario owner after registering frozen scene
and traffic-controller hashes in Config.backend_contract. This function does
not guess Town05 spawn coordinates or launch a second simulator owner.
"""
from .carla import CarlaConditionBackend
from .environment import PortablePairedEnv
from ..evaluation.paired import evaluate_paired
from ..training.runner import run


def train_carla(config, output, *, world, loader, traffic_control,
                acceleration_controls, control_factory, resume=None, max_rounds=None):
    if config.backend != "carla":
        raise ValueError("declare backend=carla and frozen backend_contract hashes")

    def backend(condition, family, seed):
        return CarlaConditionBackend(config.task, condition, family, seed, world=world, loader=loader,
            traffic_control=traffic_control, acceleration_controls=acceleration_controls,
            control_factory=control_factory)

    def environment_factory(task, gamma):
        return PortablePairedEnv(task, gamma, backend)

    def evaluation_factory(model, task, split, per_family, gamma):
        return evaluate_paired(model, task, split, per_family, gamma=gamma, backend_factory=backend,
                               backend_contract={"backend": "carla", **config.backend_contract})

    return run(config, output, resume=resume, max_rounds=max_rounds,
               environment_factory=environment_factory, evaluation_factory=evaluation_factory)
