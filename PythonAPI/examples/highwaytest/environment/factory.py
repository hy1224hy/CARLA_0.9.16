def make_env(task, gamma=0.99):
    if task.research_protocol=='bottleneck_paired_attack_v4':
        from ..bottleneck.environment import BottleneckEnv
        return BottleneckEnv(task,gamma=gamma)
    if task.research_protocol == "legacy_v2":
        from .highway import HighwayTeamEnv
        return HighwayTeamEnv(task)
    from .paired import PairedAttackEnv
    return PairedAttackEnv(task, gamma=gamma)
