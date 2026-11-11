from dataclasses import dataclass,field,asdict
import importlib
import math

BACKENDS={1:'highway_l1_style_v4',2:'highway_behavior_v3',3:'highway_hybrid'}
PROTOCOL='highway_research_v1'


@dataclass
class StudyConfig:
    task:dict
    algorithm:str='ppo'
    mutual:str='off'
    beta:float=.02
    self_kl:float=0.
    acc_multiplier:float=1.
    lane_multiplier:float=1.
    drop_components:list=field(default_factory=list)
    trpo_delta:float=.01
    trpo_damping:float=.1
    trpo_cg_steps:int=10
    trpo_backtracks:int=10
    critic_epochs:int=4
    arm:str='main'
    run_id:str='diagnostic'

    def __post_init__(self):self.validate()

    @classmethod
    def default(cls,level=1,seed=42,**changes):
        backend=importlib.import_module(BACKENDS[level]+'.config')
        cfg=backend.HybridConfig(level=level,seed=seed,total_steps=500000,
            approach_weight=0.,guidance_weight=0.,geometry_weight=2. if level<3 else 0.)
        return cls(task=cfg.to_dict(),**changes)

    def __getattr__(self,key):
        task=self.__dict__.get('task',{})
        if key in task:return task[key]
        base=self.__dict__.get('_base')
        if base is not None:return getattr(base,key)
        raise AttributeError(key)

    def validate(self):
        if self.task.get('level') not in BACKENDS:raise ValueError('unknown level')
        base=importlib.import_module(BACKENDS[self.task['level']]+'.config')
        self._base=base.HybridConfig.from_dict(self.task)
        if self.algorithm not in ('ppo','a2c','trpo'):raise ValueError('algorithm')
        if self.mutual not in ('off','both','one_way'):raise ValueError('mutual mode')
        if self.algorithm!='ppo' and (self.mutual!='off' or self.self_kl):raise ValueError('PPO-only mechanism')
        for key in ('beta','self_kl','acc_multiplier','lane_multiplier','trpo_delta','trpo_damping'):
            value=getattr(self,key)
            if not math.isfinite(value) or value<0:raise ValueError(key)
        if self.trpo_delta<=0 or self.trpo_cg_steps<1 or self.trpo_backtracks<1 or self.critic_epochs<1:raise ValueError('TRPO settings')
        if len(set(self.drop_components))!=len(self.drop_components):raise ValueError('duplicate ablation')
        if set(self.drop_components)-{'style','impact','geometry','lane_change'}:raise ValueError('unsupported ablation')
        if self.level!=1 and set(self.drop_components)-{'geometry'}:raise ValueError('wrong level for components')
        return self

    def to_dict(self):return asdict(self)

    @classmethod
    def from_dict(cls,data):return cls(**data)

    @property
    def backend(self):return BACKENDS[self.level]

    @property
    def state_dim(self):return self._base.state_dim

    @property
    def label(self):
        if self.mutual=='both':return 'Mut-PPO'
        if self.mutual=='one_way':return 'PPO-one-way'
        if self.self_kl:return 'PPO-selfKL'
        return self.algorithm.upper()
