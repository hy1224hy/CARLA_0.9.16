import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch


class AuditTests(unittest.TestCase):
    def test_episode_audit_reads_serialized_role_keys(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from training import train_mut_mappo as runner
        from utils.ppo_real_rollout_audit import instrument
        buffer=SimpleNamespace(data={2:[],3:[]})
        def episode(*args,**kwargs):
            for aid in (2,3):buffer.data[aid].append(SimpleNamespace(reward=-5.))
            return dict(team_discounted_return=-5.,agent_terminated={'2':True,'3':True},
                agent_truncated={'2':False,'3':False},team_tail_credit={'2':{},'3':{}})
        runtime=SimpleNamespace(team='A',config={'learning':{'gamma':.999}})
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'_episode',episode),instrument(tmp):
            result=runner._episode(runtime,{'case_key':'serialized_role_check'},buffer=buffer)
            self.assertEqual(result['team_discounted_return'],-5.)

    def test_observation_does_not_change_production_update_or_rng(self):
        from models.actor_critic import Actor
        from training.mut_mappo_learning import TeamLearner, actor_states_hash
        from training.train_no_rllib_mappo_robust_v2 import RolloutBuffer,Transition
        from utils.ppo_real_rollout_audit import instrument
        from training.checkpoint_protocol import capture_rng_state
        actors={a:Actor(20,2,hidden=128).state_dict() for a in (0,1)}
        cfg=dict(method='off',gamma=.999,gae_lambda=.95,ppo_epochs=2,minibatch_size=8,
                 hidden_dim=128,init_log_std=-.5,min_log_std=-2.,max_log_std=0.)
        a=TeamLearner('A',1042,actors,cfg)
        b=TeamLearner('A',1042,actors,cfg)
        buffer=RolloutBuffer([2,3])
        for ep in range(2):
            for step in range(5):
                obs={i:np.full(20,.01*(step+ep+i),dtype=np.float32) for i in range(4)}
                state=np.concatenate(list(obs.values()))
                _,cache=a.sample(obs,state,{2:True,3:True})
                for aid,c in cache.items():
                    buffer.add(aid,Transition(c['obs'],c['global_state'],c['action_z'],c['logp'],
                        float(step-2),float(step==4),float(step==4),c['value'],'synthetic_check'))
            for aid in (2,3):
                buffer.mark_episode_end(aid);buffer.add_episode_bootstrap(aid,None)
        # Match the post-collection learner stream, including its RNG.
        b.load_state_dict(a.state_dict())
        plain=a.update(copy.deepcopy(buffer))
        rng=capture_rng_state()
        with tempfile.TemporaryDirectory() as tmp,instrument(tmp):
            audited=b.update(copy.deepcopy(buffer))
            evidence=json.loads((Path(tmp)/'updates.jsonl').read_text(encoding='utf-8'))
            self.assertLess(max(evidence['gae_errors'].values()),1e-3)
            self.assertEqual(len(evidence['roles']),2)
            for role in evidence['roles'].values():self.assertLess(abs(role['old_surrogate']),1e-5)
            # Exercise the actual sample-time check, including nontrivial z.
            b.sample(obs,state,{2:True,3:True})
        self.assertEqual(actor_states_hash(a.export_actors()),actor_states_hash(b.export_actors()))
        for key,value in a.critic.state_dict().items():self.assertTrue(torch.equal(value,b.critic.state_dict()[key]))
        self.assertEqual(plain,audited)


if __name__=='__main__':unittest.main()
