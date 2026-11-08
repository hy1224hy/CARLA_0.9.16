"""Focused no-simulator regression checks for the team-only critic repair."""
import copy
import unittest
import tempfile
import torch
from models.actor_critic import Actor, AgentConditionedCritic
from training.value_normalization import ValueNormalizer
from training.mut_mappo_learning import TeamLearner
from training.mut_mappo_checkpoint import CheckpointStore, _validate_payload
from training.checkpoint_protocol import CheckpointProtocolError
from tests.test_mut_mappo_learning import buffer_for


class BlueTeamCriticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.actors={i:Actor(20,2,hidden=16).state_dict() for i in (0,1)}
        self.base=AgentConditionedCritic(40,2,16)
        self.norm=ValueNormalizer()
        self.norm.update(torch.tensor([2.,5.,8.]))
        self.value={'critic':copy.deepcopy(self.base.state_dict()),'value_normalizer':copy.deepcopy(self.norm.state_dict())}
        self.cfg={'method':'off','algorithm':'mappo','hidden':16,'beta':0.,'ppo_epochs':2,'minibatch_size':8,'shuffle':False,'target_kl':0.,'critic_scope':'blue_team','critic_initialization':'base_checkpoint'}

    def learner(self,name='A',value=True):
        return TeamLearner(name,42,self.actors,self.cfg,initial_value=self.value if value else None)

    def test_exact_base_predictions_role_mapping_and_red_slot_isolation(self):
        team=self.learner()
        state=torch.randn(7,80,requires_grad=True)
        ids=torch.tensor([2,3,2,3,3,2,2])
        actual=team.critic(state,ids)
        expected=self.base(state[:,40:80],ids-2)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        changed=state.detach().clone();changed[:,:40]+=100
        torch.testing.assert_close(team.critic(changed,ids),actual,rtol=0,atol=0)
        self.assertEqual(team.critic.net[0].in_features,42)
        actual.sum().backward()
        self.assertEqual(torch.count_nonzero(state.grad[:,:40]).item(),0)
        for aid in (2,3):
            torch.testing.assert_close(team.value_normalizer.denormalize(team.critic(state,aid)),self.norm.denormalize(self.base(state[:,40:],aid-2)),rtol=0,atol=0)
        for bad in (0,1,True,torch.tensor([0]*7),torch.tensor([2.]*7)):
            with self.assertRaises((ValueError,TypeError)):
                team.critic(state,bad)

    def test_separate_team_storage_and_joint_ppo_update(self):
        a,b=self.learner(),self.learner('B')
        for pa,pb in zip(a.critic.parameters(),b.critic.parameters()):
            torch.testing.assert_close(pa,pb,rtol=0,atol=0)
            self.assertNotEqual(pa.data_ptr(),pb.data_ptr())
        actors=copy.deepcopy(a.export_actors());critic=copy.deepcopy(a.critic.state_dict());peer=copy.deepcopy(b.state_dict())
        metrics=a.update(buffer_for(a))
        self.assertTrue(any(not torch.equal(critic[k],v) for k,v in a.critic.state_dict().items()))
        for aid in (2,3):
            self.assertTrue(any(not torch.equal(actors[aid][k],v) for k,v in a.actors[aid].state_dict().items()))
        for k,v in b.critic.state_dict().items():torch.testing.assert_close(v,peer['critic'][k],rtol=0,atol=0)
        self.assertEqual(metrics['critic_input_scope'],'own_blue_team_observations_plus_local_id')

    def test_strict_initialization_and_resume_topology(self):
        with self.assertRaises(ValueError):self.learner(value=False)
        a=self.learner();a.update(buffer_for(a));b=self.learner();b.load_state_dict(a.state_dict())
        for k,v in a.critic.state_dict().items():torch.testing.assert_close(v,b.critic.state_dict()[k],rtol=0,atol=0)
        old=TeamLearner('A',42,self.actors,{'method':'off','algorithm':'mappo','hidden':16,'beta':0.})
        self.assertEqual(old.critic.net[0].in_features,84)
        with self.assertRaises(ValueError):b.load_state_dict(old.state_dict())
        broken=copy.deepcopy(self.value);broken['critic']=old.critic.state_dict()
        with self.assertRaises((ValueError,RuntimeError)):TeamLearner('A',42,self.actors,self.cfg,initial_value=broken)

    def test_real_checkpoint_roundtrip_accepts_team42_and_rejects_mislabeled_ids(self):
        a,b=self.learner(),self.learner('B')
        payload=dict(teams={'A':a.state_dict(),'B':b.state_dict()},teachers={'A':None,'B':None},
            gate={},completed_round=0,next_team='A',counters={},last_metrics={})
        with tempfile.TemporaryDirectory() as folder:
            store=CheckpointStore(folder,{'test':'blue_team_42'})
            store.save(payload)
            decoded=store.load()
            self.learner().load_state_dict(decoded['teams']['A'])
            torch.testing.assert_close(decoded['teams']['A']['critic']['net.0.weight'],self.base.net[0].weight,rtol=0,atol=0)
        bad=copy.deepcopy(payload);bad['teams']['A']['model_spec']['critic_id_dim']=4
        with self.assertRaises(CheckpointProtocolError):_validate_payload(bad)


if __name__=='__main__':unittest.main()
