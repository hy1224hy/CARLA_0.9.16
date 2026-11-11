import copy
import tempfile
import json
import unittest
from pathlib import Path
import numpy as np
import torch
from highway_research.config import StudyConfig
from highway_research.env import StudyEnv
from highway_research.learning import update,beta_at
from highway_research.matrix import definitions
from highway_research.runtime import ActorCritic,collect,train,load
from highway_research.evaluate import audit
from highway_research.report import mean_sd,paired_interval,build_report
from highway_research.learning import conjugate_gradient


class Contracts(unittest.TestCase):
    def config(self,**changes):return StudyConfig.default(1,42,**changes)

    def test_matrix_counts_unique_and_beta_zero_reused(self):
        rows=definitions()
        self.assertEqual(len(rows),155)
        self.assertEqual(len({r.run_id for r in rows}),155)
        self.assertEqual(sum(r.arm=='main' for r in rows),60)
        self.assertFalse(any(r.arm=='beta' and r.beta==0 for r in rows))

    def test_seed_sd_is_sample_sd_and_empty_is_missing(self):
        self.assertAlmostEqual(mean_sd([0,1])['sample_sd'],2**-.5)
        self.assertIsNone(mean_sd([])['mean'])
        self.assertEqual(paired_interval([1,1,1,1,1]),[1.,1.])

    def test_conjugate_gradient_solves_spd_system(self):
        matrix=torch.tensor([[3.,1.],[1.,2.]])
        target=torch.tensor([1.,2.]);result=conjugate_gradient(lambda v:matrix@v,target,10)
        torch.testing.assert_close(matrix@result,target)

    def test_l2_canonical_reward_restores_removed_risk(self):
        cfg=StudyConfig.default(2,42);cfg.task['risk_weight']=0.;cfg.validate()
        env=StudyEnv(cfg)
        try:
            env.reset(seed=4000000,options={'family':'follow'})
            _,reward,_,_,info=env.step(dict(acceleration=np.array([0.]),lane=0))
            self.assertGreater(info['original_return_step'],reward)
        finally:env.close()

    def test_reward_ablation_preserves_physics_and_terminal_definition(self):
        cfg=self.config(drop_components=['style','geometry'])
        env=StudyEnv(cfg);obs,_=env.reset(seed=4000000,options={'family':'follow'})
        try:
            obs,reward,_,_,info=env.step(dict(acceleration=np.array([0.]),lane=0))
            self.assertEqual(info['reward_components']['geometry'],0.)
            self.assertAlmostEqual(reward,sum(info['reward_components'].values()),places=6)
            self.assertIn('original_return_step',info)
        finally:env.close()

    def test_baselines_and_mutual_update_real_batch(self):
        torch.set_num_threads(1)
        for algorithm,mutual in [('ppo','off'),('ppo','both'),('a2c','off'),('trpo','off')]:
            cfg=self.config(algorithm=algorithm,mutual=mutual)
            cfg.task.update(hidden=16,epochs=1,minibatch=16,warmup_steps=0,ramp_steps=0);cfg.validate()
            model=ActorCritic(cfg.hidden,cfg.state_dim);peer=copy.deepcopy(model)
            with torch.no_grad():peer.acceleration_head.bias.add_(.2)
            env=StudyEnv(cfg)
            try:batch,_=collect(model,env,32,np.random.default_rng(5))
            finally:env.close()
            result=update(model,torch.optim.Adam(model.parameters(),lr=cfg.lr),batch,cfg,peer,.02,.02)
            self.assertTrue(np.isfinite(list(result.values())).all())
            self.assertTrue(all(p.grad is None for p in peer.parameters()))
            if algorithm=='trpo':self.assertLessEqual(result['actual_kl'],cfg.trpo_delta+1e-6)
            if algorithm=='a2c':self.assertEqual(result['optimizer_steps'],1.)

    def test_one_way_only_student_receives_mutual_coefficient(self):
        cfg=self.config(mutual='one_way');cfg.task.update(warmup_steps=0,ramp_steps=0)
        self.assertEqual(beta_at(0,cfg,'A'),(0.,0.))
        self.assertEqual(beta_at(0,cfg,'B'),(.02,.02))

    def test_zero_beta_matches_independent_ppo(self):
        torch.manual_seed(123);cfg=self.config();cfg.task.update(hidden=16,epochs=1,minibatch=16);cfg.validate()
        model=ActorCritic(16,cfg.state_dim);other=copy.deepcopy(model);peer=copy.deepcopy(model)
        env=StudyEnv(cfg)
        try:batch,_=collect(model,env,16,np.random.default_rng(42))
        finally:env.close()
        mutual=self.config(mutual='both');mutual.task=cfg.task.copy();mutual.validate()
        for candidate,config in [(model,cfg),(other,mutual)]:
            torch.manual_seed(100)
            update(candidate,torch.optim.Adam(candidate.parameters(),lr=cfg.lr),batch,config,peer,0.,0.)
        for a,b in zip(model.parameters(),other.parameters()):self.assertTrue(torch.equal(a,b))

    def test_trpo_rejected_zero_gradient_keeps_actor(self):
        torch.manual_seed(123);cfg=self.config(algorithm='trpo');cfg.task.update(hidden=16);cfg.validate()
        model=ActorCritic(16,cfg.state_dim);env=StudyEnv(cfg)
        try:batch,_=collect(model,env,16,np.random.default_rng(42))
        finally:env.close()
        for k in ['reward','value','next_value']:batch[k].zero_()
        before={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('critic.')}
        result=update(model,torch.optim.Adam(model.parameters(),lr=cfg.lr),batch,cfg,copy.deepcopy(model),0.,0.)
        self.assertEqual(result['trpo_accepted'],0.)
        for key,value in before.items():self.assertTrue(torch.equal(value,model.state_dict()[key]))

    def test_report_refuses_missing_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'study.json').write_text(json.dumps(dict(seeds=[42,43,44,45,46],runs=[],steps=500000)))
            cfg=self.config();cfg.run_id='L1_PPO_s42'
            for team in ['A','B']:
                folder=root/'audits'/team;folder.mkdir(parents=True)
                (folder/'metrics.json').write_text(json.dumps(dict(config=cfg.to_dict(),split='development',checkpoint='latest',
                    condition='nominal',execution='deterministic',action_seed=0,team=team,overall={'attack_success_rate':.5})))
            with self.assertRaisesRegex(ValueError,'Missing training seeds'):build_report(root)

    def test_training_resume_and_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=self.config(mutual='both')
            cfg.task.update(total_steps=128,rollout_steps=32,minibatch=32,epochs=1,hidden=16,eval_cases=2,eval_interval=128)
            cfg.validate();root=Path(directory)
            train(cfg,root/'run',max_rounds=1);train(cfg,root/'run',resume=True)
            result=audit(root/'run',root/'audit',cases=2)
            self.assertEqual(result['steps'],128)
            self.assertEqual(result['overall']['cases'],2)

    def test_rollout_does_not_reset_partial_episode(self):
        cfg=self.config();cfg.task.update(hidden=16);cfg.validate()
        env=StudyEnv(cfg);model=ActorCritic(16,cfg.state_dim);rng=np.random.default_rng(1)
        try:
            collect(model,env,1,rng);before=env.tracker.time
            collect(model,env,1,rng)
            self.assertGreater(env.tracker.time,before)
        finally:env.close()

    def test_resume_reproduces_uninterrupted_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=self.config()
            cfg.task.update(total_steps=128,rollout_steps=16,minibatch=16,epochs=1,hidden=16,eval_cases=2,eval_interval=128)
            cfg.validate();train(cfg,root/'whole')
            train(cfg,root/'split',max_rounds=2);train(cfg,root/'split',resume=True)
            a,b=load(root/'whole'/'latest.pt'),load(root/'split'/'latest.pt')
            for t in ['A','B']:
                for key,value in a['models'][t].items():self.assertTrue(torch.equal(value,b['models'][t][key]))


if __name__=='__main__':unittest.main()
