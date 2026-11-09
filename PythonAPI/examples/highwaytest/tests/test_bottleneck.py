from dataclasses import replace
import unittest
import numpy as np
import torch
from ..config import Config
from ..models.team import Team
from ..bottleneck.contract import task, inputs, RED_IDS, BLUE_IDS, OBS_DIM, STATE_DIM
from ..bottleneck.environment import BottleneckEnv

class BottleneckContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_three_real_roles_and_two_to_one_topology(self):
        env=BottleneckEnv(task())
        try:
            obs,info=env.reset(seed=12,options={'family':'lead_merge'})
            self.assertEqual(set(env.snapshot().vehicles),{0,1,2})
            self.assertEqual(RED_IDS,(0,)); self.assertEqual(BLUE_IDS,(1,2))
            self.assertEqual(obs.shape,(2,OBS_DIM));self.assertEqual(info['state'].shape,(STATE_DIM,))
            net=env.road.network.graph
            self.assertEqual(len(net['in']['merge']),2)
            self.assertEqual(len(net['out']['end']),1)
            self.assertEqual(len(env.reference.vehicles),1)
            self.assertEqual(len(env.normal_reference.vehicles),3)
            self.assertEqual(env.render().ndim,3)
        finally:env.close()

    def test_complementary_information_same_blue_B_observation(self):
        a,b=BottleneckEnv(task()),BottleneckEnv(task())
        try:
            oa,ia=a.reset(seed=1,options={'family':'lead_merge','red_target':16.})
            ob,ib=b.reset(seed=1,options={'family':'lead_merge','red_target':20.})
            np.testing.assert_array_equal(oa[1],ob[1])
            self.assertGreater(float(np.max(abs(oa[0]-ob[0]))),0)
            self.assertEqual(ia['links'][0,0],1)
            model=Team(Config(task=task()))
            self.assertEqual(model.distribution(torch.tensor(oa),torch.tensor(ia['links'])).loc.shape,(2,2))
        finally:a.close();b.close()

    def test_cloned_initial_conditions_include_red_controller_style(self):
        from ..bottleneck.evaluation import Backend
        bs=[Backend(task(),c,'lead_merge',123) for c in ('C0','C1','C2')]
        try:
            self.assertEqual([b.initial_contract for b in bs],[bs[0].initial_contract]*3)
            red=[b.frame.vehicles[0] for b in bs]
            self.assertEqual([v.position for v in red],[red[0].position]*3)
        finally:
            for b in bs:b.close()

    def test_new_config_roundtrip_does_not_change_old_defaults(self):
        from ..config import Task
        self.assertNotIn('bottleneck',Config().to_dict()['task'])
        c=Config(task=task()).validate();raw=c.to_dict()
        self.assertEqual(Config(**(raw|{'task':Task(**raw['task'])})).digest(),c.digest())

    def test_rear_layout_has_headway_before_taper_and_radio_contact(self):
        from ..bottleneck.scenarios import starts
        for seed in range(32):
            c=starts(np.random.default_rng(seed),'rear_merge',task())
            red,a,b=c['layout']
            self.assertGreaterEqual(red[0]-b[0],36.)
            self.assertLessEqual(a[0]-b[0],70.)
            self.assertGreater(a[0]-b[0],task().sensing_range)

    def test_qualification_requires_exit_flow_and_no_long_stop(self):
        env=BottleneckEnv(task())
        try:
            env.reset(seed=2)
            tracker=env.tracker
            frame=env.snapshot()
            from ..common.contract import SceneFrame
            vehicles={i:replace(v,position=(400.,v.position[1]),progress=400.) for i,v in frame.vehicles.items()}
            tracker.update(SceneFrame(vehicles,env.task.duration,frame.origin))
            tracker.normal_progress={1:600.,2:600.}
            self.assertFalse(tracker.summary()['qualified'])
            self.assertFalse(tracker.summary()['flow_retention_met'])
            tracker.normal_progress={1:400.,2:400.}
            self.assertTrue(tracker.summary()['qualified'])
            tracker.stopped_seconds[1]=4.
            self.assertFalse(tracker.summary()['qualified'])
        finally:env.close()

    def test_gate_rejects_same_case_names_with_different_initial_state(self):
        from ..training.mutual import compute_gates
        report=lambda h:{'families':{'lead_merge':{'case_keys':['lead_merge:1'],
            'case_initial_hashes':[h],'episodes':1,'bad_rate':0.,'quality':.2,'success_rate':1.}}}
        with self.assertRaises(ValueError):
            compute_gates({'A':report('x'),'B':report('y')},Config(task=task(),gate_cases_per_family=1))

    def test_probability_check_distinguishes_batch_rounding_from_policy_drift(self):
        from unittest.mock import patch
        from torch.distributions import Normal
        from ..training.ppo import check_on_policy
        cfg=Config(task=task(),exploration={'initial':[.1,.03],'minimum':[.1,.03],'maximum':[.6,.2]})
        m=Team(cfg)
        obs=torch.randn(4,2,25);links=torch.ones(4,2,1)
        zs=[];logs=[]
        with torch.no_grad():
            for o,l in zip(obs,links):
                d=m.distribution(o,l);z=d.loc+3*d.scale
                zs.append(z);logs.append(Team.log_prob(d,z))
        b={'obs':obs,'links':links,'z':torch.stack(zs),'logp':torch.stack(logs)}
        original=m.distribution
        def rounded(o,l,**kw):
            d=original(o,l,**kw)
            return Normal(d.loc+2e-7,d.scale) if o.ndim==3 else d
        with patch.object(m,'distribution',side_effect=rounded):
            _,result=check_on_policy(m,b)
            self.assertTrue(result['rowwise_checked'])
            self.assertLessEqual(result['rowwise_error'],1e-5)
        with torch.no_grad():m.actors[0][-1].bias.add_(.1)
        with self.assertRaises(RuntimeError):check_on_policy(m,b)

    def test_normal_and_scripted_controls_are_feasible(self):
        from ..bottleneck.evaluation import evaluate
        normal,_=evaluate(None,task(attack_weight=1),'gate',1)
        self.assertEqual(normal['overall']['success_rate'],1.)
        self.assertEqual(normal['overall']['qualified_attack_utility'],0.)
        self.assertEqual(normal['overall']['baseline_invalid_count'],0)
        attack,_=evaluate(None,task(attack_weight=1),'gate',1,script='both')
        self.assertEqual(attack['overall']['success_rate'],1.)
        self.assertGreater(attack['overall']['qualified_attack_utility'],0.)

    def test_bottleneck_demonstrations_are_training_only_local_inputs(self):
        from ..training.driving import demonstrations
        cfg=Config(task=task(duration=.4))
        data=demonstrations(cfg,2)
        self.assertEqual(data['obs'].shape,(4,2,25));self.assertEqual(data['actions'].shape,(4,2,2))
        self.assertTrue(all(s<900000 for s in data['seeds']))

    def test_teacher_kl_uses_exact_student_rows_and_masks(self):
        from unittest.mock import patch
        from ..training.ppo import collect,update
        from ..training.mutual import snapshots
        cfg=Config(task=task(duration=.4),epochs=1,minibatch=4)
        m=Team(cfg);teacher=snapshots({'A':m})['A']
        env=BottleneckEnv(cfg.task)
        try:
            batch,_,_,_=collect(m,env,4,np.random.default_rng(2),None)
            original=teacher.distribution
            def checked(obs,links):
                self.assertTrue(torch.equal(obs,batch['obs']))
                self.assertTrue(torch.equal(links,batch['links']))
                return original(obs,links)
            with patch.object(teacher,'distribution',side_effect=checked) as spy:
                update(m,torch.optim.Adam(m.parameters()),batch,cfg,teacher,torch.ones(4))
                self.assertEqual(spy.call_count,1)
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        finally:env.close()

    def test_new_protocol_exact_round_boundary_resume(self):
        import tempfile
        from pathlib import Path
        from ..config import ROOT
        from ..training.runner import run
        from ..training.checkpoint import read
        cfg=Config(task=task(duration=.4),total_train_steps=16,rollout_steps=4,
                   epochs=1,minibatch=4,gate_cases_per_family=1,refresh_rounds=1)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            run(cfg,a);run(cfg,b,max_rounds=1);run(cfg,b,resume=b/'latest.pt')
            x,y=read(a/'latest.pt'),read(b/'latest.pt')
            for team in ('A','B'):
                for k in x['models'][team]:
                    torch.testing.assert_close(x['models'][team][k],y['models'][team][k],rtol=0,atol=0)

    def test_study_entry_requires_its_own_calibration(self):
        from unittest.mock import patch
        import tempfile
        from pathlib import Path
        from ..config import ROOT,load
        from ..staged import prepare
        from ..bottleneck.__main__ import main
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            root=Path(tmp)/'study';prepare(root,42,32,32,task=task())
            self.assertEqual(load(root/'configs/normal.json').task.research_protocol,'bottleneck_paired_attack_v4')
            with patch('sys.argv',['bottleneck','bootstrap','--study',str(root)]):
                with self.assertRaisesRegex(ValueError,'calibrate'):main()

    def test_old_calibration_entry_cannot_label_four_car_run_as_v4(self):
        from unittest.mock import patch
        import tempfile,json
        from pathlib import Path
        from ..config import ROOT
        from ..research import main
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            config=Path(tmp)/'config.json';config.write_text(json.dumps(Config(task=task(duration=.4)).to_dict()))
            with patch('sys.argv',['research','validate','--config',str(config),'--cases-per-family','1','--output',str(Path(tmp)/'out.json')]):
                with self.assertRaisesRegex(ValueError,'bottleneck'):main()

    def test_old_backend_refuses_three_car_task(self):
        from ..environment.conditions import HighwayConditionBackend
        with self.assertRaisesRegex(ValueError,'v3'):
            HighwayConditionBackend(task(),'C0','parallel',1)

    def test_source_scan_prunes_experiment_and_distribution_directories(self):
        import os
        from pathlib import Path
        from unittest.mock import patch
        from ..training.checkpoint import source_digest
        before=source_digest();original=os.scandir;seen=[]
        def checked(path):
            if Path(path).name in {'runs','dist','__pycache__'}:
                seen.append(str(path));raise PermissionError('ignored tree must not be traversed')
            return original(path)
        with patch('os.scandir',side_effect=checked),patch.object(Path,'rglob',side_effect=AssertionError('source scan must prune before traversal')):
            self.assertEqual(source_digest(),before)
        self.assertEqual(seen,[])

    def test_cli_does_not_silently_ignore_zero_round_limit_or_case_count(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        from ..config import ROOT,load
        from ..staged import prepare
        from ..training.checkpoint import source_digest
        from ..bottleneck.__main__ import main
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            root=Path(tmp)/'study';prepare(root,42,32,32,task=task())
            cfg=load(root/'configs/normal.json')
            receipt={'passed':True,'source_sha256':source_digest(),'config_sha256':cfg.digest(),'cases_per_family':8}
            (root/'calibration.json').write_text(json.dumps(receipt))
            with patch('sys.argv',['bottleneck','normal','--study',str(root),'--max-rounds','0']),patch('highwaytest.staged.main'):
                with self.assertRaises(ValueError):main()
            receipt['cases_per_family']=1;(root/'calibration.json').write_text(json.dumps(receipt))
            with patch('sys.argv',['bottleneck','calibrate','--study',str(root),'--cases-per-family','8']):
                with self.assertRaises(ValueError):main()

if __name__=='__main__': unittest.main()
