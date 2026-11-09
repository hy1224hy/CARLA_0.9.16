"""Consequential checks of reward incentives and geometry, without a simulator."""
import copy
import math
import unittest
from environment.mut_mappo_critical_reward import CriticalTeamReward, validate_settings


def snapshot(x, y=0., vx=3., vy=0., yaw=0.):
    return dict(location=dict(x=x, y=y), rotation=dict(yaw=yaw), velocity=dict(x=vx, y=vy),
        bounding_box=dict(extent=dict(x=2., y=1.), location=dict(x=0., y=0.), rotation=dict(yaw=0.)))


class CriticalRewardTests(unittest.TestCase):
    def test_v4_preserves_original_red_corridors_and_all_symmetric_templates(self):
        import json
        from pathlib import Path
        from utils.build_mut_mappo_interaction_cases_v4 import build_cases, ROOT
        manifest=build_cases()
        self.assertEqual({k:len(v) for k,v in manifest['splits'].items()},dict.fromkeys(['train','gate','selection','test'],24))
        for split in manifest['splits']:
            cases=[c for c in manifest['cases'] if c['split']==split]
            self.assertEqual({c['direction'] for c in cases},{'northbound','southbound','eastbound','westbound'})
            self.assertEqual(sum(c['family']=='blocking' for c in cases),8)
            for c in cases:
                base=json.loads((ROOT/c['base_scenario']).read_text(encoding='utf-8'))
                self.assertAlmostEqual(base['route_length_m'],32.)
                self.assertEqual(set(base['agents']),{'agent0','agent1'})
                for spec in c['blue_spawn_offsets'].values():
                    self.assertTrue(spec['forward_m']==-8. or spec['forward_m']>=10.)

    def args(self, blue_x=6.5, blue_y=0., blue_vx=1.):
        return dict(step=1, snapshots={0:snapshot(0), 2:snapshot(blue_x, blue_y, blue_vx)},
            active={0:True, 1:False, 2:True, 3:False},
            driving_context={2:dict(legal_forward=True, route_speed_mps=blue_vx,
                front_blocker=False, near_goal=False, terminal=False)},
            progress=dict.fromkeys(range(4), 0.), reached={}, eligible=True,
            safety={}, costs={}, base_details={}, reference=None)

    def test_moderate_front_conflict_but_not_safe_parallel_or_rear_chase(self):
        for x,y,v,positive in ((6.5,0,1,True), (0,3.5,3,False), (-6.5,0,5,False)):
            reward=CriticalTeamReward()
            _,p=reward.step(**self.args(x,y,v))
            self.assertEqual(p['critical']>0, positive)

    def test_contact_and_severe_risk_suppress_entire_team_positive_credit(self):
        args=self.args()
        args['snapshots'].update({1:snapshot(0, 4),3:snapshot(4.4,4,1)})
        args['active'].update({1:True,3:True})
        args['driving_context'][3]=dict(args['driving_context'][2])
        _,parts=CriticalTeamReward().step(**args)
        self.assertEqual(parts['critical'],0.)
        self.assertLess(parts['severe'],0.)
        _,parts=CriticalTeamReward().step(**self.args(3.9))
        self.assertEqual(parts['critical'],0.)

    def test_cap_and_final_qualification_are_independent_of_early_credit(self):
        reward=CriticalTeamReward()
        total=0.
        for step in range(1,51):
            args=self.args(); args['step']=step
            _,p=reward.step(**args); total+=p['critical']
        self.assertAlmostEqual(total,5.)
        self.assertEqual(reward.metrics(True)['safe_critical_event'],1.)
        self.assertEqual(reward.metrics(False)['safe_critical_event'],0.)

    def test_protection_missing_geometry_and_mirror(self):
        args=self.args(); args['eligible']=False
        self.assertEqual(CriticalTeamReward().step(**args)[1]['critical'],0.)
        args=self.args(); del args['snapshots'][2]['bounding_box']
        with self.assertRaises(ValueError): CriticalTeamReward().step(**args)
        args=self.args(); mirrored=copy.deepcopy(args)
        for s in mirrored['snapshots'].values():
            s['location']['x']*=-1; s['velocity']['x']*=-1; s['rotation']['yaw']=180.
        a=CriticalTeamReward().step(**args)[1]; b=CriticalTeamReward().step(**mirrored)[1]
        self.assertEqual(a,b)

    def test_pace_and_first_event_cost_and_validation(self):
        args=self.args(20,3.5,0.)
        args['safety']={2:{'collision':True}}; args['eligible']=False
        _,p=CriticalTeamReward().step(**args)
        self.assertEqual(p['safety_event'],-50.)
        args=self.args(20,3.5,0.)
        _,p=CriticalTeamReward().step(**args)
        self.assertLess(p['pace'],0.)
        args['driving_context'][2]['front_blocker']=True
        self.assertEqual(CriticalTeamReward().step(**args)[1]['pace'],0.)
        for bad in ({'critical_weight':float('nan')},{'severe_ttc_s':2.},{'exposure_cap_s':0}, {'surprise':1}):
            with self.assertRaises(ValueError): validate_settings(bad)

    def test_online_event_and_exposure_match_frozen_offline_geometry(self):
        from tests.test_critical_interaction_metrics import trace
        from evaluation.critical_interaction_metrics import analyze_trace, PROTOCOL_DEFAULT
        rows=trace()
        reward=CriticalTeamReward()
        for row in rows[1:]:
            args=self.args(); args.update(step=row['step'],
                snapshots={int(k):v for k,v in row['kinematics'].items()},
                active={int(k):v for k,v in row['active_mask'].items()})
            reward.step(**args)
        independent=analyze_trace(rows,PROTOCOL_DEFAULT)
        self.assertEqual(independent['status'],'COMPLETE')
        self.assertEqual(reward.metrics(True)['critical_event'],independent['critical_event'])
        self.assertAlmostEqual(reward.tet_s,independent['tet_s']['1.5'])
        self.assertAlmostEqual(reward.tit_s,independent['tit_s']['1.5'])

    def test_selection_prefers_criticality_over_red_failure_count(self):
        import tempfile
        from tests.test_mut_mappo_selection import records, candidate
        from evaluation.mut_mappo_selection import select_and_export
        rows=records('A',gain=8)+records('B',gain=0)
        for row in rows:
            row['safe_critical_event']=float(row['team']=='B' and row['condition']=='C2')
        with tempfile.TemporaryDirectory() as directory:
            result=select_and_export([candidate(rows=rows)],
                {'evaluation':{'selection_max_bad_rate':.125,'selection_metric':'safe_critical_event'},
                 'research_status':'candidate'},directory)
        self.assertEqual(result['selected']['team'],'B')

    def test_critical_gate_obeys_existing_safety_and_evidence_thresholds(self):
        from training.mut_mappo_learning import GateState
        gate=GateState({'quality_metric':'critical','gate_min_cases':8,'gate_min_q_gap':.25,
                        'gate_max_bad_rate':.125})
        stats={t:{'blocking':dict(n_all=8,q_count=q,b_count=0,case_keys=list(range(8)),
                                 teacher_hash=t,version=1)} for t,q in (('A',0),('B',4))}
        self.assertEqual(gate.refresh({'A':'A','B':'B'},stats,version=1,completed_round=1),
                         {'A':{'blocking':1},'B':{'blocking':0}})

    def test_partial_critical_measurement_does_not_rank_as_complete(self):
        import tempfile
        from tests.test_mut_mappo_selection import records, candidate
        from evaluation.mut_mappo_selection import select_and_export
        rows=records('A')+records('B')
        for row in rows: row['safe_critical_event']=1.
        del rows[2]['safe_critical_event']
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'measured'):
                select_and_export([candidate(rows=rows)],
                    {'evaluation':{'selection_metric':'safe_critical_event'},'research_status':'candidate'},directory)

    def test_frozen_lane_traffic_uses_no_client_and_brakes_for_real_front_gap(self):
        import carla
        from types import SimpleNamespace
        from unittest.mock import patch
        from environment.frozen_lane_traffic import FrozenLaneTraffic, following_target
        class Vehicle:
            is_alive=True
            bounding_box=carla.BoundingBox(carla.Location(),carla.Vector3D(2.,1.,1.))
            def __init__(self,x,y): self.transform=carla.Transform(carla.Location(x=x,y=y))
            def get_transform(self): return self.transform
            def get_location(self): return self.transform.location
            def get_velocity(self): return carla.Vector3D(x=2.5)
            def get_control(self): return carla.VehicleControl()
            def get_world(self): return None
        class WP:
            road_id=44;lane_id=1
            def __init__(self,x,y):self.transform=carla.Transform(carla.Location(x=x,y=y))
            def next(self,distance):return [WP(self.transform.location.x+distance,self.transform.location.y)]
        env=SimpleNamespace(vehicles=[None,None,Vehicle(0,0),Vehicle(0,3.5)],timestep=.1,
                            road_reference_waypoints={2:WP(0,0),3:WP(0,3.5)})
        case={'blue_spawn_offsets':{f'agent{i}':{'initial_speed_mps':2.5} for i in (2,3)}}
        with patch.object(carla,'Client',side_effect=AssertionError('offline test opened Client')):
            traffic=FrozenLaneTraffic(env,case)
            free=traffic.controls({2:True,3:True})
            self.assertAlmostEqual(float(free[2][1]),0.)
            env.vehicles[0]=Vehicle(5.,0.)
            blocked=traffic.controls({2:True,3:True})
            self.assertEqual(float(blocked[2][0]),0.)
            self.assertEqual(float(blocked[2][2]),1.)
        self.assertEqual(following_target(2.5,2.5),2.5)
        self.assertLess(following_target(2.5,2.5,2.,0.),2.5)


if __name__=='__main__': unittest.main()
