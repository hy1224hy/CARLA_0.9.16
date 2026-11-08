import copy
import math
import unittest
from evaluation.critical_interaction_metrics import PROTOCOL_DEFAULT, validate_protocol, obb_ttc, analyze_trace


def snap(x, y, vx=0, vy=0, yaw=0, bx=0, by=0, byaw=0):
    return dict(location=dict(x=x,y=y),rotation=dict(yaw=yaw),velocity=dict(x=vx,y=vy),
        bounding_box=dict(extent=dict(x=2,y=1),location=dict(x=bx,y=by),rotation=dict(yaw=byaw)))


def trace(n=3):
    init={'0':snap(0,0,10),'2':snap(10,0,0)}
    rows=[dict(step=0,kinematics=init,active_mask={'0':True,'2':True})]
    for step in range(1,n+1):
        rows.append(dict(step=step,kinematics={'0':snap(step,0,10),'2':snap(10,0,0)},
            active_mask={'0':True,'2':True},next_active_mask={'0':step<n,'2':step<n}))
    return rows


class MetricsTests(unittest.TestCase):
    def test_constant_velocity_rear_end(self):
        self.assertAlmostEqual(obb_ttc(snap(0,0,10),snap(10,0),5),.6)
    def test_side_by_side_and_nonclosing(self):
        self.assertIsNone(obb_ttc(snap(0,0,10),snap(0,3,10),5))
        self.assertIsNone(obb_ttc(snap(0,0),snap(10,0,10),5))
    def test_overlap_and_horizon(self):
        self.assertEqual(obb_ttc(snap(0,0),snap(3,0),5),0)
        self.assertIsNone(obb_ttc(snap(0,0,1),snap(10,0),5))
    def test_rotated_offset(self):
        # local forward offset 2 becomes world y+2, reaching the other box.
        self.assertEqual(obb_ttc(snap(0,0,yaw=90,bx=2),snap(0,5),5),0)
        self.assertIsNone(obb_ttc(snap(0,0,yaw=90),snap(0,5),5))
    def test_missing_bbox_rejected_and_not_safe(self):
        broken=snap(0,0);broken.pop('bounding_box')
        with self.assertRaises(ValueError):obb_ttc(broken,snap(10,0),5)
        rows=trace();rows[1]['kinematics']['0'].pop('bounding_box')
        result=analyze_trace(rows,PROTOCOL_DEFAULT)
        self.assertEqual(result['status'],'MISSING_GEOMETRY')
        self.assertIsNone(result['critical_event'])
    def test_three_steps_duration_and_no_pair_double_count(self):
        result=analyze_trace(trace(),PROTOCOL_DEFAULT)
        self.assertEqual(result['status'],'COMPLETE')
        self.assertTrue(result['critical_event'])
        self.assertAlmostEqual(result['tet_s']['1.5'],.3)
        self.assertAlmostEqual(result['events'][0]['low_ttc_duration_s'],.3)
        self.assertTrue(result['events'][0]['censored'])
    def test_inactive_missing_actors_not_counted(self):
        rows=trace();rows[1]['next_active_mask']['2']=False;rows[2]['next_active_mask']['2']=False;rows[2]['active_mask']['2']=False;rows[2]['kinematics'].pop('2')
        rows[3]['active_mask']['2']=False;rows[3]['kinematics'].pop('2')
        result=analyze_trace(rows,PROTOCOL_DEFAULT)
        self.assertEqual(result['status'],'COMPLETE')
        self.assertFalse(result['critical_event'])
        self.assertEqual(result['pair_coverage']['expected'],1)
    def test_gaps_and_protocol_rejected(self):
        rows=trace();rows.pop(2)
        self.assertEqual(analyze_trace(rows,PROTOCOL_DEFAULT)['status'],'INCOMPLETE_TRACE')
        for patch in ({'dt':0},{'dt':float('nan')},{'unknown':1},{'min_event_duration_s':-1}):
            with self.assertRaises(ValueError):validate_protocol(dict(PROTOCOL_DEFAULT,**patch))
    def test_c0_not_applicable(self):
        rows=trace()
        for row in rows:
            row['kinematics'].pop('2',None);row['active_mask'].pop('2',None)
            row.get('next_active_mask',{}).pop('2',None)
        self.assertEqual(analyze_trace(rows,PROTOCOL_DEFAULT)['status'],'NOT_APPLICABLE')

class ExtendedMetricsTests(unittest.TestCase):
    def test_multiple_pairs_do_not_multiply_scenario_exposure(self):
        rows=trace()
        for r in rows:
            r['active_mask']['3']=r['active_mask']['2']
            r.get('next_active_mask',{})['3']=r.get('next_active_mask',{}).get('2',True)
            r['kinematics']['3']=copy.deepcopy(r['kinematics']['2'])
        result=analyze_trace(rows,PROTOCOL_DEFAULT)
        self.assertAlmostEqual(result['tet_s']['1.5'],.3)
        self.assertEqual(result['pair_coverage']['expected'],6)
        self.assertEqual(len(result['events']),2)
        self.assertIn('1.0',result['tet_s'])
    def test_merge_counts_only_low_ttc_samples(self):
        rows=trace(8)
        rows[4]['kinematics']['2']['location']['y']=10
        rows[5]['kinematics']['2']['location']['y']=10
        for row in rows[1:]:
            row['kinematics']['2']['location']['x']=20
            row['kinematics']['0']['velocity']['x']=20
        result=analyze_trace(rows,PROTOCOL_DEFAULT)
        self.assertEqual(len(result['events']),1)
        self.assertAlmostEqual(result['events'][0]['low_ttc_duration_s'],.6)
        self.assertAlmostEqual(result['events'][0]['span_duration_s'],.8)
    def test_initial_geometry_and_observed_response(self):
        rows=trace()
        initial=rows[0].pop('kinematics')
        rows[0]['case_geometry']={'realized_initial_states':initial}
        rows[1]['kinematics']['0']['velocity']['x']=9
        result=analyze_trace(rows,PROTOCOL_DEFAULT)
        self.assertAlmostEqual(result['red_response']['0']['max_deceleration_mps2'],10)
        self.assertEqual(result['red_response']['0']['events'][0]['blue_id'],2)
    def test_incomplete_tail_is_not_safe(self):
        rows=trace();rows.pop()
        result=analyze_trace(rows,PROTOCOL_DEFAULT)
        self.assertEqual(result['status'],'INCOMPLETE_TRACE')
        self.assertIsNone(result['critical_event'])
    def test_box_local_rotation_changes_extent_orientation(self):
        self.assertIsNone(obb_ttc(snap(0,0),snap(0,2.5),5))
        self.assertEqual(obb_ttc(snap(0,0,byaw=90),snap(0,2.5),5),0)

class LifecycleAndContactTests(unittest.TestCase):
    def test_reset_to_first_decision_cannot_silently_drop_a_live_blue(self):
        for missing_slot in (True,False):
            rows=trace()
            for row in rows[1:]:
                row['kinematics'].pop('2')
                if missing_slot:
                    row['active_mask'].pop('2')
                    row['next_active_mask'].pop('2')
                else:
                    row['active_mask']['2']=False
                    row['next_active_mask']['2']=False
            with self.subTest(missing_slot=missing_slot):
                result=analyze_trace(rows,PROTOCOL_DEFAULT)
                self.assertEqual(result['status'],'INCOMPLETE_TRACE')
                self.assertIsNone(result['critical_event'])
                self.assertIsNone(result['tet_s']['1.5'])
    def test_reset_lifecycle_mask_requires_boolean_values(self):
        rows=trace();rows[0]['active_mask']['2']=1
        self.assertEqual(analyze_trace(rows,PROTOCOL_DEFAULT)['status'],'INCOMPLETE_TRACE')
    def test_single_contact_is_critical_despite_short_episode(self):
        rows=trace(1)
        rows[1]['kinematics']['2']['location']['x']=3
        result=analyze_trace(rows,PROTOCOL_DEFAULT)
        self.assertTrue(result['critical_event'])
        self.assertEqual(result['events'][0]['kind'],'geometric_contact')
    def test_isolated_noise_does_not_meet_continuous_threshold(self):
        rows=trace(5)
        for row in rows[1:]: row['kinematics']['2']['location']['x']=15
        rows[2]['kinematics']['2']['location']['y']=10
        rows[4]['kinematics']['2']['location']['y']=10
        result=analyze_trace(rows,PROTOCOL_DEFAULT)
        self.assertFalse(result['critical_event'])
        self.assertAlmostEqual(result['tet_s']['1.5'],.3)
    def test_lifecycle_disagreement_is_unmeasured(self):
        rows=trace();rows[1]['next_active_mask']['2']=False
        self.assertEqual(analyze_trace(rows,PROTOCOL_DEFAULT)['status'],'INCOMPLETE_TRACE')

class ExposureTerminationTests(unittest.TestCase):
    def test_pair_ends_early_while_other_actor_keeps_running(self):
        rows=trace(8)
        for row in rows[1:]:
            step=row['step']
            row['active_mask']['2']=step<=3
            row['next_active_mask']['2']=step<3
            if step>3: row['kinematics'].pop('2')
        result=analyze_trace(rows,PROTOCOL_DEFAULT)
        self.assertEqual(result['status'],'COMPLETE')
        self.assertTrue(result['events'][0]['censored'])
    def test_boolean_step_is_not_a_real_frame(self):
        rows=trace();rows[1]['step']=True
        self.assertEqual(analyze_trace(rows,PROTOCOL_DEFAULT)['status'],'INCOMPLETE_TRACE')

if __name__=='__main__':unittest.main()
