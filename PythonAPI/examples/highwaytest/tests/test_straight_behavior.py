import unittest
from highwaytest.straight.behavior import classify,build_matrix,CLASSES


class BehaviorTests(unittest.TestCase):
    def test_classify_priority(self):
        self.assertEqual(classify(dict(invalid=True,collision=True)),'invalid')
        self.assertEqual(classify(dict(invalid=False,collision=True)),'collision')
        self.assertEqual(classify(dict(invalid=False,collision=False,cut_in=True,stop_qualified=False,blocking_time=2.)),'strong')
        self.assertEqual(classify(dict(invalid=False,collision=False,cut_in=False,stop_qualified=True)),'strong')
        self.assertEqual(classify(dict(invalid=False,collision=False,cut_in=False,stop_qualified=False,blocking_time=1.5)),'mild')
        self.assertEqual(classify(dict(invalid=False,collision=False,cut_in=False,stop_qualified=False,blocking_time=0.)),'none')

    def test_build_matrix_row_normalized(self):
        entries=[(1,'independent',42,[dict(invalid=False,collision=False,cut_in=False,stop_qualified=False,blocking_time=2.)]),
                 (1,'independent',42,[dict(invalid=True,collision=False)])]
        result=build_matrix(entries)
        self.assertEqual(len(result['rows']),1)
        row=result['rows'][0]
        self.assertEqual(row['level'],1);self.assertEqual(row['method'],'independent')
        self.assertEqual(row['seeds'],[42])
        self.assertAlmostEqual(row['classes']['mild'],.5)
        self.assertAlmostEqual(row['classes']['invalid'],.5)
        self.assertAlmostEqual(row['classes']['collision'],0.)
        self.assertAlmostEqual(sum(row['classes'].values()),1.)

    def test_build_matrix_groups_by_intent_and_method(self):
        entries=[(1,'independent',42,[dict(invalid=True,collision=False)]),
                 (1,'mutual',43,[dict(invalid=False,collision=True)]),
                 (3,'mutual',43,[dict(invalid=False,collision=True)])]
        result=build_matrix(entries)
        keys={(r['level'],r['method']) for r in result['rows']}
        self.assertEqual(keys,{(1,'independent'),(1,'mutual'),(3,'mutual')})
        self.assertEqual(set(CLASSES),{'invalid','collision','strong','mild','none'})

    def test_classify_defaults_missing_fields_to_none(self):
        self.assertEqual(classify(dict()),'none')


if __name__=='__main__':unittest.main()
