import json
import tempfile
import unittest
from pathlib import Path
from baseline.metrics import score,ece,calibration,uncertainty,fit_temperature
from baseline.data import load_data,audit_splits
from baseline.evaluate import evaluate
from baseline.auxiliary import contains
from baseline.paper import TABLE1,TABLE3,TABLE6


def row(id='1',**kwargs):
    return {'id':id,'dataset':'vqa_rad','split':'test','image':'one.png',
            'question':'Is it enlarged?','answer':'yes','answer_type':'closed',**kwargs}


class MetricsTest(unittest.TestCase):
    def test_open_recall(self):
        self.assertAlmostEqual(score('left lung','The lung is clear','open'),.5)
        self.assertEqual(score('giant cells and fibrosis','Giant cells and fibrosis.','open'),1)
        self.assertEqual(score('no no','no','open'),.5)

    def test_closed(self):
        self.assertEqual(score('yes','Yes, enlarged.','closed'),1)
        self.assertEqual(score('no','Yes but no.','closed'),0)
        self.assertEqual(score('no','unknown','closed'),0)

    def test_calibration_edges(self):
        self.assertAlmostEqual(ece([0,1],[0,1]),0)
        d = calibration([0,1],[0,1],1)
        self.assertEqual(d['auc'],1)
        self.assertLess(d['bs_t'],1e-10)
        self.assertEqual(uncertainty([0,1],[.1,.9])['auc'],1)
        self.assertIsNone(uncertainty([1,1],[.1,.9])['auc'])
        with self.assertRaises(ValueError):
            fit_temperature([1,1],[.4,.6])
        with self.assertRaises(ValueError):
            calibration([0,1],[float('nan'),.4],1)

    def test_temperature_fit(self):
        t = fit_temperature([0,1,0,1],[.4,.6,.45,.55])
        self.assertLess(t,1)

    def test_coverage(self):
        pred = row(prediction='yes',entropy=.2,model='phi35',setting='zero_med')
        c = evaluate([pred],[row(),row('2')])
        self.assertFalse(c['complete'])
        with self.assertRaises(ValueError):
            evaluate([pred,pred],[row()])
        with self.assertRaises(ValueError):
            evaluate([{**pred,'answer':'no'}],[row()])

    def test_data_validation_and_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'d.jsonl'
            p.write_text(json.dumps(row())+'\n')
            self.assertEqual(len(load_data(p,'test')),1)
            with self.assertRaises(ValueError):
                load_data(p,'train')
        with self.assertRaises(ValueError):
            audit_splits([row(split='train')],[row()])

    def test_retrieval_phrase_boundaries(self):
        self.assertFalse(contains('ct','duct'))
        self.assertTrue(contains('left lung','The LEFT lung is clear.'))
        self.assertFalse(contains('left lung','lung on the left'))

    def test_paper_values(self):
        self.assertEqual(TABLE1['phi35'][0],(69.27,90.44,82.04))
        self.assertEqual(TABLE3['phi3_pubmed'],(73.90,67.31,54.38))
        self.assertEqual(TABLE6['phi35_finetune_entropy'][0],(.068,.024,.066,.958))


if __name__=='__main__':
    unittest.main()
