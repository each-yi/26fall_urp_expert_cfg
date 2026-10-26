import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ReportTest(unittest.TestCase):
    def test_end_to_end_synthetic_report(self):
        # Synthetic outputs test evaluation plumbing, never model replication.
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            rows = [{'id':str(i),'dataset':'vqa_rad','split':'test','image':'fake.png',
                     'question':'Test?','answer':answer,'answer_type':kind}
                    for i,(answer,kind) in enumerate([('yes','closed'),('no','closed'),('left lung','open')])]
            preds = [{**r,'prediction':pr,'entropy':h,'model':'phi35','setting':'downstream',
                      'truncated':False} for r,pr,h in zip(rows,['yes','yes','lung'],[.1,.8,.6])]
            data,pred,out = d/'test.jsonl',d/'pred.jsonl',d/'report'
            for path,values in [(data,rows),(pred,preds)]:
                path.write_text(''.join(json.dumps(r)+'\n' for r in values))
            subprocess.run([sys.executable,'-m','baseline.evaluate','--data',str(data),
                '--predictions',str(pred),'--output',str(out),'--plots'],check=True,capture_output=True)
            result = json.loads((out/'report.json').read_text())
            self.assertTrue(result['coverage']['complete'])
            self.assertEqual(result['datasets']['vqa_rad']['metrics']['overall'],50)
            self.assertEqual(result['datasets']['vqa_rad']['comparison']['overall']['paper'],82.04)
            self.assertEqual(result['datasets']['vqa_rad']['calibration']['entropy']['ece_t'],None)
            self.assertTrue((out/'vqa_rad_roc.png').is_file())
            self.assertIn('82.04',(out/'comparison.md').read_text())


if __name__=='__main__':
    unittest.main()
