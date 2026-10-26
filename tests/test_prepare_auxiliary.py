import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image


class ArtifactTest(unittest.TestCase):
    def run_module(self,module,*args):
        return subprocess.run([sys.executable,'-m',module,*map(str,args)],check=True,capture_output=True,text=True)

    def test_slake_filter_and_image_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            Image.new('RGB',(8,8)).save(d/'one.png')
            raw = [{'qid':1,'img_name':'one.png','q_lang':'en','question':'Normal?',
                    'answer':'yes','answer_type':'CLOSED'},
                   {'qid':2,'img_name':'one.png','q_lang':'zh','question':'正常?',
                    'answer':'是','answer_type':'CLOSED'}]
            (d/'raw.json').write_text(json.dumps(raw))
            self.run_module('baseline.prepare','--input',d/'raw.json','--output',d/'prepared.jsonl',
                            '--image-root',d,'--dataset','slake','--split','test')
            records = [json.loads(x) for x in (d/'prepared.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),1)
            self.assertEqual(records[0]['id'],'1')
            self.assertEqual(records[0]['answer_type'],'closed')

    def test_auxiliary_paper_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            r = {'dataset':'vqa_rad','answer':'left lung','answer_type':'open',
                 'retrieved':{'image':[{'caption':'The left lung is abnormal','score':.8}]}}
            (d/'retrieval.jsonl').write_text(json.dumps(r)+'\n')
            (d/'keywords.jsonl').write_text(json.dumps({'gold':['lung','MRI'],'predicted':['lung','CT']})+'\n')
            self.run_module('baseline.auxiliary','--retrieval',d/'retrieval.jsonl',
                            '--keywords',d/'keywords.jsonl','--output',d/'out.json')
            out = json.loads((d/'out.json').read_text())
            self.assertEqual(out['retrieval']['vqa_rad']['image']['top_k']['1']['percent'],100)
            self.assertEqual(out['retrieval']['vqa_rad']['image']['paper_top1_percent'],53.07)
            self.assertEqual(out['keywords']['run']['f1'],50)
            self.assertEqual(out['keywords']['paper']['f1'],93.76)

    def test_notebook_structure_and_python_syntax(self):
        import ast
        notebook = json.loads(Path('notebooks/colab_t4.ipynb').read_text())
        self.assertEqual(notebook['nbformat'],4)
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                src = ''.join(cell['source'])
                src = '\n'.join(line for line in src.splitlines() if not line.startswith('%'))
                ast.parse(src)


if __name__=='__main__':
    unittest.main()
