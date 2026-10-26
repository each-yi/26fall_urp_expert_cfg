import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from PIL import Image
from baseline.download_data import export, cached_paths, normalize, unpack_assets
from baseline.data import load_data


def sample(id=0, **changes):
    return {'qid':id,'question':'Is there a lesion?','answer':'yes','answer_type':'CLOSED',
            'image':Image.new('RGB',(4,4),color='red'), 'q_lang':'en', **changes}


class DownloadTest(unittest.TestCase):
    def test_export_and_resume_preserve_source_splits(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d)/'slake'
            splits = {'train':[sample(),sample(1,q_lang='zh')],
                      'test':[sample(question='Which organ?',answer='lung',answer_type='OPEN')],
                      'validation':[sample(question='Normal?')]}
            paths = export('slake',splits,target,'fixture/source','abc123')
            self.assertEqual(set(paths),{'train','test','validation','images'})
            train,test = load_data(paths['train'],'train'),load_data(paths['test'],'test')
            self.assertEqual(len(train),1)
            self.assertEqual(train[0]['id'],'train:0')
            self.assertEqual(test[0]['id'],'test:0')
            self.assertEqual(test[0]['answer_type'],'open')
            self.assertEqual(train[0]['image'],test[0]['image'])
            self.assertEqual(len(list((target/'images').glob('*.png'))),1)
            manifest = json.loads((target/'provenance.json').read_text())
            self.assertEqual(manifest['revision'],'abc123')
            self.assertEqual(manifest['splits']['train']['raw'],2)
            self.assertEqual(cached_paths(target,'fixture/source'),paths)
            with self.assertRaises(ValueError):
                cached_paths(target,'fixture/other')
            Path(paths['test']).write_text('modified')
            with self.assertRaises(ValueError):
                cached_paths(target,'fixture/source')

    def test_image_bytes_and_inferred_types(self):
        with tempfile.TemporaryDirectory() as d:
            buffer = io.BytesIO()
            Image.new('RGB',(4,4)).save(buffer,format='PNG')
            row = {'question':'Where?','answer':'left lung','image':{'bytes':buffer.getvalue(),'path':None}}
            paths = export('vqa_rad',{'train':[row],'test':[row]},Path(d)/'rad','fixture/rad','sha')
            self.assertEqual(load_data(paths['test'])[0]['answer_type'],'open')
            image = next((Path(paths['images'])).glob('*.png'))
            image.write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                cached_paths(Path(d)/'rad','fixture/rad')

    def test_no_random_resplit_and_failed_export_is_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d)/'rad'
            with self.assertRaises(ValueError):
                export('vqa_rad',{'train':[sample()]},target,'fixture/rad','sha')
            self.assertFalse(target.exists())
            with self.assertRaises(ValueError):
                export('vqa_rad',{'train':[sample(image=None)],'test':[sample()]},target,'fixture/rad','sha')
            self.assertFalse(target.exists())

    def test_missing_fields_and_numeric_category_rejected(self):
        with self.assertRaises(ValueError):
            normalize({'image':None},'vqa_rad','test',0)
        with self.assertRaises(ValueError):
            normalize(sample(answer_type=1),'vqa_rad','test',0)

    def test_archive_assets_and_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            archive = Path(d)/'assets.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('SLAKE/imgs/one.png',b'image-bytes')
                z.writestr('SLAKE/train.json','[]')
                z.writestr('SLAKE/executable.py','raise Exception()')
            unpack_assets([archive],Path(d)/'out')
            self.assertTrue((Path(d)/'out/SLAKE/train.json').exists())
            self.assertFalse((Path(d)/'out/SLAKE/executable.py').exists())
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('../escape.json','{}')
            with self.assertRaises(ValueError):
                unpack_assets([archive],Path(d)/'out')


if __name__=='__main__':
    unittest.main()
