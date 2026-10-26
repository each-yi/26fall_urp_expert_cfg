import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from PIL import Image
from baseline.download_data import export, cached_paths, normalize, unpack_assets, asset_resolver
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

    def test_invalid_annotations_are_quarantined_with_original_ids(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d)/'slake'
            train = [sample(), sample(1622, answer=''), sample(1623, answer=None),
                     sample(1624, question='  '), sample(1625, question=None),
                     sample(1626, q_lang='zh'), sample(1627, answer=0, answer_type='OPEN')]
            test = [sample(), sample(1700, answer='   ')]
            paths = export('slake', {'train':train,'test':test},target,'fixture/slake','sha')
            kept = load_data(paths['train'], 'train')
            self.assertEqual([r['source_id'] for r in kept], ['0','1627'])
            self.assertEqual(kept[1]['answer'], '0')
            manifest = json.loads((target/'provenance.json').read_text())
            self.assertEqual(manifest['splits']['train']['raw'],7)
            self.assertEqual(manifest['splits']['train']['filtered_language'],1)
            self.assertEqual(manifest['splits']['train']['invalid_annotations'],4)
            self.assertEqual(manifest['splits']['test']['invalid_annotations'],1)
            rejected = [json.loads(line) for line in (target/'train.rejected.jsonl').read_text().splitlines()]
            self.assertEqual(rejected[0]['source_id'],'1622')
            self.assertEqual(rejected[0]['answer'],'')
            self.assertEqual(rejected[0]['reason'],'empty_answer')
            self.assertEqual(rejected[-1]['reason'],'empty_or_nontext_question')
            self.assertEqual(cached_paths(target,'fixture/slake'),paths)

    def test_unsupported_schema_still_fails_instead_of_discarding_rows(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d)/'slake'
            bad = sample()
            del bad['answer']
            with self.assertRaisesRegex(ValueError, 'columns missing'):
                export('slake',{'train':[sample(),bad],'test':[sample()]},target,'fixture/slake','sha')
            self.assertFalse(target.exists())

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

    def test_slake_nested_paths_disambiguate_repeated_basenames(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'snapshot'
            for folder in ('xmlab1', 'xmlab2', 'xmlab11'):
                path = root/'Slake1.0'/'imgs'/folder/'source.jpg'
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (4,4)).save(path)
            hub = SimpleNamespace(snapshot_download=lambda *args, **kwargs: str(root))
            with patch.dict('sys.modules', {'huggingface_hub':hub}):
                resolve = asset_resolver('fixture/slake', 'revision', str(Path(d)/'cache'))
                for folder in ('xmlab1', 'xmlab2', 'xmlab11'):
                    self.assertEqual(Path(resolve(f'{folder}/source.jpg')),
                                     root/'Slake1.0'/'imgs'/folder/'source.jpg')
                with self.assertRaisesRegex(ValueError, '0 full-path matches'):
                    resolve('xmlab999/source.jpg')
                with self.assertRaisesRegex(ValueError, '3 full-path matches'):
                    resolve('source.jpg')

    def test_resolver_keeps_hf_symlink_logical_path(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'snapshot'
            blob = Path(d)/'blob-without-extension'
            blob.write_bytes(b'image bytes')
            for folder in ('xmlab1', 'xmlab2'):
                path = root/'imgs'/folder/'source.jpg'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.symlink_to(blob if folder=='xmlab1' else Path(d)/'missing')
            hub = SimpleNamespace(snapshot_download=lambda *args, **kwargs: str(root))
            with patch.dict('sys.modules', {'huggingface_hub':hub}):
                resolve = asset_resolver('fixture/slake', 'revision', str(Path(d)/'cache'))
                self.assertEqual(Path(resolve('xmlab1/source.jpg')), blob)
                # A missing folder cannot fall back to the only matching basename.
                with self.assertRaisesRegex(ValueError, '0 full-path matches'):
                    resolve('xmlab999/source.jpg')

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
