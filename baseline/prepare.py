"""Convert official JSON annotations without constructing a new split."""
import argparse
import json
from pathlib import Path
from .data import load_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input',required=True,help='JSON array (export PathVQA pickle to JSON locally first)')
    p.add_argument('--dataset',choices=('vqa_rad','slake','pathvqa'),required=True)
    p.add_argument('--split',choices=('train','validation','test'),required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--image-root',required=True)
    p.add_argument('--image-template',help='Optional e.g. {split}/{img_id}.jpg; fields from source row')
    args = p.parse_args()
    raw = json.loads(Path(args.input).read_text())
    if not isinstance(raw,list):
        raise ValueError('Input must be a JSON array')
    rows = []
    for i,r in enumerate(raw):
        if args.dataset == 'slake' and r.get('q_lang',r.get('language','en')) != 'en':
            continue
        image = r.get('image',r.get('img_name',r.get('image_name',r.get('img_id'))))
        if args.image_template:
            image = args.image_template.format(**{**r,'split':args.split})
        if image is None or not (Path(args.image_root)/str(image)).is_file():
            raise ValueError(f'Missing image in source row {i}: {image}; supply --image-template')
        answer = str(r['answer'])
        kind = str(r.get('answer_type','closed' if answer.lower() in ('yes','no') else 'open')).lower()
        if kind not in ('open','closed'):
            raise ValueError(f'Unknown answer_type: {kind}')
        row = {'id':str(r.get('qid',r.get('question_id',r.get('id',f'{args.split}:{i}')))),
               'dataset':args.dataset,'split':args.split,'image':str(image),
               'question':r['question'],'answer':answer,'answer_type':kind}
        rows.append(row)
    out = Path(args.output)
    if out.exists():
        raise ValueError('Refusing to overwrite existing prepared annotations')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    load_data(out,args.split)
    print(f'Wrote {len(rows)} questions to {out}; preserve source split and provenance.')


if __name__ == '__main__':
    main()
