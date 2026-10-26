"""Score externally supplied retrieval / expert-extraction artifacts; does not fabricate them."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import read_jsonl
from .metrics import tokens
from .paper import TABLE2,TABLE5


def contains(answer, caption):
    a,b = tokens(answer),tokens(caption)
    return bool(a) and any(b[i:i+len(a)]==a for i in range(len(b)-len(a)+1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--retrieval',help='JSONL dataset,answer,answer_type,retrieved:{image/text/sum/union:[{caption,score}]}')
    p.add_argument('--keywords',help='JSONL gold:[phrases],predicted:[phrases] (independent expert gold)')
    p.add_argument('--output',required=True)
    a = p.parse_args()
    result = {}
    if a.retrieval:
        rows = read_jsonl(a.retrieval)
        result['retrieval'] = {}
        for i,ds in enumerate(('vqa_rad','slake','pathvqa')):
            subset = [r for r in rows if r['dataset']==ds and r['answer_type']=='open']
            result['retrieval'][ds] = {}
            for feature in TABLE2:
                valid = [r for r in subset if feature in r['retrieved']]
                if not valid:
                    continue
                for r in valid:
                    scores = [x['score'] for x in r['retrieved'][feature]]
                    if scores != sorted(scores,reverse=True):
                        raise ValueError('Rankings must be sorted descending by similarity')
                hit = {}
                for k in (1,2,3,4,6,8,10):
                    # No silently padding short candidate lists.
                    eligible = [r for r in valid if len(r['retrieved'][feature]) >= k]
                    hit[str(k)] = {'count':len(eligible), 'percent':100*float(np.mean([
                        any(contains(r['answer'],x['caption']) for x in r['retrieved'][feature][:k])
                        for r in eligible])) if eligible else None}
                result['retrieval'][ds][feature] = {'top_k':hit, 'paper_top1_percent':TABLE2[feature][i],
                    'paper_other_k':None, 'score_distributions':{
                      label:[x['score'] for r in valid for x in r['retrieved'][feature]
                             if contains(r['answer'],x['caption'])==is_hit]
                      for label,is_hit in [('hit',True),('miss',False)]},
                    'by_answer':{answer:{str(k):float(np.mean([
                        any(contains(r['answer'],x['caption']) for x in r['retrieved'][feature][:k])
                        for r in valid if r['answer']==answer and len(r['retrieved'][feature])>=k]))
                        for k in (1,2,3) if any(r['answer']==answer and len(r['retrieved'][feature])>=k for r in valid)}
                        for answer in set(r['answer'] for r in valid)}}
    if a.keywords:
        tp=fp=fn=0
        for r in read_jsonl(a.keywords):
            gold = {' '.join(tokens(x)) for x in r['gold']}
            pred = {' '.join(tokens(x)) for x in r['predicted']}
            tp += len(gold&pred); fp += len(pred-gold); fn += len(gold-pred)
        precision = tp/(tp+fp) if tp+fp else 0
        recall = tp/(tp+fn) if tp+fn else 0
        result['keywords'] = {'paper':TABLE5,'run':{'precision':100*precision,'recall':100*recall,
                                'f1':200*precision*recall/(precision+recall) if precision+recall else 0},
                              'convention':'micro exact normalized phrase matching; paper gold unavailable'}
    if not result:
        raise ValueError('Supply at least one artifact file')
    out = Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
