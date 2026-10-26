import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .data import load_data, read_jsonl, digest
from .metrics import score, tokens, uncertainty, calibration, fit_temperature
from .paper import TABLE1, TABLE3, TABLE6, FIG3_AUC, TABLE8


def evaluate(rows, expected):
    exp = {(r['dataset'],str(r['id'])):r for r in expected}
    seen = set()
    for r in rows:
        key = (r['dataset'],str(r['id']))
        if key not in exp or key in seen:
            raise ValueError(f'Unexpected/duplicate prediction: {key}')
        for field in ('question','answer','answer_type','image','split'):
            if r[field] != exp[key][field]:
                raise ValueError(f'Prediction annotation differs: {key} {field}')
        if not np.isfinite(r['entropy']) or r['entropy'] < 0:
            raise ValueError('Invalid entropy')
        r['score'] = score(r['answer'],r['prediction'],r['answer_type'])
        seen.add(key)
    return {'expected':len(exp),'completed':len(rows),'complete':len(exp)==len(rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions',required=True)
    p.add_argument('--data',required=True,help='Full official test split; used for coverage checks')
    p.add_argument('--output',required=True)
    p.add_argument('--validation-predictions',help='Disjoint validation predictions; never test for temperature fitting')
    p.add_argument('--bins',type=int,default=15)
    p.add_argument('--error-tags',help='Optional JSONL: dataset,id,category from Table 8; manually audited')
    p.add_argument('--plots',action='store_true')
    args = p.parse_args()
    expected = load_data(args.data,'test')
    rows = read_jsonl(args.predictions)
    if not rows:
        raise ValueError('No predictions to evaluate')
    if len({(r['model'],r['setting']) for r in rows}) != 1:
        raise ValueError('Do not mix checkpoints/settings')
    model, setting = rows[0]['model'],rows[0]['setting']
    if model not in TABLE1 or setting not in ('zero_base','zero_med','downstream'):
        raise ValueError('Unknown model/setting')
    meta_path = Path(args.predictions+'.meta.json')
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else None
    coverage = evaluate(rows,expected)
    report = {'coverage':coverage,'model':model,'setting':setting,
              'prediction_sha256':digest(args.predictions),'run':metadata,
              'model_execution_provenance_present':metadata is not None,
              'scoring':'lowercase alphanumeric token multiset recall; closed first explicit yes/no',
              'uncertainty':'Eq4 mean full-vocabulary entropy in nats; error=open recall<1 or closed incorrect',
              'calibration_convention':'exp(-entropy); temperature fit on disjoint validation; not a verified paper mapping',
              'datasets':{}}
    validation = read_jsonl(args.validation_predictions) if args.validation_predictions else []
    if validation and metadata:
        validation_meta_path = Path(args.validation_predictions+'.meta.json')
        if not validation_meta_path.exists():
            raise ValueError('Calibration predictions need checkpoint metadata')
        vm = json.loads(validation_meta_path.read_text())
        if vm['runtime'] != metadata['runtime']:
            raise ValueError('Calibration checkpoint/runtime differs from test')
    test_questions = {(r['dataset'],r['image'],r['question'].strip().lower()) for r in expected}
    if any((r['dataset'],r['image'],r['question'].strip().lower()) in test_questions for r in validation):
        raise ValueError('Calibration validation duplicates a test question')
    test_ids = {(r['dataset'],str(r['id'])) for r in expected}
    if any((r['dataset'],str(r['id'])) in test_ids for r in validation):
        raise ValueError('Temperature validation overlaps test IDs')
    if any(r['split'] != 'validation' or (r['model'],r['setting']) != (model,setting) for r in validation):
        raise ValueError('Calibration requires same checkpoint/setting and validation split')
    tags = {(r['dataset'],str(r['id'])):r['category'] for r in read_jsonl(args.error_tags)} if args.error_tags else {}
    if any(category not in TABLE8 for category in tags.values()):
        raise ValueError('Unknown Table 8 error category')
    lines = ['# Baseline replication comparison', '',
             f'Model: {model}; setting: {setting}; completed {coverage["completed"]}/{coverage["expected"]}.',
             'Incomplete/smoke results are NOT full benchmark replications.', '',
             '|Dataset|Metric|Paper %|Run %|Delta (percentage points)|',
             '|---|---|---:|---:|---:|']
    for dataset in sorted({r['dataset'] for r in expected}):
        current = [r for r in rows if r['dataset'] == dataset]
        d = {'counts':{},'metrics':{},'comparison':{},'cases':current,
             'truncated':sum(r.get('truncated',False) for r in current)}
        ds_index = ('vqa_rad','slake','pathvqa').index(dataset)
        ref = TABLE1[model][ds_index] if setting == 'downstream' else None
        for i,kind in enumerate(('open','closed','overall')):
            subset = [r for r in current if kind == 'overall' or r['answer_type']==kind]
            value = 100*np.mean([r['score'] for r in subset]) if subset else None
            d['counts'][kind] = len(subset)
            d['metrics'][kind] = float(value) if value is not None else None
            paper_value = ref[i] if ref else None
            if setting != 'downstream' and kind == 'closed':
                key = model+('_base' if setting == 'zero_base' else '_pubmed')
                paper_value = TABLE3[key][ds_index]
            delta = value-paper_value if value is not None and paper_value is not None else None
            d['comparison'][kind] = {'paper':paper_value,'run':float(value) if value is not None else None,
                                     'delta_pp':float(delta) if delta is not None else None}
            fmt = lambda x: 'N/A' if x is None else f'{x:.2f}'
            lines.append(f'|{dataset}|{kind}|{fmt(paper_value)}|{fmt(value)}|{fmt(delta)}|')
        d['uncertainty'] = uncertainty([int(r['score']<1) for r in current],[r['entropy'] for r in current]) if current else None
        d['figure3_reference'] = FIG3_AUC[dataset]
        d['figure3_comparable_setting'] = setting == 'downstream' and model == 'phi35'
        by_answer = defaultdict(list)
        for r in current:
            by_answer[r['answer']].append(r['entropy'])
        d['entropy_by_answer'] = {a:{'count':len(v),'mean':float(np.mean(v)),'values':v} for a,v in by_answer.items()}
        # Table 6 only yes/no closed QA, separate from full-dataset Figure 3.
        closed = [r for r in current if r['answer_type']=='closed' and tokens(r['answer']) in (['yes'],['no'])]
        val = [r for r in validation if r['dataset']==dataset and r['answer_type']=='closed']
        d['calibration'] = {}
        for method,field in [('entropy',None),('label','label_confidence'),('true','true_confidence')]:
            available = closed if field is None else [r for r in closed if field in r]
            if not available:
                continue
            conf = lambda r: float(np.exp(-r['entropy'])) if field is None else r[field]
            temperature = 1.
            calibrated = False
            eligible_val = [r for r in val if field is None or field in r]
            if eligible_val:
                temperature = fit_temperature([int(score(r['answer'],r['prediction'],'closed')) for r in eligible_val],
                                               [conf(r) for r in eligible_val])
                calibrated = True
            result = calibration([int(r['score']) for r in available],[conf(r) for r in available],temperature,args.bins)
            if not calibrated:
                result['ece_t'] = result['bs_t'] = None
            refkey = f'phi35_{"finetune" if setting=="downstream" else "zero"}_{method}'
            reference = TABLE6.get(refkey) if model=='phi35' and setting!='zero_base' and ds_index < 2 else None
            result.update({'temperature':temperature if calibrated else None,'count':len(available),
                           'paper':dict(zip(('ece','ece_t','bs_t','auc'),reference[ds_index])) if reference else None,
                           'exact_protocol_verified':False})
            d['calibration'][method] = result
        errors = [r for r in current if r['score']<1 and r['entropy']<.35]
        d['low_entropy_errors'] = {'count':len(errors),'ids':[r['id'] for r in errors],
            'tagged':sum((dataset,str(r['id'])) in tags for r in errors),
            'categories':{cat:{'count':sum(tags.get((dataset,str(r['id'])))==cat for r in errors),
                'percent_of_all_errors':100*sum(tags.get((dataset,str(r['id'])))==cat for r in errors)/len(errors) if errors else None,
                'paper_percent':TABLE8[cat][ds_index] if ds_index<2 else None} for cat in TABLE8}}
        report['datasets'][dataset] = d
    lines += ['', 'Uncertainty / calibration (paper and run; values in [0,1]):', '',
              '|Dataset|Experiment|Metric|Paper|Run|', '|---|---|---|---:|---:|']
    for ds,d in report['datasets'].items():
        u = d['uncertainty']
        lines.append(f'|{ds}|Figure 3 (setting match={d["figure3_comparable_setting"]})|AUROC|{d["figure3_reference"]}|{u["auc"] if u else "N/A"}|')
        for method,result in d['calibration'].items():
            for metric in ('ece','ece_t','bs_t','auc'):
                refval = (result['paper'] or {}).get(metric,'N/A')
                lines.append(f'|{ds}|Table 6 / {method} (protocol provisional)|{metric}|{refval}|{result[metric]}|')
    lines += ['', 'Table 6 paper does not disclose all calibration conventions. These are provisional comparisons.',
              'Null/N/A means unrun, undefined, or unavailable; it never means zero.',
              'Full per-example outputs, ROC sensitivity/specificity, answer entropy, low-entropy errors: report.json.']
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for ds,d in report['datasets'].items():
            roc = (d['uncertainty'] or {}).get('roc',[])
            if roc:
                plt.figure(); plt.plot([1-r['specificity'] for r in roc],[r['sensitivity'] for r in roc])
                plt.xlabel('False positive rate'); plt.ylabel('Sensitivity (error detection)')
                plt.title(f'{ds}: AUROC={d["uncertainty"]["auc"]:.3f}; paper={d["figure3_reference"]}')
                plt.savefig(out/f'{ds}_roc.png',bbox_inches='tight'); plt.close()
            plt.figure(); plt.hist([r['entropy'] for r in d['cases']],bins=30)
            plt.xlabel('Predictive entropy (nats)'); plt.ylabel('Questions')
            plt.savefig(out/f'{ds}_entropy.png',bbox_inches='tight'); plt.close()
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
