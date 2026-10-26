"""One-image batches, resumable JSONL, online full-vocabulary entropy (Eq. 4)."""
import argparse
import hashlib
import json
from pathlib import Path
from .data import load_data, read_jsonl, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--image-root',required=True)
    p.add_argument('--split',choices=('test','validation'),default='test')
    p.add_argument('--self-check',action='store_true',help='Additional image-conditioned Is True? confidence')
    p.add_argument('--output',required=True)
    p.add_argument('--model',choices=('phi3','phi35'),default='phi35')
    p.add_argument('--adapter',help='HF adapter ID or local downstream checkpoint; base means no adapter')
    p.add_argument('--setting',choices=('zero_base','zero_med','downstream'),required=True)
    p.add_argument('--precision',choices=('4bit','fp16'),default='4bit')
    p.add_argument('--crops',type=int,default=16)
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--limit',type=int)
    p.add_argument('--revision',default='main')
    p.add_argument('--adapter-revision',default='main')
    p.add_argument('--label-samples',type=int,choices=(0,10),default=0)
    args = p.parse_args()
    if not 1 <= args.crops <= 16 or args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        p.error('crops must be 1..16; max-new-tokens and limit must be positive')
    rows = load_data(args.data, args.split)
    if args.limit:
        rows = rows[:args.limit]
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList, set_seed
    from .model import load, inputs, ADAPTERS
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    adapter = args.adapter or (None if args.setting == 'zero_base' else ADAPTERS[args.model])
    if adapter == 'base':
        adapter = None
    if (args.setting == 'zero_base') != (adapter is None):
        raise ValueError('zero_base requires no adapter; medical settings require adapter')
    if args.setting == 'downstream' and (not args.adapter or args.adapter in ADAPTERS.values()):
        raise ValueError('Use a dataset-specific downstream checkpoint, not the public medical adapter')
    model, processor, meta = load(args.model,adapter,args.precision,args.crops,
                                  args.revision,args.adapter_revision)
    if args.setting == 'downstream':
        provenance_path = Path(adapter)/'replication_training.json'
        if not provenance_path.exists() or not (Path(adapter)/'TRAINING_COMPLETE').exists():
            raise ValueError('Downstream checkpoint requires replication_training.json from baseline.train')
        provenance = json.loads(provenance_path.read_text())
        if set(r['dataset'] for r in rows) != {provenance['dataset']}:
            raise ValueError('Wrong dataset-specific downstream adapter')
        meta['training'] = provenance
    model.eval()
    out = Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    # Hash all adapter files, including local weights, to prevent accidental resume mixing.
    if adapter and Path(adapter).is_dir():
        meta['local_adapter_hashes'] = {str(f.relative_to(adapter)):digest(f)
             for f in sorted(Path(adapter).rglob('*')) if f.is_file()}
    config = {**vars(args),'data_sha256':digest(args.data), 'runtime':meta}
    meta_path = out.with_suffix(out.suffix+'.meta.json')
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old != config:
            raise ValueError('Run configuration changed; use a new output filename')
    elif out.exists():
        raise ValueError('Output without metadata; use a new output filename')
    else:
        meta_path.write_text(json.dumps(config,indent=2))
    done = {(r['dataset'],str(r['id'])) for r in read_jsonl(out)} if out.exists() else set()

    class EntropyRecorder(LogitsProcessor):
        def __init__(self):
            self.values = []
        def __call__(self,input_ids,scores):
            logp = scores.float().log_softmax(-1)
            prob = logp.exp()
            # Zero-probability entries can have -inf logits.
            h = -torch.where(prob > 0,prob*logp,0).sum(-1).item()
            self.values.append((h,float(logp.max(-1).values.item())))
            return scores

    with out.open('a',encoding='utf-8') as f, torch.inference_mode():
        for row in rows:
            key = (row['dataset'],str(row['id']))
            if key in done:
                continue
            # Per-question RNG: resumed sampling is identical to uninterrupted sampling.
            row_seed = args.seed + int(hashlib.sha256(repr(key).encode()).hexdigest()[:8],16)
            set_seed(row_seed % (2**32))
            batch = inputs(processor,row,args.image_root).to('cuda:0')
            if 'pixel_values' in batch:
                batch['pixel_values'] = batch['pixel_values'].to(torch.float16)
            recorder = EntropyRecorder()
            eos = model.generation_config.eos_token_id or processor.tokenizer.eos_token_id
            eos_ids = eos if isinstance(eos,list) else [eos]
            common = dict(max_new_tokens=args.max_new_tokens,use_cache=True,
                          eos_token_id=eos, num_beams=1,repetition_penalty=1.0,
                          no_repeat_ngram_size=0,min_length=0,min_new_tokens=0,
                          pad_token_id=processor.tokenizer.eos_token_id)
            sequence = model.generate(**batch,**common,do_sample=False,
                           logits_processor=LogitsProcessorList([recorder]))
            ids = sequence[0,batch['input_ids'].shape[1]:].tolist()
            # Exclude special termination tokens from both sequence statistics.
            vals = [v for i,v in zip(ids,recorder.values) if i not in processor.tokenizer.all_special_ids]
            if not vals:
                raise ValueError(f'Empty generated answer for {key}')
            prediction = processor.tokenizer.decode(ids,skip_special_tokens=True).strip()
            result = {**row, 'prediction':prediction,
                      'entropy':sum(v[0] for v in vals)/len(vals),
                      'chosen_token_nll':-sum(v[1] for v in vals)/len(vals),
                      'tokens':len(vals),'truncated':len(ids)==args.max_new_tokens and ids[-1] not in eos_ids,
                      'model':args.model,'setting':args.setting}
            if args.label_samples and row['answer_type'] == 'closed':
                from .metrics import tokens
                samples = []
                for _ in range(10):
                    s = model.generate(**batch,**common,do_sample=True,temperature=1.,
                                       top_p=1.,top_k=0)
                    samples.append(processor.tokenizer.decode(
                        s[0,batch['input_ids'].shape[1]:],skip_special_tokens=True))
                def label(s):
                    return next((t for t in tokens(s) if t in ('yes','no')),None)
                result['label_samples'] = samples
                result['label_confidence'] = sum(label(s)==label(prediction) and label(prediction) is not None
                                                  for s in samples)/10
            if args.self_check and row['answer_type'] == 'closed':
                check = {**row,'question':row['question']+'\nAnswer: '+prediction+
                         '\nIs the answer true? Respond True or False.'}
                check_batch = inputs(processor,check,args.image_root).to('cuda:0')
                check_batch['pixel_values'] = check_batch['pixel_values'].to(torch.float16)
                logits = model(**check_batch).logits[0,-1].float()
                def single_tokens(word):
                    candidates = {tuple(processor.tokenizer.encode(prefix+word,add_special_tokens=False))
                                  for prefix in ('',' ')}
                    return sorted({ids[0] for ids in candidates if len(ids)==1})
                true_ids, false_ids = single_tokens('True'),single_tokens('False')
                if not true_ids or not false_ids or set(true_ids)&set(false_ids):
                    raise ValueError('True/False must have distinct single-token encodings')
                log_true = logits[true_ids].logsumexp(0)
                log_false = logits[false_ids].logsumexp(0)
                result['true_confidence'] = (log_true-log_false).sigmoid().item()
            f.write(json.dumps(result,ensure_ascii=False)+'\n')
            f.flush()
            print(f'{len(done)+1}/{len(rows)} {key}: {prediction[:80]}',flush=True)
            done.add(key)
    print('Predictions:',out)


if __name__ == '__main__':
    main()
