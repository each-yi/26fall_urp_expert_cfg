"""Dataset-specific QLoRA adaptation of the released medical adapter for T4."""
import argparse
import json
from pathlib import Path
from .data import load_data, audit_splits, digest


def main():
    p = argparse.ArgumentParser()
    for name in ('train','test','image-root','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--model',choices=('phi3','phi35'),default='phi35')
    p.add_argument('--adapter')
    p.add_argument('--crops',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--accumulation',type=int,default=64)
    args = p.parse_args()
    train, test = load_data(args.train,'train'),load_data(args.test,'test')
    datasets = set(r['dataset'] for r in train+test)
    if len(datasets) != 1:
        raise ValueError('Train one dataset-specific adapter at a time')
    audit = audit_splits(train,test)
    import torch
    from transformers import Trainer, TrainingArguments, set_seed
    from .model import load, inputs, ADAPTERS
    set_seed(args.seed)
    model, processor, meta = load(args.model,args.adapter or ADAPTERS[args.model],
                                 crops=args.crops,train=True)
    active = next(iter(model.peft_config.values()))
    if active.r != 64:
        raise ValueError(f'Paper uses rank 64; adapter has rank {active.r}')
    model.print_trainable_parameters()

    class LazyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(train)
        def __getitem__(self,i):
            return train[i]

    def collate(rows):
        if len(rows) != 1:
            raise ValueError('Phi vision supports microbatch 1')
        batch = inputs(processor,rows[0],args.image_root,answer=True)
        if 'pixel_values' in batch:
            batch['pixel_values'] = batch['pixel_values'].to(torch.float16)
        return dict(batch)

    output = Path(args.output)
    provenance = {'dataset':next(iter(datasets)),'model':args.model,
                  'train_sha256':digest(args.train),'test_sha256':digest(args.test),
                  'split_audit':audit,'runtime':meta,'arguments':vars(args),
                  'lr':1e-4,'optimizer':'adamw_torch','scheduler':'cosine',
                  'effective_batch':args.accumulation,'rank':64,
                  'deviations':['NF4 base weights','FP16 compute instead of BF16',
                                'paper does not specify crop count/LoRA targets/warmup/weight decay']}
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Output nonempty; use --resume or a new directory')
    output.mkdir(parents=True,exist_ok=True)
    prov = output/'replication_training.json'
    if args.resume and prov.exists():
        old = json.loads(prov.read_text())
        # Resume flag is the only allowed argument difference.
        old['arguments']['resume'] = args.resume
        if old != provenance:
            raise ValueError('Training configuration changed during resume')
    prov.write_text(json.dumps(provenance,indent=2))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    training = TrainingArguments(output_dir=str(output),num_train_epochs=args.epochs,
          per_device_train_batch_size=1,gradient_accumulation_steps=args.accumulation,
          learning_rate=1e-4,lr_scheduler_type='cosine',optim='adamw_torch',
          fp16=True,bf16=False,gradient_checkpointing=True,
          gradient_checkpointing_kwargs={'use_reentrant':False},
          remove_unused_columns=False,dataloader_num_workers=0,
          logging_steps=10,save_steps=100,save_total_limit=2,
          report_to='none',seed=args.seed,data_seed=args.seed,weight_decay=0.,warmup_ratio=0.)
    trainer = Trainer(model=model,args=training,train_dataset=LazyDataset(),data_collator=collate)
    (output/'TRAINING_COMPLETE').unlink(missing_ok=True)
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model(str(output))
    processor.save_pretrained(str(output))
    (output/'TRAINING_COMPLETE').write_text('Completed requested epochs; see trainer_state.json\n')
    trainer.state.save_to_json(str(output/'trainer_state.json'))


if __name__ == '__main__':
    main()
