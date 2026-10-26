import importlib.metadata
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

MODELS = {'phi3':'microsoft/Phi-3-vision-128k-instruct',
          'phi35':'microsoft/Phi-3.5-vision-instruct'}
ADAPTERS = {'phi3':'ecoxial2007/Phi-3V-Med', 'phi35':'ecoxial2007/Phi-3.5V-Med'}


def load(model_key, adapter, precision='4bit', crops=4, revision='main',
         adapter_revision='main', train=False):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required. In Colab select Runtime > Change runtime type > T4 GPU.')
    from huggingface_hub import model_info
    # Resolve mutable refs before loading any remote code or weights; log exact commits.
    model_sha = model_info(MODELS[model_key], revision=revision).sha
    adapter_sha = None
    if adapter and not __import__('os').path.isdir(adapter):
        adapter_sha = model_info(adapter, revision=adapter_revision).sha
    quant = None
    if precision == '4bit':
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                  bnb_4bit_compute_dtype=torch.float16,
                                  bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[model_key], revision=model_sha, trust_remote_code=True,
        torch_dtype=torch.float16, quantization_config=quant,
        device_map={'':'cuda:0'}, _attn_implementation='eager')
    processor = AutoProcessor.from_pretrained(MODELS[model_key], revision=model_sha,
                       trust_remote_code=True, num_crops=crops)
    if train:
        if quant:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        else:
            model.requires_grad_(False)
            model.enable_input_require_grads()
        model.config.use_cache = False
    if adapter:
        model = PeftModel.from_pretrained(model, adapter,
                    revision=adapter_sha, is_trainable=train)
    elif train:
        raise ValueError('Downstream training requires the medical adapter as initialization')
    meta = {'model':MODELS[model_key], 'model_revision':model_sha,
            'adapter':adapter, 'adapter_revision':adapter_sha,
            'precision':precision,'num_crops':crops,'attention':'eager',
            'gpu':torch.cuda.get_device_name(),
            'versions':{p:importlib.metadata.version(p) for p in
                        ('torch','transformers','peft','bitsandbytes','accelerate')}}
    return model, processor, meta


def inputs(processor, row, image_root, answer=False):
    from PIL import Image
    from pathlib import Path
    with Image.open(Path(image_root) / row['image']) as img:
        image = img.convert('RGB')
    prompt = processor.tokenizer.apply_chat_template([
        {'role':'user','content':'<|image_1|>\n'+row['question']}],
        tokenize=False, add_generation_prompt=True)
    batch = processor(prompt, [image], return_tensors='pt')
    if answer:
        ids = processor.tokenizer(str(row['answer'])+'<|end|>\n<|endoftext|>',
                                  add_special_tokens=False, return_tensors='pt')['input_ids']
        labels = torch.cat([torch.full_like(batch['input_ids'],-100), ids], dim=1)
        batch['input_ids'] = torch.cat([batch['input_ids'], ids], dim=1)
        batch['attention_mask'] = torch.ones_like(batch['input_ids'])
        batch['labels'] = labels
    return batch
