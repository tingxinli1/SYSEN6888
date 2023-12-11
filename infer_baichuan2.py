# coding: utf-8
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'False'
import warnings
import gc
import time
from typing import Dict
import jsonlines
from tqdm import tqdm

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

# specify base model
base_model = 'Baichuan2-7B-Base'
base_model_dir = f'./models/{base_model}'

# specify lora weights
lora_path = f"./outputs/checkpoint-9000"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print('loading base model...')
model = AutoModelForCausalLM.from_pretrained(base_model_dir, load_in_4bit=True, quantization_config=bnb_config, device_map='cuda:0', trust_remote_code=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
print('loading lora adapter...')
model = PeftModel.from_pretrained(model, lora_path, quantization_config=bnb_config, device_map='cuda:0', trust_remote_code=True)


output_file = './outputs/test_output_baichuan.txt'
with open(output_file, 'a', encoding='utf-8') as f:
    f.truncate(0)
test_data = []
with jsonlines.open('./data/AdvertiseGen/dev.json') as reader:
    for line in reader:
        test_data.append(line)
for line in tqdm(test_data):
    product_description = line['content']
    # print('input:', product_description)
    initial_prompt = '商品描述：%s->广告：'
    input_text = initial_prompt % product_description
    inputs = tokenizer(input_text, return_tensors='pt')
    inputs = inputs.to('cuda:0')
    pred = model.generate(
        **inputs, 
        generation_config = GenerationConfig(
            max_new_tokens = 128,
            # do_sample=True,
            top_k=50,
            num_beams=6,
            num_beam_groups=3,
            diversity_penalty=1.3,
            penalty_alpha=0.6,
            # bad_words_ids=[[16136, 11766]],
            repetition_penalty=3.0,
            ),
        )
    pred = pred.cpu()[0][len(inputs['input_ids'][0]):]
    response = tokenizer.decode(pred, skip_special_tokens=True).split('\n')[0]
    torch.cuda.empty_cache()
    gc.collect()
    # print('output:', response)
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(response)
        f.write('\n')