import os
import datasets
from datasets import load_dataset
import torch
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
# dataset.cleanup_cache_files()

save_dir = './outputs'
raw_datasets = load_dataset('./data/AdvertiseGen')
model_path = "./models/Baichuan2-7B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

class CustomizedTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None, save_dir=save_dir):
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        output_dir = os.path.join(save_dir, checkpoint_folder)
        os.mkdir(output_dir)
        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        model.save_pretrained(output_dir)

def preprocess_inputs(examples, max_len=200, ignore_pad_token_for_loss=True):
    model_inputs = {'input_ids': [], 'labels': [], }
    # teamplate = tokenizer.bos_token + 'Keywords: %s\nAdvertisement: %s' + tokenizer.eos_token
    for i in range(len(examples['content'])):
        prompt = f"商品描述：{examples['content'][i]}->广告："
        a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
        b_ids = tokenizer.encode(text=examples['summary'][i], add_special_tokens=False)
        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
        if len(input_ids) > max_len or '<UNK>' in examples['summary'][i]:
            input_ids = []
            labels = []
        else:
            pad_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
            pad_length = max_len - len(input_ids)
            labels = [pad_id] * context_length + b_ids + [tokenizer.eos_token_id] + [pad_id] * pad_length
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
            # input_ids = input_ids[:-1]
            # labels = labels[1:]
        model_inputs['input_ids'].append(input_ids)
        model_inputs['labels'].append(labels)
    return model_inputs

def print_dataset_example(example):
    print("input_ids:", example["input_ids"])
    print("inputs:", tokenizer.decode(example["input_ids"], skip_special_tokens=False))
    print("label_ids:", example["labels"])
    print("labels:", tokenizer.decode([i if i != -100 else tokenizer.pad_token_id for i in example["labels"]], skip_special_tokens=False))

train_dataset = raw_datasets['train'].map(preprocess_inputs, batched=True, num_proc=1, load_from_cache_file=False)
original_size = len(train_dataset)
train_dataset = train_dataset.filter(lambda example: len(example['input_ids']) != 0, load_from_cache_file=False)
filtered_size = len(train_dataset)
print('filtered:', 1 - filtered_size / original_size)

train_dataset = train_dataset.train_test_split(test_size=1000)
train_dataset, eval_dataset = train_dataset['train'], train_dataset['test']

for i in range(5):
    print_dataset_example(train_dataset[i])

# exit()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
)
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cpu', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True, quantization_config=bnb_config, torch_dtype=torch.float16, device_map='cuda:0', trust_remote_code=True)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
lora_targets = find_all_linear_names(model)
lora_config = LoraConfig(
                r=16,
                lora_alpha=4,
                # target_modules=lora_targets,
                target_modules = ['W_pack'],
                lora_dropout=0.1,
                bias='none',
                task_type='CAUSAL_LM',
            )

model = get_peft_model(model, lora_config)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=None, padding=False)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    trainable_params /= 2
    print(f"""trainable params: {trainable_params} || 
                f'all params: {all_param} || 
                f'trainable: {100 * trainable_params / all_param}""")
    
print_trainable_parameters(model)

args = TrainingArguments(
    output_dir=save_dir,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=1,
    gradient_accumulation_steps=1,
    num_train_epochs=2,
    weight_decay=0.1,
    warmup_ratio=0.15,
    optim='paged_adamw_32bit',
    lr_scheduler_type="linear",
    learning_rate=5e-4,
    save_strategy='steps',
    save_steps=1000,
    eval_steps=1000,
    # use_cpu=True,
    bf16=True,
    run_name='baichuan-v3',
    report_to='wandb'
)

trainer = CustomizedTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()