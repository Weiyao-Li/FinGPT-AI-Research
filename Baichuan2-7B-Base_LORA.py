#!/usr/bin/env python
# coding: utf-8
pip install datasets transformers torch tqdm pandas huggingface_hub
pip install protobuf transformers==4.30.2 cpm_kernels torch>=2.0 gradio mdtex2html sentencepiece accelerate
pip install torch torchvision torchaudio
pip install transformers
pip install loguru
pip install datasets
pip install peft
pip install bitsandbytes
pip install tensorboard
pip install sentencepiece
pip install accelerate -U
pip install transformers==4.30.2 peft==0.4.0
pip install sentencepiece
pip install accelerate
pip install torch
pip install peft
pip install datasets
pip install bitsandbytes
pip install --upgrade peft

# In[1]:


import os
import shutil

jsonl_path = "../data/dataset_new.jsonl"
save_path = '../data/dataset_new'


if os.path.exists(jsonl_path):
    os.remove(jsonl_path)

if os.path.exists(save_path):
    shutil.rmtree(save_path)

import os

directory = "../data"
if not os.path.exists(directory):
    os.makedirs(directory)


# In[2]:


get_ipython().system('pip install datasets transformers torch tqdm pandas huggingface_hub')
get_ipython().system('pip install protobuf transformers==4.30.2 cpm_kernels torch>=2.0 gradio mdtex2html sentencepiece accelerate')


# In[3]:


from datasets import load_dataset
import datasets


# 1. TFNS

# In[4]:


dic = {
    0:"negative",
    1:'positive',
    2:'neutral',
}


# In[5]:


social_media_dataset = load_dataset('zeroshot/twitter-financial-news-sentiment')
social_media_dataset = social_media_dataset['train']
social_media_dataset = social_media_dataset.to_pandas()
social_media_dataset['label'] = social_media_dataset['label'].apply(lambda x:dic[x])
social_media_dataset['instruction'] = 'What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.'
social_media_dataset.columns = ['input', 'output', 'instruction']
social_media_dataset = datasets.Dataset.from_pandas(social_media_dataset)
social_media_dataset


# In[6]:


tmp_dataset = datasets.concatenate_datasets([social_media_dataset]*2)
train_dataset = tmp_dataset
print(tmp_dataset.num_rows)


# In[7]:


all_dataset = train_dataset.shuffle(seed = 42)
all_dataset.shape


# Make Dataset

# In[8]:


import json
from tqdm.notebook import tqdm


# In[9]:


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


# In[10]:


data_list = []
for item in all_dataset.to_pandas().itertuples():
    tmp = {}
    tmp["instruction"] = item.instruction
    tmp["input"] = item.input
    tmp["output"] = item.output
    data_list.append(tmp)


# In[11]:


with open("../data/dataset_new.jsonl", 'w') as f:
    for example in tqdm(data_list, desc="formatting.."):
        f.write(json.dumps(format_example(example)) + '\n')


# Tokenize

# In[11]:





# In[12]:


import json
from tqdm.notebook import tqdm

import datasets
from transformers import AutoTokenizer, AutoConfig

model_name = "baichuan-inc/Baichuan2-7B-Base"
jsonl_path = "../data/dataset_new.jsonl"  # updated path
save_path = '../data/dataset_new'  # updated path
max_seq_length = 512
skip_overlength = True


# In[13]:


def preprocess(tokenizer, config, example, max_seq_length):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}

def read_jsonl(path, max_seq_length, skip_overlength=False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, config, example, max_seq_length)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature


# 

# In[14]:


dataset = datasets.Dataset.from_generator(
    lambda: read_jsonl(jsonl_path, max_seq_length, skip_overlength)
    )
dataset.save_to_disk(save_path)


# In[15]:


from datasets import load_from_disk

loaded_dataset = load_from_disk('../data/dataset_new')

num_samples = loaded_dataset.num_rows

print(f'Number of samples in the dataset: {num_samples}')


# In[15]:





# The start of LORA:

# In[16]:


get_ipython().system('pip install torch torchvision torchaudio')
get_ipython().system('pip install transformers')
get_ipython().system('pip install loguru')
get_ipython().system('pip install datasets')
get_ipython().system('pip install peft')
get_ipython().system('pip install bitsandbytes')
get_ipython().system('pip install tensorboard')
get_ipython().system('pip install sentencepiece')


# In[17]:


# only for WSL
import os
os.environ["PATH"] = f"{os.environ['PATH']}:/usr/local/cuda/bin"
os.environ['LD_LIBRARY_PATH'] = "/usr/lib/wsl/lib:/usr/local/cuda/lib64"


# In[18]:


from typing import List, Dict, Optional

import datasets
import torch
from loguru import logger
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
    prepare_model_for_int8_training,
)
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING


# In[19]:


get_ipython().system('pip install accelerate -U')


# In[20]:


training_args = TrainingArguments(
        output_dir='./finetuned_model',    # saved model path
        logging_steps = 500,
        # max_steps=10000,
        num_train_epochs = 2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=1000,
        save_steps=500,
        fp16=True,
        # bf16=True,
        torch_compile = False,
        load_best_model_at_end = True,
        evaluation_strategy="steps",
        remove_unused_columns=False,

    )


# In[21]:


# Quantization
q_config = BitsAndBytesConfig(load_in_4bit=True,
                               bnb_4bit_quant_type='nf4',
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16
                               )


# In[22]:


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# In[23]:


# Load tokenizer & model
model_name = "baichuan-inc/Baichuan2-7B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=q_config,
        trust_remote_code=True,
        device='cuda'
    )
model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)


# In[24]:


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
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# In[25]:


# LoRA
TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['baichuan-inc/Baichuan2-7B-Base'] = ["W_pack", "o_proj"]
target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['baichuan-inc/Baichuan2-7B-Base']
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=target_modules,
    bias='none',
)
model = get_peft_model(model, lora_config)
print_trainable_parameters(model)


# In[26]:


resume_from_checkpoint = None
if resume_from_checkpoint is not None:
    checkpoint_name = os.path.join(resume_from_checkpoint, 'pytorch_model.bin')
    if not os.path.exists(checkpoint_name):
        checkpoint_name = os.path.join(
            resume_from_checkpoint, 'adapter_model.bin'
        )
        resume_from_checkpoint = False
    if os.path.exists(checkpoint_name):
        logger.info(f'Restarting from {checkpoint_name}')
        adapters_weights = torch.load(checkpoint_name)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        logger.info(f'Checkpoint {checkpoint_name} not found')


# In[27]:


model.print_trainable_parameters()


# In[28]:


# load data
dataset = datasets.load_from_disk("../data/dataset_new")
dataset = dataset.train_test_split(0.2, shuffle=True, seed = 42)


# In[29]:


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def prediction_step(self, model: torch.nn.Module, inputs, prediction_loss_only: bool, ignore_keys = None):
        with torch.no_grad():
            res = model(
                input_ids=inputs["input_ids"].to(model.device),
                labels=inputs["labels"].to(model.device),
            ).loss
        return (res, None, None)

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))

def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [tokenizer.pad_token_id] * (seq_len - 1) + ids[(seq_len - 1) :] + [tokenizer.pad_token_id] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


# In[30]:


from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback


# In[31]:


# Train
writer = SummaryWriter()
trainer = ModifiedTrainer(
    model=model,
    args=training_args,             # Trainer args
    train_dataset=dataset["train"], # Training set
    eval_dataset=dataset["test"],   # Testing set
    data_collator=data_collator,    # Data Collator
    callbacks=[TensorBoardCallback(writer)],
)
trainer.train()
writer.close()
# save model
model.save_pretrained(training_args.output_dir)


# In[32]:


get_ipython().system('zip -r /content/saved_model.zip /content/{training_args.output_dir}')


# In[33]:


from google.colab import files
files.download('/content/saved_model.zip')


# In[34]:


import os

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / 1024 / 1024  # Size in MB

model_size = get_folder_size(training_args.output_dir)
print(f"Model size: {model_size} MB")


# Inference in benchmarks

# In[35]:


get_ipython().system('pip install transformers==4.30.2 peft==0.4.0')
get_ipython().system('pip install sentencepiece')
get_ipython().system('pip install accelerate')
get_ipython().system('pip install torch')
get_ipython().system('pip install peft')
get_ipython().system('pip install datasets')
get_ipython().system('pip install bitsandbytes')


# In[36]:


get_ipython().system('git clone https://github.com/AI4Finance-Foundation/FinNLP.git')

import sys
sys.path.append('/content/FinNLP/')


# In[37]:


from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from peft import PeftModel
import torch

from finnlp.benchmarks.fpb import test_fpb
from finnlp.benchmarks.fiqa import test_fiqa , add_instructions
from finnlp.benchmarks.tfns import test_tfns
from finnlp.benchmarks.nwgi import test_nwgi


# In[38]:


get_ipython().system('pip install --upgrade peft')


# In[39]:


from google.colab import drive
drive.mount('/content/drive')


# In[40]:


import os

# Define the path you want to check
path_to_check = "/content/drive/My Drive/finGPTresearch/content"

# Check if the specified path exists
if os.path.exists(path_to_check):
    print("Path exists.")
else:
    print("Path does not exist.")


# In[45]:


base_model = "baichuan-inc/Baichuan2-7B-Base"
peft_model = training_args.output_dir

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, load_in_4bit=True, device_map="auto")

model = PeftModel.from_pretrained(model, peft_model)

model = model.eval()


# In[46]:


# base_model = "baichuan-inc/Baichuan2-7B-Base"
# peft_model = "/content/drive/My Drive/finGPTresearch/content/finetuned_model"

# tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
# model = AutoModel.from_pretrained(base_model, trust_remote_code=True, load_in_8bit=True, device_map="auto")

# model = PeftModel.from_pretrained(model, peft_model)

# model = model.eval()


# In[53]:


batch_size = 16


# In[54]:


# TFNS
res = test_tfns(model, tokenizer, batch_size = batch_size)


# In[ ]:


# FPB
res = test_fpb(model, tokenizer, batch_size = batch_size)


# In[ ]:


# FiQA
res = test_fiqa(model, tokenizer, prompt_fun = add_instructions, batch_size = batch_size)


# In[ ]:


# NWGI
res = test_nwgi(model, tokenizer, batch_size = batch_size)


# Conclusion:
# 
# ***TFNS:***
# Preivous:
# Acc: 0.876465661641541. F1 macro: 0.8422150651552788. F1 micro: 0.876465661641541. F1 weighted (BloombergGPT): 0.8760116165894634.
# 
# Current:
# Acc: 0.8618090452261307. F1 macro: 0.828965375193647. F1 micro: 0.8618090452261307. F1 weighted (BloombergGPT): 0.8632472452505952.
# 
# 
# 
# ----
# ***FRB:***
# Preivous:
# Acc: 0.8531353135313532. F1 macro: 0.8355311905054061. F1 micro: 0.8531353135313532. F1 weighted (BloombergGPT): 0.8518766219402919.
# 
# Current:
# Acc: 0.7970297029702971. F1 macro: 0.7575108820595529. F1 micro: 0.7970297029702971. F1 weighted (BloombergGPT): 0.7781642224313978.
# 
# ----
# ***FiQA:***
# Preivous:
# Acc: 0.8327272727272728. F1 macro: 0.7457078313253013. F1 micro: 0.8327272727272728. F1 weighted (BloombergGPT): 0.8475613362541076.
# 
# Current:
# Acc: 0.6254545454545455. F1 macro: 0.5741047534089461. F1 micro: 0.6254545454545455. F1 weighted (BloombergGPT): 0.6876670875957228.
# 
# 
# ----
# ***NWGI:***
# Preivous:
# Acc: 0.6375092661230541. F1 macro: 0.6449413175330677. F1 micro: 0.6375092661230541. F1 weighted (BloombergGPT): 0.6368355381269861.
# 
# Current:
# Acc: 0.5688164072152212. F1 macro: 0.569744027675689. F1 micro: 0.5688164072152212. F1 weighted (BloombergGPT): 0.5614102592371769.

# In[ ]:





# 

# In[ ]:





# In[ ]:




