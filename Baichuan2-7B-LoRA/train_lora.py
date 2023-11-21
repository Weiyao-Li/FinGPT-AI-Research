from transformers.integrations import TensorBoardCallback
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, TrainerCallback
from transformers.trainer import TRAINING_ARGS_NAME
from torch.utils.tensorboard import SummaryWriter
import datasets
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import os
import sys
import wandb
import argparse
from datetime import datetime
from functools import partial
from utils import *
import logging

logging.basicConfig(level=logging.INFO)

# LoRA
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

# Replace with your own api_key and project name
#os.environ['WANDB_API_KEY'] = 'ecf1e5e4f47441d46822d38a3249d62e8fc94db4'
#os.environ['WANDB_PROJECT'] = 'fingpt-benchmark'
# os.environ['WANDB_API_KEY'] = 'f01824d64a51d93e7e015b41c1c12bc764dc8548'
# os.environ['WANDB_PROJECT'] = 'AMD-FinLLMs'
# os.environ['WANDB_API_KEY'] = 'f84e0745716bae06b6e02fad953890c5f5042cab'
# os.environ['WANDB_PROJECT'] = 'sinogpt-finetuning'

os.environ['WANDB_API_KEY'] = '14b3cf25fb7dd1806db45e0dd2c68e83308daa04'
os.environ['WANDB_PROJECT'] = 'fingpt-benchmark'

class ProfCallback(TrainerCallback):

    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()

def main(args):

    model_name = parse_model_name(args.base_model, args.from_remote)

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # load_in_8bit=True,
        # device_map="auto",
        trust_remote_code=True)
    if args.local_rank == 0:
        print(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    if args.base_model != 'mpt':
        tokenizer.padding_side = "left"
    if args.base_model == 'qwen':
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(
            '<|endoftext|>')
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|extra_0|>')
    if not tokenizer.pad_token or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # load data
    dataset_list = load_dataset(args.dataset, args.from_remote)

    dataset_train = datasets.concatenate_datasets(
        [d['train'] for d in dataset_list]).shuffle(seed=42)

    if args.test_dataset:
        dataset_list = load_dataset(args.test_dataset, args.from_remote)

    dataset_test = datasets.concatenate_datasets(
        [d['test'] for d in dataset_list])

    dataset = datasets.DatasetDict({
        'train': dataset_train,
        'test': dataset_test
    })

    print(dataset['train'][0])

    dataset = dataset.map(partial(tokenize, args, tokenizer))
    print('original dataset length: ', len(dataset['train']))
    dataset = dataset.filter(lambda x: not x['exceed_max_length'])
    print('filtered dataset length: ', len(dataset['train']))
    dataset = dataset.remove_columns(
        ['instruction', 'input', 'output', 'exceed_max_length'])

    print(dataset['train'][0])

    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y%m%d%H%M')

    training_args = TrainingArguments(
        output_dir=f'finetuned_models/{args.run_name}_{formatted_time}',  # 保存位置
        logging_steps=args.log_interval,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        dataloader_num_workers=args.num_workers * 2,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        fp16=True,
        # bf16=True,
        # fp16_full_eval=True,
        deepspeed=args.ds_config,
        evaluation_strategy=args.evaluation_strategy,
        remove_unused_columns=False,
        report_to=args.report_to,
        run_name=args.run_name,
        save_total_limit=args.save_total_limit,
        max_steps=args.max_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        include_tokens_per_second=True)

    if not args.base_model == 'mpt':
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.config.use_cache = (False)
    # model = prepare_model_for_int8_training(model

    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=lora_module_dict[args.base_model],
        bias='none',
    )
    model = get_peft_model(model, peft_config)

    # Train
    writer = SummaryWriter()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer,
                                             padding=True,
                                             return_tensors="pt"),
        callbacks=[TensorBoardCallback(writer)],
    )

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    torch.cuda.empty_cache()

    def trace_handler(prof):
        prof.export_chrome_trace(
            f"/home/eacloud/bruce_yang/Benchmark/llama2_13b_training_trace_{prof.step_num}.json"
        )
        # prof.export_stacks("/home/eacloud/bruce_yang/Benchmark/llama2_13b_profiler_stacks.txt", "self_cuda_time_total")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=torch.profiler.schedule(skip_first=10,
                                                  wait=1,
                                                  warmup=1,
                                                  active=1,
                                                  repeat=1),
                 on_trace_ready=torch.profiler.tensorboard_trace_handler(
                     f"./log/{args.run_name}_trace"),
                 profile_memory=True,
                 with_stack=True) as prof:
        # trainer.add_callback(ProfCallback(prof=prof))
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    writer.close()
    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--run_name", default='local-test', type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument("--base_model",
                        required=True,
                        type=str,
                        choices=[
                            'chatglm2', 'llama2', 'llama2-13b', 'falcon',
                            'internlm', 'internlm-20b', 'qwen', 'mpt', 'bloom',
                            'baichuan'
                        ])
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--batch_size",
                        default=4,
                        type=int,
                        help="The train batch size per device")
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help="The learning rate")
    parser.add_argument("--num_epochs",
                        default=8,
                        type=float,
                        help="The training epochs")
    parser.add_argument("--num_workers",
                        default=8,
                        type=int,
                        help="dataloader workers")
    parser.add_argument("--log_interval", default=20, type=int)
    parser.add_argument("--warmup_ratio", default=0.05, type=float)
    parser.add_argument("--ds_config", default='./config_new.json', type=str)
    parser.add_argument("--scheduler", default='linear', type=str)
    parser.add_argument("--instruct_template", default='default')
    parser.add_argument("--evaluation_strategy", default='steps', type=str)
    parser.add_argument("--eval_steps", default=0.1, type=float)
    parser.add_argument("--from_remote", default=True, type=bool)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--save_total_limit", default=1, type=int)
    parser.add_argument("--report_to", type=str, default="wandb")
    args = parser.parse_args()

    wandb.login()
    main(args)
