from FinNLP.finnlp.benchmarks.fpb import test_fpb
from FinNLP.finnlp.benchmarks.fiqa import test_fiqa, add_instructions
from FinNLP.finnlp.benchmarks.tfns import test_tfns
from FinNLP.finnlp.benchmarks.nwgi import test_nwgi

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from utils import *

FROM_REMOTE = False


def load_model(base_model, peft_model, from_remote=False):
    model_name = parse_model_name(base_model, from_remote)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        device_map="auto",
    )
    model.model_parallel = True

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    tokenizer.padding_side = "left"
    if base_model == 'qwen':
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|extra_0|>')
    if not tokenizer.pad_token or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(model, peft_model)
    model = model.eval()
    return model, tokenizer

if __name__ == "__main__":
    base_model = 'baichuan'
    peft_model = '/data/gpu/bruce_yang/weiyaoli/FinGPT-AI-Research/FinGPT_Benchmark/finetuned_models/sentiment-baichuan-7b-20epoch-8batch_202311130422#'

    model, tokenizer = load_model(base_model, peft_model, FROM_REMOTE)

    batch_size = 8
    res = test_tfns(model, tokenizer, batch_size=batch_size)
    print("tfns", res)

    res = test_fpb(model, tokenizer, batch_size=batch_size)
    print("fpb", res)

    res = test_fiqa(model, tokenizer, prompt_fun=add_instructions, batch_size=batch_size)
    print("fiqa", res)

    res = test_nwgi(model, tokenizer, batch_size=batch_size)
    print("nwgi", res)
