# https://github.com/THUDM/LongBench/blob/main/pred.py
import os
from datasets import load_from_disk, load_dataset
import torch
import json
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
from inf_llm.utils import patch_hf, GreedySearch, patch_model_center
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import gc
import sys

MODELNAME = os.environ.get("MODEL", "llama")
if MODELNAME == "llama":
    from inf_llm.baselines.usa_llama import convert_usa, load_usa, reset_usa, set_train_usa_mode, set_eval_usa_mode, print_stats
elif MODELNAME == "mistral":
    from inf_llm.baselines.usa_mistral import convert_usa, load_usa, reset_usa, set_train_usa_mode, set_eval_usa_mode, print_stats
else:
    raise NotImplementedError

att_cfg_file = os.environ.get("ATT_CONFIG", None)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--train_dataset", type=str, default=None)
    parser.add_argument("--validation_dataset", type=str, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--truncate_len", type=int, default=64000)
    
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save_usa", type=str, default=None)
    parser.add_argument("--load_usa", type=str, default=None)
    parser.add_argument("--token_budget", type=int, default=4096)
    parser.add_argument("--edge_budget", type=int, default=128)
    parser.add_argument("--skip_first_examples", type=int, default=-1)

    #loss stuff
    parser.add_argument("--loss", type=str, default='bce')
    parser.add_argument("--bce_alpha", type=float, default=20.0)
    parser.add_argument("--bce_beta", type=float, default=0.)
    parser.add_argument('--usa_num_layers', type=int, default=3)
    parser.add_argument('--usa_final_dim', type=int, default=32)
    

    args, extra_args = parser.parse_known_args()
    conf = OmegaConf.load(args.config_path)
    cli_conf = OmegaConf.from_cli(extra_args)
    conf = OmegaConf.merge(conf, cli_conf)
    conf.rank = args.rank
    conf.world_size = args.world_size
    conf.verbose = args.verbose
    conf.limit = args.limit
    conf.epochs = args.epochs
    conf.save_usa = args.save_usa
    conf.load_usa = args.load_usa
    conf.truncate_len = args.truncate_len
    conf.token_budget = args.token_budget
    conf.edge_budget = args.edge_budget
    conf.skip_first_examples = args.skip_first_examples
    conf.loss = args.loss
    conf.bce_alpha = args.bce_alpha
    conf.bce_beta = args.bce_beta
    conf.usa_num_layers = args.usa_num_layers
    conf.usa_final_dim = args.usa_final_dim

    if not hasattr(conf.model, "tokenizer_path"):
        conf.model.tokenizer_path = conf.model.path
    if not hasattr(conf, "truncation"):
        conf.truncation = None

    train_datasets_str = args.train_dataset.strip().strip(",")
    train_datasets_list = train_datasets_str.split(",")
    conf.train_datasets = []
    for d in train_datasets_list:
        conf.train_datasets.append(d.strip())

    conf.validation_dataset = args.validation_dataset
    

    conf.chunk_size = args.chunk_size

    print(conf)
    return conf


def get_model_and_tokenizer(config, args):
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(config.path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda", attn_implementation="eager")
    config = AutoConfig.from_pretrained(config.path)

    config.lth_init_dim = 128
    config.lth_final_dim = args.usa_final_dim
    config.lth_thold = 0
    config.init_budget = args.edge_budget
    config.heavy_budget = args.token_budget
    config.recent_budget = args.edge_budget
    config.usa_retrieve_depth = 6
    config.usa_eval_mode = "vanilla"
    config.lth_num_layers = args.usa_num_layers
    usa_modules = load_usa(config, args.load_usa)
    usa_modules = usa_modules.bfloat16()
    model = convert_usa(model, config, usa_modules, collect_stats=True, train_usa=True)

    def get_bce_loss_function(alpha, beta):
        def loss_function(yhat ,ytarget):
            w = ytarget.shape[-1] * beta +  alpha
            weight = ytarget * (w - 1) + torch.ones_like(ytarget)
            loss = torch.nn.functional.binary_cross_entropy(yhat.reshape(-1), ytarget.reshape(-1), weight = weight.reshape(-1))
            return loss
        return loss_function
    optimizer = torch.optim.Adam(usa_modules.parameters(), lr = 0.001)

    if args.loss == 'bce':
        print("Using BCE loss with", args.bce_alpha, args.bce_beta)
        loss_function = get_bce_loss_function(args.bce_alpha, args.bce_beta)
    else:
        raise NotImplementedError
    set_train_usa_mode(model, loss_function, optimizer)
    print(model)

        
    return model, tokenizer, usa_modules

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    model_name = model_name.strip().lower()
    #  model_name in ["mistral-inst", "qwen", "minicpm", "llama-3-inst"]:
    assert model_name == "llama-3-inst"
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def train(
    args, model, tokenizer, 
    train_data, valid_data,
    train_prompt_format, valid_prompt_format,
    max_gen, gen_chunk_size,
    tr_truncate_len, 
    save_usa_path,
    usa_modules_ptr,
    skip_first_examples = -1,
    finetune=False
):
    searcher = GreedySearch(model, tokenizer)
    cur = 0

    text = ""
    text_len = 0
    char_to_token_factor = 5
    itr = 0

    for i, json_obj in tqdm(enumerate(train_data)):
        if i < skip_first_examples:
            continue
        gc.collect()
        if not finetune:
            text_len += len(json_obj['text'])
            text = text + "Passage: " + json_obj['text']
            if text_len < tr_truncate_len * char_to_token_factor:
                #accumulate more text
                continue

            prompt = train_prompt_format.format(text=text)
        else:
            prompt = train_prompt_format.format(**json_obj)
        # reset values
        text = ""
        text_len = 0

        extra_end_token_ids = []
        ## ASSERT MODEL IS LLAMA
        extra_end_token_ids.append(tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0])
        add_special_tokens = True
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]    
        print(tokenized_prompt.shape, len(text), flush=True)
        if tr_truncate_len is not None:
            tokenized_prompt = tokenized_prompt[:tr_truncate_len]
        searcher.clear()
        output = searcher.generate(
            input_ids = tokenized_prompt,
            max_length=max_gen,
            chunk_size=gen_chunk_size,
            extra_end_token_ids=extra_end_token_ids,
            prefetch_offset=1
        )
        itr += 1
        if (itr+1) % 100 == 0:
            set_eval_usa_mode(model)
            print("Evaluating ... ")
            for j, json_obj_valid in tqdm(enumerate(valid_data)):
                gc.collect()
                prompt = valid_prompt_format.format(**json_obj_valid)
                extra_end_token_ids = []
                ## ASSERT MODEL IS LLAMA
                extra_end_token_ids.append(tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0])
                add_special_tokens = True
                tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]    
                print(tokenized_prompt.shape, flush=True)
                if tr_truncate_len is not None:
                    tokenized_prompt = tokenized_prompt[:tr_truncate_len]
                searcher.clear()
                output = searcher.generate(
                                input_ids = tokenized_prompt,
                                max_length=max_gen,
                                chunk_size=gen_chunk_size,
                                extra_end_token_ids=extra_end_token_ids,
                                prefetch_offset=1
                         )
                break
            print_stats(model)
            reset_usa(model)
            set_train_usa_mode(model)
            print("Training ... ")
        if save_usa_path is not None:
            torch.save(usa_modules_ptr.cpu().state_dict(), save_usa_path)
            usa_modules_ptr = usa_modules_ptr.cuda()



def get_dataset(dataset):
    if dataset == "openwebtext":
        dataset =  load_dataset("Skylion007/openwebtext", trust_remote_code=True)
        data = dataset['train']
    # benchmark datasets
    elif dataset in set([
                    "kv_retrieval", "passkey", "number_string", "code_run", "code_debug", "longdialogue_qa_eng", "longbook_qa_eng", "longbook_sum_eng", "longbook_choice_eng", "longbook_qa_chn", "math_find", "math_calc"
                    ]):
        path = "benchmark/data/infinite-bench"
        data = load_infinite_bench(path, dname)
    else:
        data = load_from_disk(
                    f"benchmark/data/longbench/{dataset}"
        )
    return data


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define your model
    model, tokenizer, usa_modules = get_model_and_tokenizer(args.model, args)

    train_datasets = args.train_datasets
    validation_dataset = args.validation_dataset

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("benchmark/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("benchmark/config/dataset2maxlen.json", "r"))
    
    valid_data = get_dataset(validation_dataset)

    for epoch in range(args.epochs): # for USA training
        for train_dataset in train_datasets:
            train_data = get_dataset(train_dataset)
            
            print(f"Train {train_dataset}")
            train_prompt_format = dataset2prompt[train_dataset]
            valid_prompt_format = dataset2prompt[validation_dataset]
            max_gen = 1
            train( args, model, tokenizer, 
                        train_data, valid_data,
                        train_prompt_format, valid_prompt_format,
                        max_gen, args.chunk_size, args.truncate_len,
                        args.save_usa, usa_modules, args.skip_first_examples, 
                        train_dataset != "openwebtext"
                       )
