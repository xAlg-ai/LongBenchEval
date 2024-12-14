# https://github.com/THUDM/LongBench/blob/main/pred.py
import os
from datasets import load_from_disk
import torch
import json
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
from inf_llm.utils import patch_hf, GreedySearch, patch_model_center
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
#from transformers.models.llama.modeling_llama import FAISS,dump_faiss_stats,save_usa,load_usa, USA_STAT
import gc
import sys
from inf_llm.baselines.h2O_llama_from_ds import convert_h2o,reset_h2o
from inf_llm.baselines.doublesparse_llama import convert_kvcache_llama_heavy_recent, convert_llama_channel_config
from inf_llm.baselines.streaming_llama import convert_streaming
from inf_llm.baselines.usa_llama import convert_usa, load_usa_llama, reset_usa, set_train_usa_mode
from inf_llm.baselines.quest_attention import enable_quest_attention_eval
from inf_llm.baselines.quest import convert_quest
from inf_llm.baselines.topk_llama import convert_exact_topk

att_cfg_file = os.environ.get("ATT_CONFIG", None)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--model_center", action="store_true", default=False)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--truncate_len", type=int, default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--save_usa", type=str, default=None)
    parser.add_argument("--load_usa", type=str, default=None)
    parser.add_argument("--skip_first_examples", type=int, default=-1)
    parser.add_argument("--max_prompt_len", type=int, default=1000000)
    parser.add_argument("--min_prompt_len", type=int, default=-1)
    parser.add_argument("--samples", type=str, default=None)
    parser.add_argument("--prefetch_offset", type=int, default=1)
    parser.add_argument("--token_budget", type=float, default=4096)
    parser.add_argument("--edge_budget", type=int, default=128)
    parser.add_argument('--baseline', type=str, default=None)
    parser.add_argument('--collect_stats', action='store_true', default=False)
    parser.add_argument('--train_usa', action='store_true', default=False)
    parser.add_argument('--usa_ret_mode', type=str, default='depthnum')
    parser.add_argument('--ds_num_channels', type=int, default=16)
    parser.add_argument('--ds_label_bits', type=int, default=4)
    parser.add_argument('--quest_page_size', type=int, default=16)
    parser.add_argument('--quest_label_bits', type=int, default=4)
    parser.add_argument('--infllm_topk', type=int, default=16)
    parser.add_argument('--infllm_repr_topk', type=int, default=4)
    parser.add_argument('--infllm_exc_block_size', type=int, default=32)

    args, extra_args = parser.parse_known_args()
    conf = OmegaConf.load(args.config_path)
    cli_conf = OmegaConf.from_cli(extra_args)
    conf = OmegaConf.merge(conf, cli_conf)
    conf.output_dir_path = args.output_dir_path
    conf.model.model_center = args.model_center
    conf.rank = args.rank
    conf.world_size = args.world_size
    conf.verbose = args.verbose
    conf.limit = args.limit
    conf.runs = args.runs 
    conf.save_usa = args.save_usa
    conf.load_usa = args.load_usa
    conf.truncate_len = args.truncate_len
    conf.skip_first_examples = args.skip_first_examples
    conf.max_prompt_len = args.max_prompt_len
    conf.min_prompt_len = args.min_prompt_len
    if args.samples is not None:
        samples = []
        samples = [ int(x) for x in args.samples.strip().split(',')]
        conf.samples = samples
    else:
        conf.samples = None
    conf.baseline = args.baseline
    conf.token_budget = args.token_budget
    conf.edge_budget = args.edge_budget
    conf.prefetch_offset = args.prefetch_offset
    conf.collect_stats = args.collect_stats
    conf.train_usa = args.train_usa
    conf.usa_ret_mode = args.usa_ret_mode
    conf.ds_num_channels = args.ds_num_channels
    conf.ds_label_bits = args.ds_label_bits
    conf.quest_page_size = args.quest_page_size
    conf.quest_label_bits = args.quest_label_bits
    conf.model.topk = args.infllm_topk
    conf.model.repr_topk = args.infllm_repr_topk
    conf.model.exc_block_size = args.infllm_exc_block_size

    if not hasattr(conf.model, "tokenizer_path"):
        conf.model.tokenizer_path = conf.model.path
    if not hasattr(conf, "truncation"):
        conf.truncation = None

    datasets_str = args.datasets.strip().strip(",")
    datasets_list = datasets_str.split(",")
    conf.datasets = []
    for d in datasets_list:
        conf.datasets.append(d.strip())
    conf.chunk_size = args.chunk_size
    print(conf)
    return conf


def get_model_and_tokenizer(config, baseline, token_budget):
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    if config.model_center:
        import bmtrain as bmt
        bmt.init_distributed(seed=233)
        from model_center.model import Llama, LlamaConfig
        model_config = LlamaConfig.from_pretrained(config.path)
        model_config.dtype = torch.bfloat16
        model = Llama(model_config)
        bmt.load(model, os.path.join(config.path, "pytorch_model.pt"), strict=False)
        model = patch_model_center(model, config.type, **config)
    else:
        impl = "eager"
        if att_cfg_file is not None:
            impl = "eager"
        model = AutoModelForCausalLM.from_pretrained(config.path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda", attn_implementation=impl)
        if baseline is not None:
            assert(att_cfg_file is None)
            if baseline == "h2o":
                config = AutoConfig.from_pretrained(config.path)
                config.heavy_budget = args.token_budget
                config.recent_budget = args.edge_budget
                config.init_budget = args.edge_budget

                model = convert_h2o(model, config)
            elif baseline == "ds":
                channel_path = "/data/apdesai/DoubleSparse/config/" + config.path + ".json"
                config = AutoConfig.from_pretrained(config.path)
                channel_config = None
                with open(channel_path, "r") as f:
                    channel_config = json.load(f)
                    model = convert_kvcache_llama_heavy_recent(model, config, heavy_const=args.token_budget, 
                            group_factor=128//args.ds_num_channels, label_bits=args.ds_label_bits, init_const=args.edge_budget, local_const=args.edge_budget,
                            collect_stats=args.collect_stats)
                    #group_factor = 8 => sorted channels = 128 / 8
                    #label_bits = 16 # no quantization ; 4 => 4 bit quantization
                    model = convert_llama_channel_config(model, channel_config, "q")
            elif baseline == "inf-llm":
                model = patch_hf(model, baseline, **config)
            elif baseline == "streaming":
                config = AutoConfig.from_pretrained(config.path)
                model = convert_streaming(model, config, args.token_budget+args.edge_budget, args.edge_budget)
            elif baseline == "usa":
                config = AutoConfig.from_pretrained(config.path)
                config.lth_init_dim = 128
                config.lth_final_dim = 32
                config.lth_thold = 0
                config.init_budget = args.edge_budget
                config.heavy_budget = args.token_budget
                config.recent_budget = args.edge_budget
                config.usa_retrieve_depth = 6
                config.usa_eval_mode = args.usa_ret_mode
                usa_modules = load_usa_llama(config, args.load_usa)
                if args.train_usa:
                    usa_modules = usa_modules.bfloat16()
                model = convert_usa(model, config, usa_modules, collect_stats = args.collect_stats, train_usa=args.train_usa)

                if args.train_usa:
                    def loss_function(yhat ,ytarget, beta=20):
                        weight = ytarget * (beta - 1) + torch.ones_like(ytarget)
                        loss = torch.nn.functional.binary_cross_entropy(yhat.reshape(-1), ytarget.reshape(-1), weight = weight.reshape(-1))
                        return loss
                    optimizer = torch.optim.Adam(usa_modules.parameters(), lr = 0.001)
                    set_train_usa_mode(model, loss_function, optimizer)
            elif baseline == "quest":
                config = AutoConfig.from_pretrained(config.path)
                config.token_budget = args.token_budget
                config.chunk_size = args.quest_page_size
                config.init_budget = args.edge_budget
                config.recent_budget = args.edge_budget
                config.label_bits = args.quest_label_bits
                model = convert_quest(model, config, collect_stats=args.collect_stats)

            elif baseline == "quest_original":
                config.chunk_size = 32
                config.token_budget = args.token_budget
                config.edge_budget = args.edge_budget
                model = enable_quest_attention_eval(model, config)
            elif baseline == "exact":
                config = AutoConfig.from_pretrained(config.path)
                config.token_budget = args.token_budget
                config.init_budget = args.edge_budget
                config.recent_budget = args.edge_budget
                model = convert_exact_topk(model, config)
            else:
                raise NotImplementedError
            
    print(model)

        
    return model, tokenizer

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    model_name = model_name.strip().lower()
    if model_name == "vicuna":
        from fastchat.conversation import get_conv_template
        conv = get_conv_template("vicuna_v1.1")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif model_name in ["mistral-inst", "qwen", "minicpm", "llama-3-inst"]:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        raise NotImplementedError

    return prompt

def load_infinite_bench(path, data_name) -> str:
    import re
    """
    Create prompt for a given example.

    Args:
        eg: example dict
        data_name: name of the dataset/task
    """
    print(f"read {data_name}.jsonl")
    fin = open(os.path.join(path, data_name + ".jsonl"), "r")
    lines = fin.readlines()
    fin.close()
    data = [json.loads(line) for line in lines]
    def get_answer(inp: dict):
        if data_name in ["code_debug", "longbook_choice_eng"]:
            OPTIONS = "ABCD"
            if isinstance(inp["answer"], str):
                ret = [inp["answer"], OPTIONS[inp['options'].index(inp["answer"])]]
            elif isinstance(inp["answer"], list):
                if len(inp["answer"]) == 1:
                    ret = [inp["answer"][0], OPTIONS[inp['options'].index(inp["answer"][0])]]
                elif len(inp["answer"]) == 2 and inp["answer"][1] in ['A', 'B', 'C', 'D']:
                    ret = inp['answer']
                else:
                    raise ValueError
            else:
                raise ValueError
            return ret
        return inp["answer"]

    ret = []
    for eg in data:
        # ================= Code tasks
        if data_name == "code_run":
            find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", eg['input'])
            func_call = find_result[0]
            func = func_call.split("(")[0]
            instance = {"func": func, "func_call": func_call, "context": eg["context"]}
        elif data_name in ["code_debug", "code_debug_qa"]:
            # Load source code
            instance = {"context": eg["context"]}
            if data_name == "code_debug":
                instance.update({
                    "OPTION_A": eg["options"][0], 
                    "OPTION_B": eg["options"][1], 
                    "OPTION_C": eg["options"][2], 
                    "OPTION_D": eg["options"][3]})
        # ================= Code tasks
        elif data_name == "longdialogue_qa_eng":
            instance = {"context": eg["context"]}
        # ==================== Long book tasks
        elif data_name in [
            "longbook_choice_eng",
            "longbook_qa_eng",
            "longbook_sum_eng",
            "longbook_qa_chn",
        ]:
            instance = {"context": eg["context"]}
            if data_name == "longbook_choice_eng":
                instance.update({
                    "question": eg["input"],
                    "OPTION_A": eg["options"][0],
                    "OPTION_B": eg["options"][1],
                    "OPTION_C": eg["options"][2],
                    "OPTION_D": eg["options"][3],
                })
            elif data_name in ["longbook_qa_eng", "longbook_qa_chn"]:
                instance.update({
                    "question": eg["input"],
                })
        elif data_name == "math_calc":
            instance = {"context": eg["context"]}
        elif data_name == "math_find":
            prompt = eg['input']
            context = eg['context']
            # Find "the * number" from the prompt
            find_result = re.findall(r"The .+ of", prompt)
            assert find_result, f"Cannot find the target number in {prompt}"
            target_number = find_result[0].lower()[:-3]
            # Replace the number with the answer
            prefix = f"What is {target_number} in the following list?"
            instance = {"prefix": prefix, "context": context, "input": prompt}
        elif data_name == "kv_retrieval":
            instance = {
                "context": eg["content"] if "content" in eg else eg["context"],
                "input": eg["input"],
                "key": eg["input"][6:44]
            }
            assert eg['input'][6] == '"'
            assert eg['input'][43] == '"'
        else:
            instance = {
                "context": eg["content"] if "content" in eg else eg["context"],
                "input": eg["input"],
            }
        ans = get_answer(eg)
        instance["answers"] = ans if isinstance(ans, list) else [ans]
        instance["length"] = len(instance["context"].split())
        instance["all_classes"] = None
        
        ret.append(instance)
        # if len(ret) > 4:
        #     break
    return ret

def post_process(pred, model_name, dataset):
    if model_name == "qwen":
        pred = pred.split("<|im_end|>")[0]

    if dataset == "samsum":
        pred = pred.split("\n")[0].strip()

    return pred

def get_pred(
    args, model, tokenizer, data, max_length,
    max_gen, prompt_format, dataset, model_name, 
    gen_chunk_size = None, truncation: str = None, 
    rank: int = None, world_size: int = None,
    verbose: bool = False, limit=None,
    truncate_len: int = None,
    save_usa_path: str = None,
    skip_first_examples: int = -1,
    max_prompt_len: int = 1000000000,
    min_prompt_len: int = -1,
    samples = None,
    prefetch_offset = 1,
):
    if save_usa_path is not None:
        save_usa(save_usa_path)

    preds = []
    data = list(data)
    if samples is not None:
        data = [data[i] for i in samples]
    if world_size is not None:
        data = data[rank::world_size]

    searcher = GreedySearch(model, tokenizer)
    cur = 0
    total = len(data)

    for i, json_obj in tqdm(enumerate(data)):
        if i < skip_first_examples:
            print("skip_first_examples", i)
            continue
        gc.collect()
        #if len(FAISS) > 0:
        #    print("resetting FAISS")
        #    for _i in range(len(FAISS)):
        #        for _j in range(len(FAISS[_i])):
        #            FAISS[_i][_j].reset()
        #    print("resetting done")
        if args.baseline is not None:
            if args.baseline == "h2o":
                reset_h2o(model)
            if args.baseline == "usa":
                reset_usa(model)
        if limit is not None and i >= limit:
            break
        prompt = prompt_format.format(**json_obj)

        extra_end_token_ids = []
        if model_name == "llama-3-inst":
            extra_end_token_ids.append(tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0])

        if model_name == "qwen":
            extra_end_token_ids.append(tokenizer.encode("<|im_end|>", add_special_tokens=False)[0])

        if dataset == "samsum":
            extra_end_token_ids.append(tokenizer.encode("\n", add_special_tokens=False)[-1])

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: 
            # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

            if model_name.strip().lower() in ['mistral-inst']:
                add_special_tokens = False
            else:
                add_special_tokens = True
        
        else:
            add_special_tokens = True

        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]

        if truncation is None:
            if len(tokenized_prompt) > max_length - max_gen:
                if verbose:
                    print(f"Length {len(tokenized_prompt)}. Skipped.")
                continue

        else:
            if truncation == "suffix":
                length = len(tokenized_prompt)
                if length > max_length - max_gen:
                    if verbose:
                        print("over length")
                    init_token_num = 128
                    prompt = tokenizer.decode(tokenized_prompt[:init_token_num].tolist() + tokenized_prompt[- (max_length - max_gen - init_token_num):].tolist())
                    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]
            else:
                raise NotImplementedError
    
        print(tokenized_prompt.shape, flush=True)
        if tokenized_prompt.shape[0] > max_prompt_len:
            print("too long",tokenized_prompt.shape, "Skipping")
            continue
        if tokenized_prompt.shape[0] < min_prompt_len:
            print("too short",tokenized_prompt.shape, "Skipping")
            continue
        if truncate_len is not None:
            tokenized_prompt = tokenized_prompt[:truncate_len]
        output = searcher.generate(
            input_ids = tokenized_prompt,
            max_length=max_gen,
            chunk_size=gen_chunk_size,
            extra_end_token_ids=extra_end_token_ids,
            prefetch_offset=prefetch_offset
        )

        pred = post_process(output[0], model_name, dataset)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"], "token_length": len(tokenized_prompt) + max_gen})
        searcher.clear()
        cur += 1
        if verbose:
            print(f"----------{cur}/{total}----------")
            print("Length: ", len(tokenized_prompt))
            print("Question:", prompt[-100:])
            print("Pred:", pred)
            print("Answer:", json_obj["answers"])
            print("", flush=True)
        if save_usa_path is not None:
            save_usa(save_usa_path)
        #if USA_STAT is not None:
        #    print(USA_STAT)
    return preds


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args = parse_args()

    dump_cfg = args.output_dir_path + "config." + '-'.join(args.datasets) + '.yaml'
    with open(dump_cfg, "w") as f:
        OmegaConf.save(config=args, f=dump_cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # define your model
    model, tokenizer = get_model_and_tokenizer(args.model, args.baseline, args.token_budget)
    output_dir_path = args.output_dir_path

    datasets = args.datasets


    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("benchmark/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("benchmark/config/dataset2maxlen.json", "r"))

    # if args.load_usa is not None:
    #     print("LOADING USA...", flush=True)
    #     load_usa(args.load_usa)
    
    multiprocessing = args.world_size is not None and args.world_size > 1
    if multiprocessing:
        assert args.rank in list(range(args.world_size))

    # predict on each dataset
    for run in range(args.runs): # for USA training
        for dataset in datasets:
            dname = dataset
            if dataset in set([
                "kv_retrieval", "passkey", "number_string", "code_run", "code_debug", "longdialogue_qa_eng", "longbook_qa_eng", "longbook_sum_eng", "longbook_choice_eng", "longbook_qa_chn", "math_find", "math_calc"
            ]):
                path = "benchmark/data/infinite-bench"
                data = load_infinite_bench(path, dname)

            else:
                data = load_from_disk(
                    f"benchmark/data/longbench/{dataset}"
                )

            out_path = os.path.join(
                output_dir_path,
                f"{dname}.jsonl"
            )

            print(f"Pred {dname}")
            prompt_format = dataset2prompt[dataset]

            max_gen = dataset2maxlen[dataset]
            preds = get_pred(args,
                model, tokenizer, data, 
                args.max_len, max_gen, 
                prompt_format, dataset, 
                args.conv_type, 
                args.chunk_size, args.truncation,
                args.rank, args.world_size,
                args.verbose,
                args.limit,
                args.truncate_len,
                args.save_usa,
                args.skip_first_examples,
                args.max_prompt_len,
                args.min_prompt_len,
                args.samples,
                args.prefetch_offset
            )
            if multiprocessing:
                out_path = out_path + f"_{args.rank}"
            with open(out_path, "w+", encoding="utf-8") as f:
                for pred in preds:
                    json.dump(pred, f, ensure_ascii=False)
                    f.write('\n')

    att_cfg_file = os.environ.get("ATT_CONFIG", None)
    if att_cfg_file is not None:
        basename = os.path.basename(att_cfg_file).strip('.yaml')
        dump_faiss_stats("./logs/stats-"+basename+".npz")

