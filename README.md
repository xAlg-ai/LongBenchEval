Built on the top of InfLLM repo

```
python benchmark/pred.py --config_path llama.yaml --output_dir_path benchmark/test/ --datasets multifieldqa_zh \
                          --load_usa artifacts/<artifact> --chunk_size 256 --max_prompt_len 19000 \
                          --prefetch_offset 128 --token_budget 256 --limit 100 --baseline usa
```
1. use chunk_size if running on older GPUs or getting out of memory.
2. --token_budget 
		200 implies fixed budget
		0.25 implies dynamic budget of 1/4th of context length

3. usa -- legacy name for hashattention
