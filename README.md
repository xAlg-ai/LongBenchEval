Built on the top of InfLLM repo. See the original repository for how to install requirements and download data.

Sample evaluation:
```
MODEL=llama python benchmark/pred.py --config_path llama.yaml --output_dir_path benchmark/test/ --datasets passage_retrieval_en --load_usa HashAttention-1.0/artifacts/llama3.1-8b-patch.32K.v1.pt --chunk_size 256 --max_prompt_len 19000  --prefetch_offset 128 --token_budget 256 --baseline usa --overwrite  --verbose
```
1. use chunk_size if running on older GPUs or getting out of memory.
2. --token_budget 
		200 implies fixed budget
		0.25 implies dynamic budget of 1/4th of context length

3. usa -- legacy name for hashattention
