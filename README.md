Built on the top of InfLLM repo

```
python benchmark/pred.py --config_path llama.yaml --output_dir_path benchmark/test/ --datasets multifieldqa_zh \
                          --load_usa artifacts/usa.all25.t64.b32.th0.1400itr.pt --chunk_size 256 --max_prompt_len 19000 \
                          --prefetch_offset 128 --token_budget 256 --limit 100 --baseline usa
```
