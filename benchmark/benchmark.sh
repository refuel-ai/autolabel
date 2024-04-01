export HF_HOME=/workspace/cache
python benchmark.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --few-shot 8
python benchmark.py --model mistralai/Mistral-7B-Instruct-v0.1 --few-shot 8
python benchmark.py --model gpt-3.5-turbo --few-shot 8
python benchmark.py --model gpt-4-1106-preview --few-shot 8
python benchmark.py --model claude-3-opus-20240229 --few-shot 8
python benchmark.py --model claude-3-sonnet-20240229 --few-shot 8