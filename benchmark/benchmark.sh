export HF_HOME=/workspace/cache
# python benchmark.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --few-shot 8
# python benchmark.py --model mistralai/Mistral-7B-Instruct-v0.1 --few-shot 8
# python benchmark.py --model gpt-3.5-turbo --few-shot 8
# python benchmark.py --model gpt-4-1106-preview --few-shot 8
# python benchmark.py --model claude-3-opus-20240229 --few-shot 8
# python benchmark.py --model claude-3-sonnet-20240229 --few-shot 8

aws s3 sync s3://refuel-forge/axolotl_test/dry_runs/llm_v2_lctxt_2/2500/ /workspace/2500_lctxt
python /workspace/add_tokenizer.py --output_dir /workspace/2500_lctxt
python benchmark.py --model /workspace/2500_lctxt --few-shot 8 --max-items 200

aws s3 sync s3://refuel-forge/refuel_llm_v2/7m_v2/21000/ /workspace/21000v2refuelllm
python /workspace/add_tokenizer.py --output_dir /workspace/21000v2refuelllm
python benchmark.py --model /workspace/21000v2refuelllm --few-shot 8 --max-items 200

aws s3 sync s3://refuel-forge/refuel_llm_v2/7m_v2/9000/ /workspace/9000v2refuelllm
python /workspace/add_tokenizer.py --output_dir /workspace/9000v2refuelllm
python benchmark.py --model /workspace/9000v2refuelllm --few-shot 8 --max-items 200

#aws s3 sync s3://refuel-forge/axolotl_test/dry_runs/test_mixtral_lctxt_6/4500/ /workspace/4500lctxt6
#python /workspace/add_tokenizer.py --output_dir /workspace/4500lctxt6
#python benchmark.py --model /workspace/4500lctxt6 --few-shot 8 --max-items 200

# python benchmark.py --model /workspace/refuelllm-1p-900 --few-shot 0 --max-items 200
# python benchmark.py --model gpt-4-1106-preview --few-shot 0 --max-items 200
# python benchmark.py --model claude-3-opus-20240229 --few-shot 0 --max-items 200