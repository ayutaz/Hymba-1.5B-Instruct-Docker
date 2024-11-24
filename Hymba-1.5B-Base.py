from transformers import LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch

# Load the tokenizer and model
repo_name = "nvidia/Hymba-1.5B-Base"

tokenizer = AutoTokenizer.from_pretrained(repo_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo_name, trust_remote_code=True)
model = model.cuda().to(torch.bfloat16)

# Chat with Hymba
prompt = "まどマギで一番可愛いキャラクターは、"
inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
outputs = model.generate(**inputs, max_length=64, do_sample=True, temperature=0.7, use_cache=True)
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print(f"Model response: {response}")
