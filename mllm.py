import requests
import json
import os
from transformers import AutoTokenizer
import transformers
import torch
import re
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

def extract_output(text):
    # Find the output in the text
    output_pattern = r'### Output:(.*?)(?=###|$)'
    output_match = re.search(output_pattern, text, re.DOTALL)
    return output_match.group(1).strip() if output_match else None

def GPT4(prompt,version,key):
    url = "https://api.aigcbest.top/v1/chat/completions"
    api_key = key
    with open('template/template.txt', 'r') as f:
        template=f.readlines()
    if version=='multi-attribute':
        with open('template/human_multi_attribute_examples.txt', 'r') as f:
            incontext_examples=f.readlines()
    elif version=='complex-object':
        with open('template/complex_multi_object_examples.txt', 'r') as f:
            incontext_examples=f.readlines()
    user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
    
    textprompt= f"{' '.join(template)} \n {' '.join(incontext_examples)} \n {user_textprompt}"
    
    payload = json.dumps({
    "model": "gpt-4-all",
    "messages": [
        {
            "role": "user",
            "content": textprompt
        }
    ]
    })
    headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {api_key}',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
    }
    print('waiting for GPT-4 response')
    response = requests.request("POST", url, headers=headers, data=payload)
    obj=response.json()
    text=obj['choices'][0]['message']['content']
    print(text)
    # Extract the split ratio and regional prompt

    return get_params_dict(text)

def local_llm(prompt,version,model_path=None):
    if model_path==None:
        model_id = "Llama-2-13b-chat-hf" 
    else:
        model_id=model_path
    print('Using model:',model_id)
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=False, device_map='auto', torch_dtype=torch.float16)
    with open('template/template.txt', 'r') as f:
        template=f.readlines()
    if version=='multi-attribute':
        with open('template/human_multi_attribute_examples.txt', 'r') as f:
            incontext_examples=f.readlines()
    elif version=='complex-object':
        with open('template/complex_multi_object_examples.txt', 'r') as f:
            incontext_examples=f.readlines()
    user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
    textprompt= f"{' '.join(template)} \n {' '.join(incontext_examples)} \n {user_textprompt}"
    model_input = tokenizer(textprompt, return_tensors="pt").to("cuda")
    model.eval()
    with torch.no_grad():
        print('waiting for LLM response')
        res = model.generate(**model_input, max_new_tokens=1024)[0]
        output=tokenizer.decode(res, skip_special_tokens=True)
        output = output.replace(textprompt,'')
    return get_params_dict(output)

def get_params_dict(output_text):
    split_ratio_marker = "Split ratio: "
    regional_prompt_marker = "Regional Prompt: "
    output_text=extract_output(output_text)
    print(output_text)
    # Find the start and end indices for the split ratio and regional prompt
    split_ratio_start = output_text.find(split_ratio_marker) + len(split_ratio_marker)
    split_ratio_end = output_text.find("\n", split_ratio_start)
    regional_prompt_start = output_text.find(regional_prompt_marker) + len(regional_prompt_marker)
    regional_prompt_end = len(output_text)  # Assuming Regional Prompt is at the end

    # Extract the split ratio and regional prompt from the text
    split_ratio = output_text[split_ratio_start:split_ratio_end].strip()
    regional_prompt = output_text[regional_prompt_start:regional_prompt_end].strip()
    #Delete the possible "(" and ")" in the split ratio
    split_ratio=split_ratio.replace('(','').replace(')','')
    # Create the dictionary with the extracted information
    image_region_dict = {
        'split ratio': split_ratio,
        'Regional Prompt': regional_prompt
    }
    print(image_region_dict)
    return image_region_dict