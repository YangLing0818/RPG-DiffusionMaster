import openai

total_tokens = 4096
max_output_tokens = 1024
max_input_tokens = total_tokens - max_output_tokens

from mllm import get_params_dict

def GPT4_by_openapi(prompt, version, key, model_name="gpt-4", temperature=0.0):
    url = "https://api.openai.com/v1/chat/completions"
    openai.api_key = key
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
    
    # model_name = "gpt-4-1106-preview"
    model_name = "gpt-4"
    messages = [
                {
                    "role": "user",
                    "content": textprompt
                }
            ]
            
    print(f'waiting for GPT-4 api response')
    obj = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_output_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=[" Human:", " AI:"]
    )            

    print(f'GPT-4_by_openapi response__{obj}')
    text=obj['choices'][0]['message']['content']

    # Extract the split ratio and regional prompt

    return get_params_dict(text)

style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Disney Charactor",
        "prompt": "A Pixar animation character of {prompt} . pixar-style, studio anime, Disney, high-quality",
        "negative_prompt": "lowres, bad anatomy, bad hands, text, bad eyes, bad arms, bad legs, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, blurry, grayscale, noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Photographic (Default)",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "Enhance",
        "prompt": "breathtaking {prompt} . award-winning, professional, highly detailed",
        "negative_prompt": "ugly, deformed, noisy, blurry, distorted, grainy",
    },
    {
        "name": "Comic book",
        "prompt": "comic {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
        "negative_prompt": "photograph, deformed, glitch, noisy, realistic, stock photo",
    },
    {
        "name": "Lowpoly",
        "prompt": "low-poly style {prompt} . low-poly game art, polygon mesh, jagged, blocky, wireframe edges, centered composition",
        "negative_prompt": "noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo",
    },
    {
        "name": "Line art",
        "prompt": "line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics",
        "negative_prompt": "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic",
    }
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}

model_name_list = [
        {
            "model_name": "absolutereality_v181.safetensors",
            "model_style_name": "realistic style",
            "size": "2.13G",
            "url": "https://civitai.com/models/81458/absolutereality",
        },
        {
            "model_name": "anything-v3-full.safetensors",
            "model_style_name": "anime style",
            "size": "7.7G",
            "url": "https://civitai.com/models/81458/absolutereality",
        },
        {
            "model_name": "disneyPixarCartoon_v10.safetensors",
            "model_style_name": "cartoon style",
            "size": "4.24G",
            "url": "https://civitai.com/models/75650/disney-pixar-cartoon-type-b",
        },
        {
            "model_name": "albedobaseXL_v20.safetensors",
            "model_style_name": "SDXL-baed photorealistic style",
            "size": "9.94G",
            "url": "https://civitai.com/models/140737?modelVersionId=281176",
        },
        {
            "model_name": "dreamshaperXL_turboDpmppSDE.safetensors",
            "model_style_name": "SDXL-Turbo based photorealistic",
            "size": "6.49G",
            "url": "https://civitai.com/models/112902/dreamshaper-xl",
        },
    ]
models = {k["model_style_name"]: (k["model_name"], k["model_style_name"]) for k in model_name_list}


