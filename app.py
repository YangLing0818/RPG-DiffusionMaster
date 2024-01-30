import os, sys, time, re, random, json
import logging
logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())
from modules import timer, errors
import torch
startup_timer = timer.Timer()
from modules import extensions
import modules.scripts
from mllm import GPT4,local_llm
import argparse
from template.demo import demo_list

import gradio as gr
import numpy as np

from utils_rpg import styles, models

STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Photographic (Default)" # "Photographic (Default)"
def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + ' ' + negative

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "1") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1920"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

MODEL_STYLE_NAMES = list(models.keys())
DEFAULT_MODEL_STYLE_NAME = "SDXL-baed photorealistic style"

myFilename = os.path.basename(__file__)

WORKING_ROOT = './workshops'
tempDir = f'{WORKING_ROOT}'
if not os.path.exists(tempDir):
    os.makedirs(tempDir)

def load_model(model_name=None):
    from modules import shared
    from modules.shared import cmd_opts
    
    from modules import options, shared_options
    shared.options_templates = shared_options.options_templates #{}
    shared.opts = options.Options(shared_options.options_templates, shared_options.restricted_opts)
    shared.restricted_opts = shared_options.restricted_opts
    if os.path.exists(shared.config_filename):
        shared.opts.load(shared.config_filename)
    extensions.list_extensions()
    startup_timer.record("list extensions")
    
    from modules import devices
    devices.device, devices.device_interrogate, devices.device_gfpgan, devices.device_esrgan, devices.device_codeformer = \
        (devices.cpu if any(y in cmd_opts.use_cpu for y in [x, 'all']) else devices.get_optimal_device() for x in ['sd', 'interrogate', 'gfpgan', 'esrgan', 'codeformer'])

    devices.dtype = torch.float32 if cmd_opts.no_half else torch.float16
    devices.dtype_vae = torch.float32 if cmd_opts.no_half or cmd_opts.no_half_vae else torch.float16

    shared.device = devices.device
    shared.weight_load_location = None if cmd_opts.lowram else "cpu"
    from modules import shared_state
    shared.state = shared_state.State()

    from modules import styles
    shared.prompt_styles = styles.StyleDatabase(shared.styles_filename)
    from modules import interrogate
    shared.interrogator = interrogate.InterrogateModels("interrogate")

    from modules import shared_total_tqdm
    shared.total_tqdm = shared_total_tqdm.TotalTQDM()

    from modules import memmon, devices
    shared.mem_mon = memmon.MemUsageMonitor("MemMon", devices.device, shared.opts)
    shared.mem_mon.start()

    import modules.sd_models
    modules.sd_models.setup_model() # load models
    modules.sd_models.list_models()
    startup_timer.record("list SD models")

    modules.scripts.load_scripts()
    modules.scripts.scripts_txt2img.initialize_scripts(is_img2img=False)
    
    startup_timer.record("load scripts")
    print('txt2img_scripts',modules.scripts.scripts_txt2img.scripts)
 
    try:
        modules.sd_models.load_model(model_name=model_name)
    except Exception as e:
        errors.display(e, "loading stable diffusion model")
        print("", file=sys.stderr)
        print("Stable diffusion model failed to load, exiting", file=sys.stderr)
        exit(1)
    startup_timer.record("load SD checkpoint")
 
negative_prompt_value = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, asymmetry, illustration, 3d, 2d, painting, cartoons, sketch, open mouth"

def RPG(user_prompt, diffusion_model, version, split_ratio=None, use_llm=True,key=None, use_gpt=True, use_local=False,
        llm_path=None, activate=True, use_base=False, base_ratio=0, base_prompt=None, batch_size=1, 
        seed=1234, demo=False, use_personalized=False, cfg=5, steps=20, height=1024, width=1024,
        model_name = "albedobaseXL_v20.safetensors",
        style = DEFAULT_STYLE_NAME,        
        negative_prompt_value=negative_prompt_value,
        timestamp='',
        ):
    global opt

    # Prompt for regional diffusion
    if demo:# and opt.user_prompt != 'GRADIO':
        regional_prompt=user_prompt
        #TODO: add personalized regional split and regional prompt 
    else:
        user_prompt, negative_prompt_value = apply_style(style, user_prompt, negative_prompt_value)        
        if use_llm:    
            input_prompt=user_prompt
            if use_gpt:
                # assert key is not None
                if key is None:
                    key = os.getenv("OPENAI_API_KEY")            
                if 0==1:
                    params=GPT4(input_prompt,version,key)
                else:
                    from utils_rpg import GPT4_by_openapi
                    params=GPT4_by_openapi(input_prompt,version,key)
            elif use_local:
                params=local_llm(input_prompt,version,model_path=llm_path)
            
            regional_prompt=params['Regional Prompt']
            split_ratio=params['split ratio']
            if use_base:
                if base_prompt is None:
                    regional_prompt= user_prompt+' BREAK\n'+regional_prompt
                else:
                    regional_prompt= base_prompt+' BREAK\n'+regional_prompt
        else:
            regional_prompt= user_prompt
        
    # if opt.user_prompt == 'GRADIO' and timestamp != "": 
    if not demo and timestamp != "": 
        regional_data ={
                    'model': model_name,
                    'seed': seed,
                    'CFG': cfg,
                    'steps': steps,
                    'Split_ratio': split_ratio,
                    'Use_Base': use_base,
                    'Use_common': False,
                    'Base_ratio': base_ratio,
                    'width': width,
                    'height': height,
                    'Regional_prompt': regional_prompt,
                    },

        tempDir = f'{WORKING_ROOT}/{timestamp}'
        if not os.path.exists(tempDir):
            os.makedirs(tempDir)            
        with open(f"{tempDir}/regional_setting.json", "w") as file:
            json.dump(regional_data, file)
        return []
    else:
        # set model
        import modules.txt2img        

        # Regional settings:
        regional_settings = {
            'activate':activate, # To activate regional diffusion, set True, if don't use this, set False
            'split_ratio':split_ratio, # Split ratio for regional diffusion, default is 1,1, which means vertically split the image into two regions with same height and width,
            'base_ratio':base_ratio, # The weight of base prompt
            'use_base':use_base, # Whether to use base prompt
            'use_common':False # Whether to use common prompt
            }
        image, _, _, _ = modules.txt2img.txt2img(
            id_task="task",
            prompt=regional_prompt,
            negative_prompt=negative_prompt_value,
            prompt_styles=[],
            steps=steps,
            sampler_index=0,
            restore_faces=False,
            tiling=False,
            n_iter=1,
            batch_size=batch_size,
            cfg_scale=cfg,
            seed=seed, # -1 means random, choose a number larger than 0 to get a deterministic result
            subseed=-1,
            subseed_strength=0,
            seed_resize_from_h=0,
            seed_resize_from_w=0,
            seed_enable_extras=False,
            height=height,
            width=width,
            enable_hr=False,
            denoising_strength=0.7,
            hr_scale=0,
            hr_upscaler="Latent",
            hr_second_pass_steps=0,
            hr_resize_x=0,
            hr_resize_y=0,
            override_settings_texts=[],
            **regional_settings,
        )
        return image

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def generate(
        prompt: str = "",
        negative_prompt: str = "",
        use_negative_prompt: bool = False,
        use_llm: bool = True,
        model_style_name:str = DEFAULT_MODEL_STYLE_NAME,
        style:str = DEFAULT_STYLE_NAME,
        seed: int = 0,
        width: int = 1024,
        height: int = 1024,
        randomize_seed: bool = False,
        num_inference_steps:int = 25,
        num_images_per_prompt:int = 1,
        use_resolution_binning: bool = True,
        progress=gr.Progress(track_tqdm=True),
    ):
    global opt

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    startTime = time.time()

    seed = int(randomize_seed_fn(seed, randomize_seed))

    if not use_negative_prompt:
        negative_prompt = None  # type: ignore
    
    model_name, _ = models.get(model_style_name, models[DEFAULT_MODEL_STYLE_NAME])

    version_list=['multi-attribute','complex-object'] # Choose different in-context examples for generation, for multi-people attribute, choose the first one, for complex mulitiple object, choose the former
    version=version_list[opt.version_number]   
    try:     
        images = RPG(user_prompt=prompt,
                    diffusion_model=opt.model_name,
                    version=version,
                    split_ratio=None,
                    use_llm=use_llm,
                    key=opt.api_key,
                    use_gpt=True, 
                    use_local=opt.use_local,
                    llm_path=opt.llm_path,
                    use_base=opt.use_base,
                    base_ratio=opt.base_ratio,
                    base_prompt=opt.base_prompt,
                    batch_size=opt.batch_size,
                    seed=seed,
                    demo=opt.demo,
                    use_personalized=False,
                    cfg=opt.cfg,
                    steps=num_inference_steps,
                    height=height,
                    width=width,
                    model_name=model_name,
                    style=style,
                    negative_prompt_value=negative_prompt,
                    timestamp=timestamp,
                    ) 
        cmd = f"python {myFilename} --demo --user_prompt {timestamp}"
        result = os.system(cmd)
    
        predict_cost = int(time.time() - startTime)
        tempDir = f'{WORKING_ROOT}/{timestamp}'
        image_paths = []
        for filename in os.listdir(tempDir):
            if filename.endswith('.png'):
                image_paths.append(f'{tempDir}/{filename}')
        if len(image_paths) > 0:
            images = None
        else:
            image_paths = None
            predict_cost = f"{predict_cost}___No images"
    except Exception as e:
        image_paths = None
        predict_cost = str(e)

    torch.cuda.empty_cache()
    return image_paths, seed, predict_cost

def run_rpg(regional_settings):
            user_prompt=regional_settings['Regional_prompt']
            model_name=regional_settings['model']
            version=None
            use_llm=True
            api_key=None
            use_gpt=False
            use_local=False
            llm_path=None
            activate=True
            use_base=regional_settings['Use_Base']
            base_ratio=regional_settings['Base_ratio']
            base_prompt=None
            batch_size=1
            seed=regional_settings['seed']
            cfg=regional_settings['CFG']
            steps=regional_settings['steps']
            height=regional_settings['height']
            width=regional_settings['width']

            images=RPG(user_prompt=user_prompt,
                diffusion_model=model_name,
                version=version,
                split_ratio=regional_settings['Split_ratio'],
                use_llm=use_llm,
                key=api_key,
                use_gpt=use_gpt,
                use_local=use_local,
                llm_path=llm_path,
                use_base=use_base,
                base_ratio=base_ratio,
                base_prompt=base_prompt,
                batch_size=1,
                seed=seed,
                demo=demo,
                use_personalized=False,
                cfg=cfg,
                steps=steps,
                height=height,
                width=width,
                )   
            return images    


def main_gradio(opt):
    examples = [
        "A blonde hair girl with black suit and white skirt",
        "From left to right, an acient Chinese city in spring, summer, autumn and winter in four different regions",
        "A beautiful landscape with a river in the middle the left of the river is in the evening and in the winter with a big iceberg and a small village while some people are skating on the river and some people are skiing, the right of the river is in the summer with a volcano in the morning and a small village while some people are playing.",
        "A twin-tail girl wearing a brwon cowboy hat and white shirt printed with apples, and blue denim jeans with knee boots,full body shot.",
        "A couple, the beautiful girl on the left, silver hair, braided ponytail, happy, dynamic, energetic, peaceful, the handsome young man on the right detailed gorgeous face, grin, blonde hair, enchanting",
        "Two beautiful Chinese girls wearing cheongsams are drinking tea in the tea room, and a Chinese Landscape Painting is hanging on the wall, the girl on the left is black ponytail in red cheongsam, the girl on the right is white ponytail in orange cheongsam",
        "Five watermelons arranged in X shape on a wooden table, with the one in the middle being cut, realistic style, top down view. ",
        "From left to right ,bathed in soft morning light,a cozy nook features a steaming Starbucks latte on a rustic table beside an elegant vase of blooming roses,while a plush ragdoll cat purrs contentedly nearby,its eyes half-closed in blissful serenity.",
        "A green twintail girl in orange dress is sitting on the sofa while a messy desk under a big window on the left, a lively aquarium is on the top right of the sofa",
    ]

    css = '''
    .gradio-container{max-width: 800px !important}
    h1{text-align:center}
    '''

    with gr.Blocks(css=css, title='RPG-DiffusionMaster') as demo:
        with gr.Group():
            with gr.Row():
                prompt = gr.Text(
                    label="prompt",
                    show_label=False,
                    max_lines=6,
                    lines=4,
                    placeholder="Enter the content you want to paint",
                    container=False,
                )
                run_button = gr.Button("run", scale=0)
            with gr.Row():
                use_llm = gr.Checkbox(value=True, label="Is LLM used?")

            with gr.Row():
                with gr.Column(scale=1, min_width=200):
                    model_style_name = gr.Dropdown(label="Model type:", choices=MODEL_STYLE_NAMES, value=DEFAULT_MODEL_STYLE_NAME) 
                with gr.Column(scale=1, min_width=200):
                    style = gr.Dropdown(label="Image style:", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME)                  
            result = gr.Gallery(label="Result", 
                        show_label=False)
            predict_cost = gr.Textbox(label="generate time(s):")

        with gr.Accordion("Advanced Options", open=False):
            with gr.Row():
                negative_prompt_value = "nsfw, lowres, bad anatomy, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, asymmetry, illustration, 3d, 2d, painting, cartoons, sketch"
                with gr.Column(scale=1, min_width=200):
                    use_negative_prompt = gr.Checkbox(label="use negative prompt", value=True)
                with gr.Column(scale=3, min_width=200):
                    negative_prompt = gr.Text(
                        label="negative prompt",
                        max_lines=3,
                        placeholder="Enter negative prompt",
                        visible=True,
                        value=negative_prompt_value,
                    )
            seed = gr.Slider(
                label="seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            randomize_seed = gr.Checkbox(label="randomize seed", value=True)
            with gr.Row(visible=True):
                width = gr.Slider(
                    label="width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=opt.width, 
                )
                height = gr.Slider(
                    label="height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=opt.height, 
                )
        
            with gr.Row():
                num_inference_steps = gr.Slider(
                    label="inference steps", minimum=1, maximum=25, value=20, step=1,
                    visible=True
                )            
            with gr.Row():
                num_images_per_prompt = gr.Slider(
                    label="images per prompt", minimum=1, maximum=4, value=1, step=1,
                    visible=False,
                )
        
        gr.Examples(
            label="examples",
            examples=examples,
            inputs=prompt,
            outputs=[result, seed, predict_cost],
            fn=None, #generate,
            examples_per_page=12,
            cache_examples=False, #CACHE_EXAMPLES,
        )

        use_negative_prompt.change(
            fn=lambda x: gr.update(visible=x),
            inputs=use_negative_prompt,
            outputs=negative_prompt,
            api_name=False,
        )
        run_button.click(fn=generate, 
                inputs=[
                    prompt,
                    negative_prompt,
                    use_negative_prompt,
                    use_llm,
                    model_style_name,
                    style,
                    seed,
                    width,
                    height,
                    randomize_seed,
                    num_inference_steps,
                    num_images_per_prompt,
                ],
                outputs=[result, seed, predict_cost]
            ) 

    demo.queue(max_size=20)
    port = 13533
    # print(f"Start a gradio server: http://0.0.0.0:{port}")
    demo.launch(server_name='0.0.0.0', server_port=port)

opt = None
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_prompt', type=str,help='input user prompt')
    parser.add_argument('--model_name', type=str,default='albedobaseXL_v20.safetensors',help='the name of the ckpt, in the folder of models/Stable-diffusion')
    parser.add_argument('--version_number',type=int,default=0, help='the version of the prompt, multi-attribute or complex-object')
    parser.add_argument('--use_gpt',action='store_true',help='whether to use GPT-4 or Gemini-Pro')
    parser.add_argument('--api_key',default=None,type=str,help='the api key of GPT-4 or Gemini-Pro')
    parser.add_argument('--use_local',action='store_true',help='whether to use local LLM')
    parser.add_argument('--demo', default=False,action='store_true', help='whether to use demo')
    parser.add_argument('--llm_path',default=None,type=str,help='the path of local LLM')
    parser.add_argument('--activate',default=True,type=bool,help='whether to activate regional diffusion')
    parser.add_argument('--use_base',action='store_true',help='whether to use base prompt')
    parser.add_argument('--base_ratio',default=0.3,type=float,help='the weight of base prompt')
    parser.add_argument('--base_prompt',default=None,type=str,help='the base prompt')
    parser.add_argument('--batch_size',default=1,type=int,help='the batch size of txt2img')
    parser.add_argument('--seed',default=1234,type=int,help='the seed of txt2img')
    parser.add_argument('--cfg',default=5,type=float,help='context-free guidance scale')
    parser.add_argument('--steps',default=20,type=int,help='the steps of txt2img')
    parser.add_argument('--height',default=1024,type=int,help='the height of the generated image')
    parser.add_argument('--width',default=1024,type=int,help='the width of the generated image')

    opt = parser.parse_args()

    user_prompt=opt.user_prompt #This is what we need in all the situations except for demo
    model_name=opt.model_name #This is what we need in all the situations except for demo
    activate=opt.activate #If you want to try direct generation, set False
    demo=opt.demo  # Use our preset params to generate images
    use_gpt=opt.use_gpt #If you want to use GPT-4 or Gemini-Pro, set True, strongly recommended
    use_local=opt.use_local #If you want to use local LLM, set True, but we don't recommend this
    version_list=['multi-attribute','complex-object'] # Choose different in-context examples for generation, for multi-people attribute, choose the first one, for complex mulitiple object, choose the former
    version=version_list[opt.version_number]
    api_key=opt.api_key # If you use GPT-4 or Gemini-Pro, you need to input your api key
    llm_path=opt.llm_path # If you use local LLM, you need to input the path of your local LLM
    use_base=opt.use_base # If you want to use base prompt, set True
    base_ratio=opt.base_ratio # The weight of base prompt
    base_prompt=opt.base_prompt # The base prompt, if you don't input this, we will use the user prompt as the base prompt
    batch_size=opt.batch_size # The batch size of txt2img
    seed=opt.seed # The seed of txt2img
    cfg=opt.cfg # The context-free guidance scale
    steps=opt.steps # The steps of txt2img
    height=opt.height
    width=opt.width

    # if opt.user_prompt == 'GRADIO':
    if not demo:
        main_gradio(opt)
    elif opt.user_prompt != '':
        tempDir = f'{WORKING_ROOT}/{opt.user_prompt}'
        with open(f"{tempDir}/regional_setting.json", "r") as file:
            regional_settings = json.load(file)[0]
        load_model(model_name = regional_settings["model"])
        images = run_rpg(regional_settings)
        for i in range(len(images)):
            file_name = f"{tempDir}/image_{i}.png"
            images[i].save(f"{file_name}")

    