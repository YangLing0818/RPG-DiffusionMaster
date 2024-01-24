import sys
import logging
import os
logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())
from modules import timer, errors
import torch
startup_timer = timer.Timer()
from modules import extensions
import modules.scripts
from mllm import GPT4,local_llm
import argparse
from template.demo import demo_list
import time
def initialize(model_name=None):
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
 
 
def RPG(user_prompt,diffusion_model,version,split_ratio=None,key=None,use_gpt=True,use_local=False,
        llm_path=None,activate=True,use_base=False,base_ratio=0,base_prompt=None,batch_size=1,seed=1234,demo=False,use_personalized=False,cfg=5,steps=20,height=1024,width=1024):
     # set model
    import modules.txt2img
    # Prompt for regional diffusion
    if demo:
        print('demo')
        regional_prompt=user_prompt
    #TODO: add personalized regional split and regional prompt 
    else:    
        input_prompt=user_prompt
        if use_gpt:
            assert key is not None
            params=GPT4(input_prompt,version,key)
        elif use_local:
            params=local_llm(input_prompt,version,model_path=llm_path)
        
        regional_prompt=params['Regional Prompt']
        split_ratio=params['split ratio']
        if use_base:
            if base_prompt is None:
                regional_prompt= user_prompt+'BREAK\n'+regional_prompt
            else:
                regional_prompt= base_prompt+'BREAK\n'+regional_prompt
        
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
        negative_prompt="",
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
def demo_version(demo_list):
        for i in range(len(demo_list)):
                print('demo_',i)
                demo=demo_list[i]
                user_prompt=demo['Regional_prompt']
                model_name=demo['model']
                version=None
                api_key=None
                use_gpt=False
                use_local=False
                llm_path=None
                activate=True
                use_base=demo['Use_Base']
                base_ratio=demo['Base_ratio']
                base_prompt=None
                batch_size=1
                seed=demo['seed']
                cfg=demo['CFG']
                steps=demo['steps']
                height=demo['height']
                width=demo['width']
                image=RPG(user_prompt=user_prompt,
                diffusion_model=model_name,
                version=version,
                split_ratio=demo['Split_ratio'],
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
                for j in range(len(image)):
                        file_name = f"demo_{i}"
                        image[j].save(f"generated_imgs/demo_imgs/{file_name}.png")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_prompt', type=str,help='input user prompt')
    parser.add_argument('--model_name', type=str,default='albedobaseXL_v20.safetensors',help='the name of the ckpt, in the folder of models/Stable-diffusion')
    parser.add_argument('--version_number',type=int,default=0, help='the version of the prompt, multi-attribute or complex-object')
    parser.add_argument('--api_key',default=None,type=str,help='the api key of GPT-4 or Gemini-Pro')
    parser.add_argument('--use_gpt',action='store_true',help='whether to use GPT-4 or Gemini-Pro')
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
    if demo:
        initialize(model_name='albedobaseXL_v20.safetensors')
        demo_version(demo_list)
    else:
        if use_gpt:
            appendix='gpt4'
        elif use_local:
            appendix='local'
        initialize(model_name=model_name)
        image=RPG(
        user_prompt=user_prompt,
        diffusion_model=model_name,
        version=version,
        split_ratio=None,
        key=api_key,
        use_gpt=use_gpt,
        use_local=use_local,
        llm_path=llm_path,
        use_base=use_base,
        base_ratio=base_ratio,
        base_prompt=base_prompt,
        batch_size=batch_size,
        seed=seed,
        demo=demo,
        use_personalized=False,
        cfg=cfg,
        steps=steps,
        height=height,
        width=width,
        )
        for i in range(len(image)):
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            file_name = f"{appendix}_image_{timestamp}.png"
            image[i].save(f"generated_imgs/{file_name}")
