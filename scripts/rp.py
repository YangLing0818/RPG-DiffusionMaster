import inspect
import os.path
from importlib import reload
import launch
from pprint import pprint
import gradio as gr
import numpy as np
from PIL import Image
import modules.ui
import modules # SBM Apparently, basedir only works when accessed directly.
from modules import paths, scripts, shared, extra_networks, prompt_parser
from modules.processing import Processed
from modules.script_callbacks import (on_ui_settings,
                                      CFGDenoisedParams, CFGDenoiserParams, on_cfg_denoised, on_cfg_denoiser)
import scripts.attention
import scripts.latent
import scripts.regions
try:
    reload(scripts.regions) # update without restarting web-ui.bat
    reload(scripts.attention)
    reload(scripts.latent)
except:
    pass
import json  # Presets.
from json.decoder import JSONDecodeError
from scripts.attention import (TOKENS, hook_forwards, reset_pmasks, savepmasks)
from scripts.latent import (denoised_callback_s, denoiser_callback_s, lora_namer, setuploras, unloadlorafowards)
from scripts.regions import (MAXCOLREG, IDIM, KEYBRK, KEYBASE, KEYCOMM, KEYPROMPT, ALLKEYS, ALLALLKEYS,
                             create_canvas, draw_region, #detect_mask, detect_polygons,  
                             draw_image, save_mask, load_mask, changecs,
                             floatdef, inpaintmaskdealer, makeimgtmp, matrixdealer)

FLJSON = "regional_prompter_presets.json"
OPTAND = "disable convert 'AND' to 'BREAK'"
OPTUSEL = "Use LoHa or other"
# Modules.basedir points to extension's dir. script_path or scripts.basedir points to root.
PTPRESET = modules.scripts.basedir()
PTPRESETALT = os.path.join(paths.script_path, "scripts")



def lange(l):
    return range(len(l))

orig_batch_cond_uncond = shared.opts.batch_cond_uncond if hasattr(shared.opts,"batch_cond_uncond") else shared.batch_cond_uncond

PRESETSDEF =[
    ["Vertical-3", "Vertical",'1,1,1',"",False,False,False,"Attention",False,"0","0"],
    ["Horizontal-3", "Horizontal",'1,1,1',"",False,False,False,"Attention",False,"0","0"],
    ["Horizontal-7", "Horizontal",'1,1,1,1,1,1,1',"0.2",True,False,False,"Attention",False,"0","0"],
    ["Twod-2-1", "Horizontal",'1,2,3;1,1',"0.2",False,False,False,"Attention",False,"0","0"],
]

ATTNSCALE = 8 # Initial image compression in attention layers.

fhurl = lambda url, label: r"""<a href="{}">{}</a>""".format(url, label)
GUIDEURL = r"https://github.com/hako-mikan/sd-webui-regional-prompter"
MATRIXURL = GUIDEURL + r"#2d-region-assignment"
MASKURL = GUIDEURL + r"#mask-regions-aka-inpaint-experimental-function"
PROMPTURL = GUIDEURL + r"/blob/main/prompt_en.md"
PROMPTURL2 = GUIDEURL + r"/blob/main/prompt_ja.md"


def ui_tab(mode, submode):
    """Structures components for mode tab.
    
    Semi harcoded but it's clearer this way.
    """
    vret = None
    if mode == "Matrix":
        with gr.Row():
            mguide = gr.HTML(value = fhurl(MATRIXURL, "Matrix mode guide")) 
        with gr.Row():
            mmode = gr.Radio(label="Main Splitting", choices=submode, value="Columns", type="value", interactive=True,elem_id="RP_main_splitting")
            ratios = gr.Textbox(label="Divide Ratio",lines=1,value="1,1",interactive=True,elem_id="RP_divide_ratio",visible=True)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    twid = gr.Slider(label="Width", minimum=64, maximum=2048, value=512, step=8,elem_id="RP_matrix_width")
                    thei = gr.Slider(label="Height", minimum=64, maximum=2048, value=512, step=8,elem_id="RP_matrix_height")
                maketemp = gr.Button(value="visualize and make template")

                template = gr.Textbox(label="template",interactive=True,visible=True,elem_id="RP_matrix_template")
                flipper = gr.Checkbox(label = 'flip "," and ";"', value = False,elem_id="RP_matrix_flip")
                overlay = gr.Slider(label="Overlay Ratio", minimum=0, maximum=1, step=0.1, value=0.5,elem_id="RP_matrix_overlay")

            with gr.Column():
                areasimg = gr.Image(type="pil", show_label  = False, height=256, width=256,source = "upload", interactive=True)
        # Need to add maketemp function based on base / common checks.
        vret = [mmode, ratios, maketemp, template, areasimg, flipper, thei, twid, overlay]
    elif mode == "Mask":
        with gr.Row():
            xguide = gr.HTML(value = fhurl(MASKURL, "Inpaint+ mode guide"))
        with gr.Row(): # Creep: Placeholder, should probably make this invisible.
            xmode = gr.Radio(label="Mask mode", choices=submode, value="Mask", type="value", interactive=True,elem_id="RP_mask_mode")
        with gr.Row(): # CREEP: Css magic to make the canvas bigger? I think it's in style.css: #img2maskimg -> height.
            polymask = gr.Image(label = "Do not upload here until bugfix",elem_id="polymask",
                                source = "upload", mirror_webcam = False, type = "numpy", tool = "sketch")#.style(height=480)
        with gr.Row():
            with gr.Column():
                num = gr.Slider(label="Region", minimum=-1, maximum=MAXCOLREG, step=1, value=1,elem_id="RP_mask_region")
                canvas_width = gr.Slider(label="Inpaint+ Width", minimum=64, maximum=2048, value=512, step=8,elem_id="RP_mask_width")
                canvas_height = gr.Slider(label="Inpaint+ Height", minimum=64, maximum=2048, value=512, step=8,elem_id="RP_mask_height")
                btn = gr.Button(value = "Draw region + show mask")
                # btn2 = gr.Button(value = "Display mask") # Not needed.
                cbtn = gr.Button(value="Create mask area")
            with gr.Column():
                showmask = gr.Image(label = "Mask", shape=(IDIM, IDIM))
                # CONT: Awaiting fix for https://github.com/gradio-app/gradio/issues/4088.
                uploadmask = gr.Image(label="Upload mask here cus gradio",source = "upload", type = "numpy")
        # btn.click(detect_polygons, inputs = [polymask,num], outputs = [polymask,num])
        btn.click(draw_region, inputs = [polymask, num], outputs = [polymask, num, showmask])
        # btn2.click(detect_mask, inputs = [polymask,num], outputs = [showmask])
        cbtn.click(fn=create_canvas, inputs=[canvas_height, canvas_width], outputs=[polymask])
        uploadmask.upload(fn = draw_image, inputs = [uploadmask], outputs = [polymask, uploadmask, showmask])
        
        vret = [xmode, polymask, num, canvas_width, canvas_height, btn, cbtn, showmask, uploadmask]
    elif mode == "Prompt":
        with gr.Row():
            pguide = gr.HTML(value = fhurl(PROMPTURL, "Prompt mode guide"))
            pguide2 = gr.HTML(value = fhurl(PROMPTURL2, "Extended prompt guide (jp)"))
        with gr.Row():
            pmode = gr.Radio(label="Prompt mode", choices=submode, value="Prompt", type="value", interactive=True, elem_id="RP_prompt_mode")
            threshold = gr.Textbox(label = "threshold", value = 0.4, interactive=True, elem_id="RP_prompt_threshold")
        
        vret = [pmode, threshold]

    return vret
            
# modes, submodes. Order must be maintained so dict is inadequate. Must have submode for component consistency.
RPMODES = [
("Matrix", ("Columns","Rows","Random")),
("Mask", ("Mask",)),
("Prompt", ("Prompt", "Prompt-Ex")),
]
fgrprop = lambda x: {"label": x, "id": "t" + x, "elem_id": "RP_" + x}

def mode2tabs(mode):
    """Converts mode (in preset) to gradio tab + submodes.
    
    I dunno if it's possible to nest components or make them optional (probably not),
    so this is the best we can do.
    """
    vret = ["Nope"] + [None] * len(RPMODES)
    for (i,(k,v)) in enumerate(RPMODES):
        if mode in v:
            vret[0] = k
            vret[i + 1] = mode
    return vret
    
def tabs2mode(tab, *submode):
    """Converts ui tab + submode list to a single value mode.
    
    Picks current submode based on tab, nothing clever. Submodes must be unique.
    """
    for (i,(k,_)) in enumerate(RPMODES):
        if tab == k:
            return submode[i]
    return "Nope"
    
def expand_components(l):
    """Converts json preset to component format.
    
    Assumes mode is the first value in list.
    """
    l = list(l) # Tuples cannot be altered.
    tabs = mode2tabs(l[0])
    return tabs + l[1:]

def compress_components(l):
    """Converts component values to preset format.
    
    Assumes tab + submodes are the first values in list.
    """
    l = list(l)
    mode = tabs2mode(*l[:len(RPMODES) + 1])
    return [mode] + l[len(RPMODES) + 1:]
    
class Script(modules.scripts.Script):
    def __init__(self,active = False,mode = "Matrix",calc = "Attention",h = 0, w =0, debug = False, debug2 = False, usebase = False, 
    usecom = False, usencom = False, batch = 1,isxl = False, lstop=0, lstop_hr=0, diff = None):
        self.active = active
        if mode == "Columns": mode = "Horizontal"
        if mode == "Rows": mode = "Vertical"
        self.mode = mode
        self.calc = calc
        self.h = h
        self.w = w
        self.debug = debug
        self.debug2 = debug2
        self.usebase = usebase
        self.usecom = usecom
        self.usencom = usencom
        self.batch_size = batch
        self.isxl = isxl

        self.aratios = []
        self.bratios = []
        self.divide = 0
        self.count = 0
        self.eq = True
        self.pn = True
        self.hr = False
        self.hr_scale = 0
        self.hr_w = 0
        self.hr_h = 0
        self.in_hr = False
        self.xsize = 0
        self.imgcount = 0
        # for latent mode
        self.filters = []
        self.lora_applied = False
        self.lstop = int(lstop)
        self.lstop_hr = int(lstop_hr)
        # for inpaintmask
        self.regmasks = None
        self.regbase = None
        #for prompt region
        self.pe = []
        self.step = 0
        
        #for Differential
        self.diff = diff
        self.rps = None

        #script communicator
        self.hooked = False
        self.condi = 0

        self.used_prompt = ""
        self.logprops = ["active","mode","usebase","usecom","usencom","batch_size","isxl","h","w","aratios",
                        "divide","count","eq","pn","hr","pe","step","diff","used_prompt"]
        self.log = {}

    def logger(self):
        for prop in self.logprops:
            print(f"{prop} = {getattr(self,prop,None)}")
        for key in self.log.keys():
            print(f"{key} = {self.log[key]}")

    def title(self):
        return "Regional Prompter"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    infotext_fields = None
    paste_field_names = []

    def ui(self, is_img2img):
        filepath = os.path.join(PTPRESET, FLJSON)

        presets = []

        presets = loadpresets(filepath)
        presets = LPRESET.update(presets)

        with gr.Accordion("Regional Prompter", open=False, elem_id="RP_main"):
            with gr.Row():
                active = gr.Checkbox(value=False, label="Active",interactive=True,elem_id="RP_active")
                urlguide = gr.HTML(value = fhurl(GUIDEURL, "Usage guide"))
            with gr.Row():
                # mode = gr.Radio(label="Divide mode", choices=["Horizontal", "Vertical","Mask","Prompt","Prompt-Ex"], value="Horizontal",  type="value", interactive=True)
                calcmode = gr.Radio(label="Generation mode", choices=["Attention", "Latent"], value="Attention",  type="value", interactive=True, elem_id="RP_generation_mode",)
            with gr.Row(visible=True):
                # ratios = gr.Textbox(label="Divide Ratio",lines=1,value="1,1",interactive=True,elem_id="RP_divide_ratio",visible=True)
                baseratios = gr.Textbox(label="Base Ratio", lines=1,value="0.2",interactive=True,  elem_id="RP_base_ratio", visible=True)
            with gr.Row():
                usebase = gr.Checkbox(value=False, label="Use base prompt",interactive=True, elem_id="RP_usebase")
                usecom = gr.Checkbox(value=False, label="Use common prompt",interactive=True,elem_id="RP_usecommon")
                usencom = gr.Checkbox(value=False, label="Use common negative prompt",interactive=True,elem_id="RP_usecommon_negative")
            
            # Tabbed modes.
            with gr.Tabs(elem_id="RP_mode") as tabs:
                rp_selected_tab = gr.State("Matrix") # State component to document current tab for gen.
                # ltabs = []
                ltabp = []
                for (i, (md,smd)) in enumerate(RPMODES):
                    with gr.TabItem(**fgrprop(md)) as tab: # Tabs with a formatted id.
                        # ltabs.append(tab)
                        ltabp.append(ui_tab(md, smd))
                    # Tab switch tags state component.
                    tab.select(fn = lambda tabnum = i: RPMODES[tabnum][0], inputs=[], outputs=[rp_selected_tab])

            # Hardcode expansion back to components for any specific events.
            (mmode, ratios, maketemp, template, areasimg, flipper, thei, twid, overlay) = ltabp[0]
            (xmode, polymask, num, canvas_width, canvas_height, btn, cbtn, showmask, uploadmask) = ltabp[1]
            (pmode, threshold) = ltabp[2]
            
            with gr.Accordion("Presets",open = False):
                with gr.Row():
                    availablepresets = gr.Dropdown(label="Presets", choices=presets, type="index")
                    applypresets = gr.Button(value="Apply Presets",variant='primary',elem_id="RP_applysetting")
                with gr.Row():
                    presetname = gr.Textbox(label="Preset Name",lines=1,value="",interactive=True,elem_id="RP_preset_name",visible=True)
                    savesets = gr.Button(value="Save to Presets",variant='primary',elem_id="RP_savesetting")
            with gr.Row():
                lstop = gr.Textbox(label="LoRA stop step",value="0",interactive=True,elem_id="RP_ne_tenc_ratio",visible=True)
                lstop_hr = gr.Textbox(label="LoRA Hires stop step",value="0",interactive=True,elem_id="RP_ne_unet_ratio",visible=True)
                lnter = gr.Textbox(label="LoRA in negative textencoder",value="0",interactive=True,elem_id="RP_ne_tenc_ratio_negative",visible=True)
                lnur = gr.Textbox(label="LoRA in negative U-net",value="0",interactive=True,elem_id="RP_ne_unet_ratio_negative",visible=True)
            with gr.Row():
                options = gr.CheckboxGroup(value=False, label="Options",choices=[OPTAND, OPTUSEL, "debug", "debug2"], interactive=True, elem_id="RP_options")
            mode = gr.Textbox(value = "Matrix",visible = False, elem_id="RP_divide_mode")

            dummy_img = gr.Image(type="pil", show_label  = False, height=256, width=256,source = "upload", interactive=True, visible = False)

            dummy_false = gr.Checkbox(value=False, visible=False)

            areasimg.upload(fn=lambda x: x,inputs=[areasimg],outputs = [dummy_img])
            areasimg.clear(fn=lambda x: None,outputs = [dummy_img])

            def changetabs(mode):
                modes = ["Matrix", "Mask", "Prompt"]
                if mode not in modes: mode = "Matrix"
                return gr.Tabs.update(selected="t"+mode)

            mode.change(fn = changetabs,inputs=[mode],outputs=[tabs])
            settings = [rp_selected_tab, mmode, xmode, pmode, ratios, baseratios, usebase, usecom, usencom, calcmode, options, lnter, lnur, threshold, polymask, lstop, lstop_hr, flipper]
        
        self.infotext_fields = [
                (active, "RP Active"),
                # (mode, "RP Divide mode"),
                (mode, "RP Divide mode"),
                (mmode, "RP Matrix submode"),
                (xmode, "RP Mask submode"),
                (pmode, "RP Prompt submode"),
                (calcmode, "RP Calc Mode"),
                (ratios, "RP Ratios"),
                (baseratios, "RP Base Ratios"),
                (usebase, "RP Use Base"),
                (usecom, "RP Use Common"),
                (usencom, "RP Use Ncommon"),
                (options,"RP Options"),
                (lnter,"RP LoRA Neg Te Ratios"),
                (lnur,"RP LoRA Neg U Ratios"),
                (threshold,"RP threshold"),
                (lstop,"RP LoRA Stop Step"),
                (lstop_hr,"RP LoRA Hires Stop Step"),
                (flipper, "RP Flip")
        ]

        for _,name in self.infotext_fields:
            self.paste_field_names.append(name)

        def setpreset(select, *settings):
            """Load preset from list.
            
            SBM: The only way I know how to get the old values in gradio,
            is to pass them all as input.
            Tab mode converts ui to single value.
            """
            # Need to swap all masked images to the source,
            # getting "valueerror: cannot process this value as image".
            # Gradio bug in components.postprocess, most likely.
            settings = [s["image"] if (isinstance(s,dict) and "image" in s) else s for s in settings]
            presets = loadpresets(filepath)
            preset = presets[select]
            preset = loadblob(preset)
            preset = [fmt(preset.get(k, vdef)) for (k,fmt,vdef) in PRESET_KEYS]
            preset = preset[1:] # Remove name.
            preset = expand_components(preset)
            # Change nulls to original value.
            preset = [settings[i] if p is None else p for (i,p) in enumerate(preset)]
            while  len(settings) >= len(preset):
                    preset.append(0)
            # return [gr.update(value = pr) for pr in preset] # SBM Why update? Shouldn't regular return do the job? 
            if preset[0] == "Vertical":preset[0] = "Rows"
            if preset[0] == "Horizontal":preset[0] = "Columns"
            return preset

        maketemp.click(fn=makeimgtmp, inputs =[ratios,mmode,usecom,usebase,flipper,thei,twid,dummy_img,overlay],outputs = [areasimg,template])
        applypresets.click(fn=setpreset, inputs = [availablepresets, *settings], outputs=settings)
        savesets.click(fn=savepresets, inputs = [presetname,*settings],outputs=availablepresets)
        
        return [active, dummy_false, rp_selected_tab, mmode, xmode, pmode, ratios, baseratios,
                usebase, usecom, usencom, calcmode, options, lnter, lnur, threshold, polymask, lstop, lstop_hr, flipper]

    def process(self, p, active, a_debug , rp_selected_tab, mmode, xmode, pmode, aratios, bratios,
                usebase, usecom, usencom, calcmode, options, lnter, lnur, threshold, polymask, lstop, lstop_hr, flipper):
        if type(options) is bool:
            options = ["disable convert 'AND' to 'BREAK'"] if options else []
        elif type(options) is str:
            options = options.split(",")

        if a_debug == True:
            options.append("debug")

        debug = "debug" in options
        debug2 = "debug2" in options
        self.slowlora = OPTUSEL in options

        if type(polymask) == str:
            try:
                polymask,_,_ = draw_image(np.array(Image.open(polymask)))
            except:
                pass
        
        if rp_selected_tab == "Nope": rp_selected_tab = "Matrix"

        if debug: pprint([active, debug, rp_selected_tab, mmode, xmode, pmode, aratios, bratios,
                usebase, usecom, usencom, calcmode, options, lnter, lnur, threshold, polymask, lstop, lstop_hr, flipper])

        tprompt = p.prompt[0] if type(p.prompt) == list else p.prompt

        if hasattr(p,"rps_diff"):
            if p.rps_diff:
                active = True
                mmode = "Prompt"
                xmode = "Prompt-Ex"
                diff = p.rps_diff
                if hasattr(p, "all_prompts_rps"):
                    p.all_prompts = p.all_prompts_rps
                if hasattr(p,"threshold"):
                    if p.threshold is not None:threshold = str(p.threshold)
        else:
            diff = False

        if not any(key in tprompt for key in ALLALLKEYS) or not active:
            return unloader(self,p)

        p.extra_generation_params.update({
            "RP Active":active,
            "RP Divide mode": rp_selected_tab,
            "RP Matrix submode": mmode,
            "RP Mask submode": xmode,
            "RP Prompt submode": pmode,
            "RP Calc Mode":calcmode,
            "RP Ratios": aratios,
            "RP Base Ratios": bratios,
            "RP Use Base":usebase,
            "RP Use Common":usecom,
            "RP Use Ncommon": usencom,
            "RP Options" : options,
            "RP LoRA Neg Te Ratios": lnter,
            "RP LoRA Neg U Ratios": lnur,
            "RP threshold": threshold,
            "RP LoRA Stop Step":lstop,
            "RP LoRA Hires Stop Step":lstop_hr,
            "RP Flip": flipper,
        })

        savepresets("lastrun",rp_selected_tab, mmode, xmode, pmode, aratios,bratios,
                     usebase, usecom, usencom, calcmode, options, lnter, lnur, threshold, polymask,lstop, lstop_hr, flipper)

        if flipper:aratios = changecs(aratios)

        self.__init__(active, tabs2mode(rp_selected_tab, mmode, xmode, pmode) ,calcmode ,p.height, p.width, debug, debug2,
        usebase, usecom, usencom, p.batch_size, hasattr(shared.sd_model,"conditioner"),lstop, lstop_hr, diff = diff)

        self.all_prompts = p.all_prompts.copy()
        self.all_negative_prompts = p.all_negative_prompts.copy()

        # SBM ddim / plms detection.
        self.isvanilla = p.sampler_name in ["DDIM", "PLMS", "UniPC"]

        if self.h % ATTNSCALE != 0 or self.w % ATTNSCALE != 0:
            # Testing shows a round down occurs in model.
            print("Warning: Nonstandard height / width.")
            self.h = self.h - self.h % ATTNSCALE
            self.w = self.w - self.w % ATTNSCALE

        if hasattr(p,"enable_hr"): # Img2img doesn't have it.
            self.hr = p.enable_hr
            self.hr_w = (p.hr_resize_x if p.hr_resize_x > p.width else p.width * p.hr_scale)
            self.hr_h = (p.hr_resize_y if p.hr_resize_y > p.height else p.height * p.hr_scale)
            if self.hr_h % ATTNSCALE != 0 or self.hr_w % ATTNSCALE != 0:
                # Testing shows a round down occurs in model.
                print("Warning: Nonstandard height / width for ulscaled size")
                self.hr_h = self.hr_h - self.hr_h % ATTNSCALE
                self.hr_w = self.hr_w - self.hr_w % ATTNSCALE
    
        loraverchekcer(self)                                                  #check web-ui version
        if OPTAND not in options: allchanger(p, "AND", KEYBRK)                                          #Change AND to BREAK
        if any(x in self.mode for x in ["Ver","Hor"]):
            keyconverter(aratios, self.mode, usecom, usebase, p) #convert BREAKs to ADDROMM/ADDCOL/ADDROW
        bckeydealer(self, p)                                                      #detect COMM/BASE keys
        keycounter(self, p)                                                       #count keys and set to self.divide
            
        if "Pro" not in self.mode: # skip region assign in prompt mode
            ##### region mode
            if "Mask" in self.mode:
                keyreplacer(p) #change all keys to BREAK
                inpaintmaskdealer(self, p, bratios, usebase, polymask)

            elif any(x in self.mode for x in ["Ver","Hor","Ran"]):
                matrixdealer(self, p, aratios, bratios, self.mode)
    
            ##### calcmode 
            if "Att" in calcmode:
                self.handle = hook_forwards(self, p.sd_model.model.diffusion_model)
                if hasattr(shared.opts,"batch_cond_uncond"):
                    shared.opts.batch_cond_uncond = orig_batch_cond_uncond
                else:                    
                    shared.batch_cond_uncond = orig_batch_cond_uncond
                unloadlorafowards(p)
            else:
                self.handle = hook_forwards(self, p.sd_model.model.diffusion_model,remove = True)
                setuploras(self)
                # SBM It is vital to use local activation because callback registration is permanent,
                # and there are multiple script instances (txt2img / img2img). 

        elif "Pro" in self.mode: #Prompt mode use both calcmode
            self.ex = "Ex" in self.mode
            if not usebase : bratios = "0"
            self.handle = hook_forwards(self, p.sd_model.model.diffusion_model)
            denoiserdealer(self)

        neighbor(self,p)                                                    #detect other extention
        keyreplacer(p)                                                      #replace all keys to BREAK
        blankdealer(self, p)                                               #add "_" if prompt of last region is blank
        commondealer(p, self.usecom, self.usencom)          #add commom prompt to all region
        if "La" in self.calc: allchanger(p, KEYBRK,"AND")      #replace BREAK to AND in Latent mode
        if tokendealer(self, p): return unloader(self,p)          #count tokens and calcrate target tokens
        thresholddealer(self, p, threshold)                          #set threshold
        
        bratioprompt(self, bratios)
        if not self.diff: hrdealer(p)

        print(f"Regional Prompter Active, Pos tokens : {self.ppt}, Neg tokens : {self.pnt}")
        self.used_prompt = p.all_prompts[0]

        if debug : debugall(self)

    def before_process_batch(self, p, *args, **kwargs):
        if self.active:
            self.current_prompts = kwargs["prompts"].copy()
            p.disable_extra_networks = False

    def before_hr(self, p, active, _, rp_selected_tab, mmode, xmode, pmode, aratios, bratios,
                      usebase, usecom, usencom, calcmode,nchangeand, lnter, lnur, threshold, polymask,lstop, lstop_hr, flipper):
        if self.active:
            self.in_hr = True
            if "La" in self.calc:
                lora_namer(self, p, lnter, lnur)
                self.log["before_hr"] = "passed"
                try:
                    import lora
                    self.log["before_hr_loralist"] = [x.name for x in lora.loaded_loras]
                except:
                    pass

    def process_batch(self, p, active, _, rp_selected_tab, mmode, xmode, pmode, aratios, bratios,
                      usebase, usecom, usencom, calcmode,nchangeand, lnter, lnur, threshold, polymask,lstop, lstop_hr,flipper,**kwargs):
        # print(kwargs["prompts"])

        if self.active:
            resetpcache(p)
            self.in_hr = False
            self.xsize = 0
            # SBM Before_process_batch was added in feb-mar, adding fallback.
            if not hasattr(self,"current_prompts"):
                self.current_prompts = kwargs["prompts"].copy()
            p.all_prompts[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size] = self.all_prompts[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size]
            p.all_negative_prompts[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size] = self.all_negative_prompts[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size]
            if "Pro" in self.mode:
                reset_pmasks(self)
            if "La" in self.calc:
                lora_namer(self, p, lnter, lnur)
                try:
                    import lora
                    self.log["loralist"] = [x.name for x in lora.loaded_loras]
                except:
                    pass

                if self.lora_applied: # SBM Don't override orig twice on batch calls.
                    pass
                else:
                    denoiserdealer(self)
                    self.lora_applied = True
                #escape reload loras in hires-fix

    def postprocess(self, p, processed, *args):
        if self.active : 
            with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
                processedx = Processed(p, [], p.seed, "")
                file.write(processedx.infotext(p, 0))
        
        if "Pro" in self.mode and not fseti("hidepmask"):
            savepmasks(self, processed)

        if self.debug or self.debug2 : self.logger()

        unloader(self, p)

    def denoiser_callback(self, params: CFGDenoiserParams):
        denoiser_callback_s(self, params)

    def denoised_callback(self, params: CFGDenoisedParams):
        denoised_callback_s(self, params)


def unloader(self,p):
    if hasattr(self,"handle"):
        #print("unloaded")
        hook_forwards(self, p.sd_model.model.diffusion_model, remove=True)
        del self.handle

    self.__init__()
    
    if hasattr(shared.opts,"batch_cond_uncond"):
        shared.opts.batch_cond_uncond = orig_batch_cond_uncond
    else:                    
        shared.batch_cond_uncond = orig_batch_cond_uncond

    unloadlorafowards(p)

def denoiserdealer(self):
    if self.calc =="Latent": # prompt mode use only denoiser callbacks
        if not hasattr(self,"dd_callbacks"):
            self.dd_callbacks = on_cfg_denoised(self.denoised_callback)
        if hasattr(shared.opts,"batch_cond_uncond"):
            shared.opts.batch_cond_uncond = False
        else:                    
            shared.batch_cond_uncond = False

    if not hasattr(self,"dr_callbacks"):
        self.dr_callbacks = on_cfg_denoiser(self.denoiser_callback)

    if self.diff:
        if not hasattr(self,"dd_callbacks"):
            self.dd_callbacks = on_cfg_denoised(self.denoised_callback)


############################################################
##### prompts, tokens
def blankdealer(self, p):
    seps = "AND" if "La" in self.calc else KEYBRK
    all_prompts=[]
    for prompt in p.all_prompts:
        regions = prompt.split(seps)
        if regions[-1].strip() in ["",","]:
            prompt = prompt + " _"
        all_prompts.append(prompt)
    p.all_prompts = all_prompts

def commondealer(p, usecom, usencom):
    all_prompts = []
    all_negative_prompts = []

    def comadder(prompt):
        ppl = prompt.split(KEYBRK)
        for i in range(len(ppl)):
            if i == 0:
                continue
            ppl[i] = ppl[0] + ", " + ppl[i]
        ppl = ppl[1:]
        prompt = f"{KEYBRK} ".join(ppl)
        return prompt

    if usecom:
        for pr in p.all_prompts:
            all_prompts.append(comadder(pr))
        p.all_prompts = all_prompts
        p.prompt = all_prompts[0]

    if usencom:
        for pr in p.all_negative_prompts:
            all_negative_prompts.append(comadder(pr))
        p.all_negative_prompts = all_negative_prompts
        p.negative_prompt = all_negative_prompts[0]

def hrdealer(p):
    p.hr_prompt = p.prompt
    p.hr_negative_prompt = p.negative_prompt
    p.all_hr_prompts = p.all_prompts
    p.all_hr_negative_prompts = p.all_negative_prompts

def allchanger(p, a, b):
    p.prompt = p.prompt.replace(a, b)
    for i in lange(p.all_prompts):
        p.all_prompts[i] = p.all_prompts[i].replace(a, b)
    p.negative_prompt = p.negative_prompt.replace(a, b)
    for i in lange(p.all_negative_prompts):
        p.all_negative_prompts[i] = p.all_negative_prompts[i].replace(a, b)

def tokendealer(self, p):
    seps = "AND" if "La" in self.calc else KEYBRK
    self.seps = seps
    text, _ = extra_networks.parse_prompt(p.all_prompts[0]) # SBM From update_token_counter.
    text = prompt_parser.get_learned_conditioning_prompt_schedules([text],p.steps)[0][0][1]
    ppl = text.split(seps)
    ntext, _ = extra_networks.parse_prompt(p.all_negative_prompts[0]) 
    npl = ntext.split(seps)
    eqb = len(ppl) == len(npl)
    targets =[p.split(",")[-1] for p in ppl[1:]]
    pt, nt, ppt, pnt, tt = [], [], [], [], []

    padd = 0
    
    tokenizer = shared.sd_model.conditioner.embedders[0].tokenize_line if self.isxl else shared.sd_model.cond_stage_model.tokenize_line

    for pp in ppl:
        tokens, tokensnum = tokenizer(pp)
        pt.append([padd, tokensnum // TOKENS + 1 + padd])
        ppt.append(tokensnum)
        padd = tokensnum // TOKENS + 1 + padd

    if "Pro" in self.mode:
        for target in targets:
            ptokens, tokensnum = tokenizer(ppl[0])
            ttokens, _ = tokenizer(target)

            i = 1
            tlist = []
            while ttokens[0].tokens[i] != 49407:
                for (j, maintok) in enumerate(ptokens): # SBM Long prompt.
                    if ttokens[0].tokens[i] in maintok.tokens:
                        tlist.append(maintok.tokens.index(ttokens[0].tokens[i]) + 75 * j)
                i += 1
            if tlist != [] : tt.append(tlist)

    paddp = padd
    padd = 0
    for np in npl:
        _, tokensnum = tokenizer(np)
        nt.append([padd, tokensnum // TOKENS + 1 + padd])
        pnt.append(tokensnum)
        padd = tokensnum // TOKENS + 1 + padd

    self.eq = paddp == padd and eqb

    self.pt = pt
    self.nt = nt
    self.pe = tt
    self.ppt = ppt
    self.pnt = pnt

    notarget = "Pro" in self.mode and tt == []
    if notarget:
        print("No target word is detected in Prompt mode")
    return notarget

def thresholddealer(self, p ,threshold):
    if "Pro" in self.mode:
        threshold = threshold.split(",")
        while len(self.pe) >= len(threshold) + 1:
            threshold.append(threshold[0])
        self.th = [floatdef(t, 0.4) for t in threshold] * self.batch_size
        if self.debug :print ("threshold", self.th)

def bratioprompt(self, bratios):
    if not "Pro" in self.mode: return self
    bratios = bratios.split(",")
    bratios = [floatdef(b, 0) for b in bratios]
    while len(self.pe) >= len(bratios) + 1:
        bratios.append(bratios[0])
    self.bratios = bratios

def neighbor(self,p):
    from modules.scripts import scripts_txt2img
    for script in scripts_txt2img.alwayson_scripts:
        if "negpip.py" in script.filename:
            self.negpip = script

    for script in scripts_txt2img.selectable_scripts:
        if "rps.py" in script.filename:
            self.rps = script
            #print(dir(script))
            #script.test1 = "kawattayone?"
            #script.settest1("kawatta?") 

    try:
        args = p.script_args
        multi = ["MultiDiffusion",'Mixture of Diffusers']
        if any(x in args for x in multi):
            for key in multi:
                if key in args:
                    self.nei_multi = [args[args.index(key)+5],args[args.index(key)+6]]
    except:
        pass

#####################################################
##### Presets - Save  and Load Settings

fimgpt = lambda flnm, fext, *dirparts: os.path.join(*dirparts, flnm + fext)

class PresetList():
    """Preset list must be the same object throughout its lifetime, otherwise updates will err.

    See gradio issue #4210 for details.
    """
    def __init__(self):
        self.lpr = []
    
    def update(self, newpr):
        """Replace all values, return the reference.
        
        Will convert dicts to the names only.
        Might be more efficient to add the new names only, but meh.
        """
        if len(newpr) > 0 and isinstance(newpr[0],dict):
            newpr = [pr["name"] for pr in newpr] 
        self.lpr.clear()
        self.lpr.extend(newpr)
        return self.lpr
        
    def get(self):
        return self.lpr

class JsonMask():
    """Mask saved as image with some editing work.
    
    """
    blobdir = "regional_masks"
    ext = ".png"
    
    def __init__(self, img):
        self.img = img
    
    def makepath(self, name):
        pt = fimgpt(name, self.ext, PTPRESET, self.blobdir)
        os.makedirs(os.path.dirname(pt), exist_ok = True)
        return pt
    
    def save(self, name, preset = None):
        """Save image to subdir.
        
        Only saved when in mask mode - Hardcoded, don't have a better idea atm.
        """
        if (preset is None) or (preset[1] == "Mask"): # Check mode.  
            save_mask(self.img, self.makepath(name))
            return name
        return None
    
    def load(self, name, preset = None):
        """Load image from subdir (no editing, that comes later).
        
        Prefer to use the given key, rather than name. SBM CONT: Load / save in dict mode? Debugging needed.
        """
        if name is None or self.img is None:
            return None
        return load_mask(self.makepath(self.img))

LPRESET = PresetList()

fcountbrk = lambda x: x.count(KEYBRK)
fint = lambda x: int(x)

# Json formatters.
fjstr = lambda x: x.strip()
#fjbool = lambda x: (x.upper() == "TRUE" or x.upper() == "T")
fjbool = lambda x: x # Json can store booleans reliably.
fjmask = lambda x: draw_image(x, inddict = False)[0] # Ignore mask reset value. 

# (json_name, value_format, default)
# If default = none then will use current gradio value. 
PRESET_KEYS = [
("name",fjstr,"") , # Name is special, preset's key.
("mode", fjstr, None) ,
("ratios", fjstr, None) ,
("baseratios", fjstr, None) ,
("usebase", fjbool, None) ,
("usecom", fjbool, False) ,
("usencom", fjbool, False) ,
("calcmode", fjstr, "Attention") , # Generation mode.
("nchangeand", fjbool, False) ,
("lnter", fjstr, "0") ,
("lnur", fjstr, "0") ,
("threshold", fjstr, "0") ,
("polymask", fjmask, "") , # Mask has special corrections and logging.
]
# (json_name,blob_class)
# Handles save + lazy load of blob data outside of presets.
BLOB_KEYS = {
"polymask": JsonMask
}

def saveblob(preset):
    """Preset variables saved externally (blob).
    
    Returns modified list containing the refernces instead of data.
    Currently, this includes polymask, which is saved as an image,
    with a filename = preset.
    A blob class should contain a save method which returns the reference. 
    """
    preset = list(preset) # Tuples don't have copy.
    for (i,(vkey,vfun,vdef)) in enumerate(PRESET_KEYS):
        if vkey in BLOB_KEYS:
            # Func should accept raw form and convert it to a class.
            x = BLOB_KEYS[vkey](preset[i])
            # Class should have a save func given identifier, returning an access key.
            x = x.save(preset[0], preset)
            # Update the preset.
            preset[i] = x
    return preset

def loadblob(preset):
    """Load blob presets based on key.
    
    Returns modified list containing the refernces instead of 
    Currently, this includes polymask, which is saved as an image,
    with a filename = preset.
    A blob class should contain a load method which retrieves the data based on reference. 
    """
    for (vkey,vval) in BLOB_KEYS.items():
        # Func should accept refrence form and convert it to a class.
        x = vval(preset.get(vkey))
        # Class should have a load func given identifier, returning data.
        x = x.load(preset["name"], preset)
        # Update the preset.
        preset[vkey] = x
    return preset

def savepresets(*settings):
    # NAME must come first.
    name = settings[0]
    settings = [name] + compress_components(settings[1:])
    settings = saveblob(settings)
    
    # path_root = modules.scripts.basedir()
    # filepath = os.path.join(path_root, "scripts", "regional_prompter_presets.json")
    filepath = os.path.join(PTPRESET, FLJSON)

    try:
        with open(filepath, mode='r', encoding="utf-8") as f:
            # presets = json.loads(json.load(f))
            presets = json.load(f)
            pr = {PRESET_KEYS[i][0]:settings[i] for i,_ in enumerate(PRESET_KEYS)}
            # SBM Ordereddict might be better than list, quick search.
            written = False
            # if name == "lastrun": # SBM We should check the preset is unique in any case.
            for i, preset in enumerate(presets):
                if name == preset["name"]:
                # if "lastrun" in preset["name"]:
                    presets[i] = pr
                    written = True
            if not written:
                presets.append(pr)
        with open(filepath, mode='w', encoding="utf-8") as f:
            # json.dump(json.dumps(presets), f, indent = 2)
            json.dump(presets, f, indent = 2)
    except Exception as e:
        print(e)

    presets = loadpresets(filepath)
    presets = LPRESET.update(presets)
    return gr.update(choices=presets)

def presetfallback():
    """Swaps main json dir to alt if exists, attempts reload.
    
    """
    global PTPRESET
    global PTPRESETALT
    
    if PTPRESETALT is not None:
        print("Unknown preset error, fallback.")
        PTPRESET = PTPRESETALT
        PTPRESETALT = None
        return loadpresets(PTPRESET)
    else: # Already attempted swap.
        print("Presets could not be loaded.") 
        return None

def loadpresets(filepath):
    presets = []
    try:
        with open(filepath, encoding="utf-8") as f:
            # presets = json.loads(json.load(f))
            presets = json.load(f)
            # presets = loadblob(presets) # DO NOT load all blobs - that's the point.
    except OSError as e:
        print("Init / preset error.")
        presets = initpresets(filepath)
    except TypeError:
        print("Corrupted preset file, resetting.")
        presets = initpresets(filepath)
    except JSONDecodeError:
        print("Preset file could not be decoded.")
        presets = initpresets(filepath)
    return presets

def initpresets(filepath):
    lpr = PRESETSDEF
    # if not os.path.isfile(filepath):
    try:
        with open(filepath, mode='w', encoding="utf-8") as f:
            lprj = []
            for pr in lpr:
                plen = min(len(PRESET_KEYS), len(pr)) # Future setting additions ignored.
                prj = {PRESET_KEYS[i][0]:pr[i] for i in range(plen)}
                lprj.append(prj)
            #json.dump(json.dumps(lprj), f, indent = 2)
            json.dump(lprj, f, indent = 2)
            return lprj
    except Exception as e:
        return presetfallback()

#####################################################
##### Global settings

EXTKEY = "regprp"
EXTNAME = "Regional Prompter"
# (id, label, type, extra_parms)
EXTSETS = [
("debug", "(PLACEHOLDER, USE THE ONE IN 2IMG) Enable debug mode", "check", dict()),
("hidepmask", "Hide subprompt masks in prompt mode", "check", dict()),

]
# Dynamically constructed list of default values, because shared doesn't allocate a value automatically.
# (id: def)
DEXTSETV = dict()
fseti = lambda x: shared.opts.data.get(EXTKEY + "_" + x, DEXTSETV[x])

class Setting_Component():
    """Creates gradio components with some standard req values.
    
    All must supply an id (used in code), label, component type. 
    Default value and specific type settings can be overridden. 
    """
    section = (EXTKEY, EXTNAME)
    def __init__(self, cid, clabel, ctyp, vdef = None, **kwargs):
        self.cid = EXTKEY + "_" + cid
        self.clabel = clabel
        self.ctyp = ctyp
        method = getattr(self, self.ctyp)
        method(**kwargs)
        if vdef is not None:
            self.vdef = vdef
        
    def get(self):
        """Get formatted setting.
        
        Input for shared.opts.add_option().
        """
        if self.ctyp == "textb":
            return (self.cid, shared.OptionInfo(self.vdef, self.clabel, section = self.section))
        return (self.cid, shared.OptionInfo(self.vdef, self.clabel,
                                            self.ccomp, self.cparms, section = self.section))
    
    def textb(self, **kwargs):
        """Textbox unusually requires no component.
        
        """
        self.ccomp = gr.Textbox
        self.vdef = ""
        self.cparms = {}
        self.cparms.update(kwargs)
    
    def check(self, **kwargs):
        self.ccomp = gr.Checkbox
        self.vdef = False
        self.cparms = {"interactive": True}
        self.cparms.update(kwargs)
        
    def slider(self, **kwargs):
        self.ccomp = gr.Slider
        self.vdef = 0
        self.cparms = {"minimum": 1, "maximum": 10, "step": 1}
        self.cparms.update(kwargs)

def ext_on_ui_settings():
    for (cid, clabel, ctyp, kwargs) in EXTSETS:
        comp = Setting_Component(cid, clabel, ctyp, **kwargs)
        opt = comp.get()
        shared.opts.add_option(*opt)
        DEXTSETV[cid] = comp.vdef

def debugall(self):
    print(f"mode : {self.mode}\ncalcmode : {self.calc}\nusebase : {self.usebase}")
    print(f"base ratios : {self.bratios}\nusecommon : {self.usecom}\nusenegcom : {self.usencom}")
    print(f"divide : {self.divide}\neq : {self.eq}")
    print(f"tokens : {self.ppt},{self.pnt},{self.pt},{self.nt}")
    print(f"ratios : {self.aratios}\n")
    print(f"prompt : {self.pe}")
    print(f"env : before15:{self.isbefore15},isxl:{self.isxl}")
    print(f"loras{self.log}")

def bckeydealer(self, p):
    '''
    detect COMM/BASE keys and set flags and change to BREAK
    '''
    if KEYCOMM in p.prompt:
        self.usecom = True
    if self.usecom and KEYCOMM not in p.prompt:
        p.prompt = p.prompt.replace(KEYBRK,KEYCOMM,1)
    
    if KEYCOMM in p.negative_prompt:
        self.usencom = True
    if self.usencom and KEYCOMM not in p.negative_prompt:
        p.negative_prompt = p.negative_prompt.replace(KEYBRK,KEYCOMM,1)
    
    if KEYBASE in p.prompt:
        self.usebase = True
    if self.usebase and KEYBASE not in p.prompt:
        p.prompt = p.prompt.replace(KEYBRK,KEYBASE,1)
        
    if KEYPROMPT in p.prompt.upper():
        self.mode = "Prompt"

def keyconverter(aratios,mode,usecom,usebase,p):
    '''convert BREAKS to ADDCOMM/ADDBASE/ADDCOL/ADDROW'''
    keychanger = makeimgtmp(aratios,mode,usecom,usebase,False,512,512, inprocess = True)
    keychanger = keychanger[:-1]
    #print(keychanger,p.prompt)
    for change in keychanger:
        if change == KEYCOMM and KEYCOMM in p.prompt: continue
        if change == KEYBASE and KEYBASE in p.prompt: continue
        p.prompt= p.prompt.replace(KEYBRK,change,1)

def keyreplacer(p):
    '''
    replace all separators to BREAK
    p.all_prompt and p.all_negative_prompt
    '''
    for key in ALLKEYS:
        for i in lange(p.all_prompts):
            p.all_prompts[i]= p.all_prompts[i].replace(key,KEYBRK)
        
        for i in lange(p.all_negative_prompts):
            p.all_negative_prompts[i] = p.all_negative_prompts[i].replace(key,KEYBRK)

def keycounter(self, p):
    pc = sum([p.prompt.count(text) for text in ALLALLKEYS])
    npc = sum([p.negative_prompt.count(text) for text in ALLALLKEYS])
    self.divide = [pc + 1, npc + 1]

def resetpcache(p):
    p.cached_c = [None,None]
    p.cached_uc = [None,None]
    p.cached_hr_c = [None, None]
    p.cached_hr_uc = [None, None]

def loraverchekcer(self):
    try:
        self.ui_version = int(launch.git_tag().replace("v","").replace(".",""))
    except:
        self.ui_version = 100
        
    try:
        import lora
        self.isbefore15 =  "assign_lora_names_to_compvis_modules" in dir(lora)
        self.layer_name = "lora_layer_name" if self.isbefore15 else "network_layer_name"
    except:
        self.isbefore15 = False
        self.layer_name = "lora_layer_name"

def log(prop):
    frame = inspect.currentframe().f_back
    local_vars = frame.f_locals
    var_name = None
    for k, v in local_vars.items():
        if v is prop:
            var_name = k
            break

    if var_name:
        print(f"{var_name} = {prop}")
    else:
        print("Property not found in local scope.")

on_ui_settings(ext_on_ui_settings)
