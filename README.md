## Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs
<div align="left">
  <a href="https://arxiv.org/abs/2401.11708"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:RPG&color=red&logo=arxiv"></a> &ensp;
</div>

This repository contains the official implementation of our [RPG](https://arxiv.org/abs/2401.11708).

> [**Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs**](https://arxiv.org/abs/2401.11708)   
> [Ling Yang](https://yangling0818.github.io/), 
> [Zhaochen Yu](https://github.com/BitCodingWalkin), 
> [Chenlin Meng](https://cs.stanford.edu/~chenlin/),
> [Minkai Xu](https://minkaixu.com/),
> [Stefano Ermon](https://cs.stanford.edu/~ermon/), 
> [Bin Cui](https://cuibinpku.github.io/) 
> <br>**Peking University, Stanford University, Pika Labs**<br>

## Introduction


<table class="center">
    <tr>
    <td width=100% style="border: none"><img src="__asset__/method.png" style="width:100%"></td>
    </tr>
    <tr>
    <td width="100%" style="border: none; text-align: center; word-wrap: break-word">Overview of our RPG
</td>
  </tr>
</table>



**Abstract**: RPG is a powerful training-free paradigm that can utilize proprietary MLLMs (e.g., GPT-4, Gemini-Pro) or open-source local MLLMs (e.g., miniGPT-4) as the **prompt recaptioner and region planner** with our **complementary regional diffusion** to achieve SOTA text-to-image generation and editing. Our framework is very flexible and can generalize to arbitrary MLLM architectures and diffusion backbones. RPG is also capable of generating image with super high resolutions, here is an example:

<table class="center">
    <tr>
    <td width=100% style="border: none"><img src="__asset__/demo/object/icefire.png" style="width:100%"></td>
    </tr>
    <tr>
    <td width="100%" style="border: none; text-align: center; word-wrap: break-word">Text prompt: A beautiful landscape with a river in the middle the left of the river is in the evening and in the winter with a big iceberg and a small village while some people are skating on the river and some people are skiing, the right of the river is in the summer with a volcano in the morning and a small village while some people are playing.
</td>
  </tr>
</table>



## 🚩 New Updates 

**[2024.1]** Our main code along with the demo release, supporting different diffusion backbones (**SDXL**, **SD v2.0/2.1** **SD v1.4/1.5**), and one can reproduce our good results utilizing GPT-4 and Gemini-Pro. Our RPG is also compatible with local MLLMs, and we will continue to improve the results in the future.

## TODO

- [ ] Update Gradio Demo
- [ ] Release Self-Refined RPG
- [ ] Release RPG for Image Editing
- [ ] Release RPG v2 with ControlNet
- [x] Release RPG v1

## Gallery

### 1. Multi-people with complex attribute binding

<details open>
<summary>1024*1024 Examples</summary> 
<table class="center">
    <tr>
    <td width=25% style="border: none"><img src="__asset__/demo/people/cafe.png" style="width:100%"></td>
    <td width=25% style="border: none"><img src="__asset__/demo/people/cowboy.png" style="width:100%"></td>
    <td width=25% style="border: none"><img src="__asset__/demo/people/couple.png" style="width:100%"></td>
    <td width=25% style="border: none"><img src="__asset__/demo/people/tea.png" style="width:100%"></td>
  <tr>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word">A girl with white ponytail and black dress are chatting with a blonde curly hair girl in a white dress in a cafe.</td>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word">A twin-tail girl wearing a brwon cowboy hat and white shirt printed with apples, and blue denim jeans with knee boots,full body shot.</td>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word">A couple, the beautiful girl on the left, silver hair, braided ponytail, happy, dynamic, energetic, peaceful, the handsome young man on the right detailed gorgeous face, grin, blonde hair, enchanting</td>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word"> Two beautiful Chinese girls wearing cheongsams are drinking tea in the tea room, and a Chinese Landscape Painting is hanging on the wall, the girl on the left is black ponytail in red cheongsam, the girl on the right is white ponytail in orange cheongsam</td>
  </tr>
</table>
    </details>

<details open>
<summary> 2048*1024 Example</summary> 
<table class="center">
    <tr>
    <td width=100% style="border: none"><img src="__asset__/demo/people/three.png" style="width:100%"></td>
    </tr>
    <tr>
    <td width="100%" style="border: none; text-align: center; word-wrap: break-word">From left to right, a blonde ponytail Europe girl in white shirt, a brown curly hair African girl in blue shirt printed with a bird,  an Asian young man with black short hair in suit are walking in the campus happily.</td>
  </tr>
</table>
</details>    

### 2. Multi-object with complex relationship

<details open>
<summary> 1024*1024 Examples</summary> 
<table class="center">
    <tr>
    <td width=25% style="border: none"><img src="__asset__/demo/object/apple.png" style="width:100%"></td>
    <td width=25% style="border: none"><img src="__asset__/demo/object/mug.png" style="width:100%"></td>
    <td width=25% style="border: none"><img src="__asset__/demo/object/watermelon.png" style="width:100%"></td>
    <td width=25% style="border: none"><img src="__asset__/demo/object/ragdoll.png" style="width:100%"></td>
    <tr>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word">From left to right, two red apples and an apple printed shirt and an ipad on the wooden floor 
</td>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word">Seven white ceramic mugs with different geometric patterns on the marble table while a bunch of rose on the left
</td>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word">Five watermelons arranged in X shape on a wooden table, with the one in the middle being cut, realistic style, top down view.
        </td>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word"> From left to right ,bathed in soft morning light,a cozy nook features a steaming Starbucks latte on a rustic table beside an elegant vase of blooming roses,while a plush ragdoll cat purrs contentedly nearby,its eyes half-closed in blissful serenity.</td>
  </tr>
</table>
</details>   

<details open>
<summary> 2048*1024 Example</summary> 
<table class="center">
    <tr>
    <td width=100% style="border: none"><img src="__asset__/demo/object/girl.png" style="width:100%"></td>
    </tr>
    <tr>
    <td width="100%" style="border: none; text-align: center; word-wrap: break-word">A green twintail girl in orange dress is sitting on the sofa while a messy desk under a big window on the left, a lively aquarium is on the top right of the sofa, realistic style
</td>
  </tr>
</table>
</details>

### 3. RPG With ControlNet

<details open>
<summary>Open Pose Example</summary>
<table class="center">
    <tr style="line-height: 0">
    <td colspan="2" style="border: none; text-align: center">Open Pose</td>
    </tr>
    <tr>
    <td style="border: none"><img src="__asset__/demo/Controlnet/Pose.png"></td>
    <td style="border: none"><img src="__asset__/demo/Controlnet/Pose_girl.png"></td>
    </tr>
    </table>
Text prompt: A beautiful black hair girl with her eyes closed in champagne long sleeved formal dress standing in her bright room with delicate blue vases with pink roses on the left and some white roses, filled with upgraded growth all around on the right.
</details>

<details open>
<summary>Depth Map Example</summary> 
<table class="center">
    <tr style="line-height: 0">
    <td colspan="2" style="border: none; text-align: center">Depth Map</td>
    </tr>
    <tr>
    <td style="border: none"><img src="__asset__/demo/Controlnet/depth.jpg", style="width: 256px,height: 448px"></td>
    <td style="border: none"><img src="__asset__/demo/Controlnet/Depth_valley.png", style="width: 1024px, height:1792px"></td>
    </tr>
    </table>
Text prompt: Under the clear starry sky, clear river water flows in the mountains, and the lavender flower sea dances with the wind, a peaceful, beautiful, and harmonious atmosphere.
</details>

<details open>
<summary>Canny Edge Example </summary>
<table class="center">
    <tr style="line-height: 0">
    <td colspan="2" style="border: none; text-align: center">Canny Edge</td>
    </tr>
    <tr>
    <td style="border: none"><img src="__asset__/demo/Controlnet/Canny.png", style="width: 768px"></td>
    <td style="border: none"><img src="__asset__/demo/Controlnet/Canny_town.png",style="width: 2048px, height: 1024px"></td>
    </tr>
    </table>
Text prompt: From left to right, an acient Chinese city in spring, summer, autumn and winter in four different regions
</details>

## Preparations

**1. Set Environment**

```bash
git clone https://github.com/YangLing0818/RPG-DiffusionMaster
cd RPG-DiffusionMaster
conda create -n RPG python==3.9
conda activate RPG
pip install -r requirements.txt
mkdir repositories
mkdir -p generated_imgs/demo_imgs
mkdir models/Stable-diffusion
```

**2. Download Libraries**
```bash
cd repositories
git clone https://github.com/Stability-AI/generative-models
git clone https://github.com/Stability-AI/stablediffusion
git clone https://github.com/sczhou/CodeFormer
git clone https://github.com/crowsonkb/k-diffusion
git clone https://github.com/salesforce/BLIP
mv stablediffusion stable-diffusion-stability-ai
cd ..
```


**3. Download Diffusion Models and MLLMs**

In our experiments designed to attain state-of-the-art generative capabilities, we predominantly employ [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0),  [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo), and [Playground v2](https://huggingface.co/playgroundai/playground-v2-1024px-aesthetic). To generate images of high fidelity across various styles, such as photorealism, cartoons, and anime, we judiciously incorporate certain models from [CIVITA](https://civitai.com/).  For images aspiring to photorealism, we advocate the use of [AlbedoBase XL](https://civitai.com/models/140737/albedobase-xl?modelVersionId=281176) , and [DreamShaper XL](https://civitai.com/models/112902/dreamshaper-xl?modelVersionId=251662). Moreover, we generalized our paradigm to SD v1.5 and SD v2.1 to accommodate a spectrum of requisites. All pertinent checkpoints are accessible within our [Hugging Face spaces](https://huggingface.co/BitStarWalkin/RPG_models), with detailed descriptions found on the accompanying model cards.
Then we need move the downloaded diffusion model weights into the folder **models/Stable-diffusion/**, and please note that the generated images in generated_imgs/.

We recommend the utilization of GPT-4 or Gemini-Pro for users of Multilingual Large Language Models (MLLMs), as they not only exhibit superior performance but also reduce local memory. According to our experiments, the minimum requirements of VRAM is 10GB with GPT-4, if you want to use local LLM, it would need more VRAM. For those interested in using MLLMs locally, we suggest deploying [miniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) or directly engaging with substantial Local LLMs such as [Llama2-13b-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) and  [Llama2-70b-chat](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf). 
 

## Text-to-Image Generation

#### 1. Quick Start

For individuals equipped with constrained computational resources, we here provide a simple notebook demonstration that partitions the image into two equal-sized subregions. By making minor alterations to select functions within the diffusers library, one may achieve commendable outcomes utilizing base diffusion models such as SD v1.4, v1.5, v2.0, and v2.1, as mentioned in our paper. Additionally, you can apply your customized configurations to experiment with a graphics card possessing 8GB of VRAM. For an in-depth exposition, kindly refer to our [Example_Notebook](RegionalDiffusion_playground.ipynb).

#### **2. Demo** 

Note that we have uploaded detailed parameters of some examples in our paper, to make perfect reproduction, the only thing is to download the models we specify in [demo.py](template/demo.py) and run

```
python RPG.py --demo
```

You can find the results in outputs/txt2img-images which caches the generated history, or directly in generated_imgs/demo_imgs/

#### **3. Regional Diffusion with GPT-4**
Our approach can automatically generates output without pre-storing MLLM responses, leveraging Chain-of-Thought reasoning and high-quality in-context examples to obtain satisfactory results. Users only need to understand specific parameters. For example, to use GPT-4 as the planner, we can run:

```bash
python RPG.py --user_prompt 'A blonde hair girl with black suit and white skirt' --model_name 'input your model name here' --version_number 0 --api_key 'put your api_key here' --use_gpt
```

**--user_prompt** is the original prompt that roughly summarize the content contained in the image

**--model_name** is the name of the model in the directory models/Stable-diffusion/

**--version_number** is the class of our in-context examples used in generation. Our experiments suggest that in various scenarios, by employing proper in-context exemplars as few-shot samples, the planning capabilities of MLLMs can be substantially enhanced. For this case, we aim to synthesize a character bearing multiple attributes. We elect option 0, which is apt for a plan that binds multiple attributes.

**--api_key** is needed if you use GPT-4.

#### **4. Regional Diffusion with local LLMs**

We recommend to use base models with over 13 billion parameters for high-quality results, but it will increase load times and graphical memory use at the same time. We have conducted experiments on three different sized models,  Here we take llama2-13b-chat as an example, we can run:

```bash
python RPG.py --user_prompt 'A blonde hair girl with black suit and white skirt' --model_name 'input your model name here' --version_number 0 --use_local --llm_path 'local_llms/llama2-13b-chat' 
```

In local version, we only need to clarify the local llm_path to use llm locally.

Here we can also specify other usual parameters in diffusion model like:

**--cfg** which is the context-free guidance scale

**--steps** the steps to generate an image

**--seed** control the seed to make the generation reproducible

**It should be noted that we also introduce some new parameters into diffusion generation:**

**--use_base** the function of this boolean variable is to activate the base prompt in diffusion process. Utilizing the base prompt signifies that we avoid the direct amalgamation of subregions as the latent representation. Instead, we use a foundational prompt that summarizes the image's key components and obatin the overall structure latent of the image. We then compute the weighted aggregate of these latents to yield the conclusive output. This method is instrumental in addressing the problems like omission of entities in complicated prompt generation tasks, and it also contributes to refining the edges of each subregion, ensuring they are seamlessly integrated and resonate harmony.

**--base_ratio** the weight of the base prompt latent, if too small, it is difficult to work, if too big, it will  confuse the composition and properties of subregions.  We conduct ablation experiment in our paper, see our paper for more detailed information and analysis.


# 📖BibTeX
```
@article{yang2024mastering,
  title={Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs},
  author={Yang, Ling and Yu, Zhaochen and Meng, Chenlin and Xu, Minkai and Ermon, Stefano and Cui, Bin},
  journal={arXiv preprint arXiv:2401.11708},
  year={2024}
}
```

# Acknowledgements
Our RPG is a general MLLM-controlled text-to-image generation/editing framework, which is builded upon several solid works. Thanks to [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui), [regional-prompter](https://github.com/hako-mikan/sd-webui-regional-prompter), [SAM](https://github.com/facebookresearch/segment-anything)
and [IA](https://github.com/geekyutao/Inpaint-Anything) for their wonderful work and codebase! We also thank Hugging Face for sharing our [paper](https://huggingface.co/papers/2401.11708). 
