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



**Abstract**: RPG is a powerful training-free paradigm utilizing MLLMs (e.g., GPT-4 and Gemini-Pro) as the prompt recaptioner and region planner with our complementary regional diffusion to achieve SOTA text-to-image generation and editing. Our framework is very flexible and can generalize to arbitrary MLLM architectures and diffusion backbones. For MLLMs,  we  can also use local MLLMs (e.g., MiniGPT-4) as the alternative choice. RPG is capable of generating image with super high resolutions, here is an example:

<table class="center">
    <tr>
    <td width=100% style="border: none"><img src="__asset__/demo/object/icefire.png" style="width:100%"></td>
    </tr>
    <tr>
    <td width="100%" style="border: none; text-align: center; word-wrap: break-word">Text prompt: A beautiful landscape with a river in the middle the left of the middle is in the evening and in the winter with a big iceberg and a small village while some people are skating on the river and some people are skiing, the right of the river is in the summer with a volcano in the morning and a small village while some people are playing.
</td>
  </tr>
</table>



## 🚩 New Updates 

**[2024.1]** Our main code along with the demo release, with diffusion models supports **SDXL**, **SD v2.0/2.1** **SD v1.4/1.5** ,  and we can produce good results utilizing GPT-4 and Gemini-Pro. We are also compatible with local MLLMs, but due to the limited context window and the size the model, it cannot always yield good results, we will try to fix this problem in the future.

## TODO

- [ ] Update Gradio demo
- [ ] Release RPG for image editing
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
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word">A twin-tail girl wearing a brwon cowboy hat and white shirt printed with apple, and blue denim jeans with knee boots,full body shot.</td>
    <td width="25%" style="border: none; text-align: center; word-wrap: break-word">A couple, the girl beautiful girl on the right, silver hair, braided ponytail, happy, dynamic, energetic, peaceful, the handsome young man on the right detailed gorgeous face, grin, blonde hair, enchanting</td>
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
    <td width="100%" style="border: none; text-align: center; word-wrap: break-word">A green twintail girl in orange dress is sitting on the sofa while a messy desk in under a big window on the left, while a lively aquarium is on the top right of the sofa, realistic style
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
Text prompt: A beautiful black hair girl with her eyes closed in champagne long sleeved formal dress standing in her bright room with delicate blue vases with pink roses on the left and Some white roses, filled with upgraded growth all around on the right.
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
Text prompt: Under the clear starry sky, clear river water flows in the mountains, and the lavender flower sea in front of me dances with the wind, A peaceful, beautiful, and harmonious atmosphere.
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

**Setup repository and conda environment**

```bash
git clone https://github.com/YangLing0818/RPG-DiffusionMaster
cd RPG-DiffusionMaster
conda create -n RPG python==3.9
conda activate RPG
pip install -r requirements.txt
mkdir repositories
cd repositories
git clone https://github.com/Stability-AI/generative-models
git clone https://github.com/Stability-AI/stablediffusion
git clone https://github.com/sczhou/CodeFormer
git clone https://github.com/crowsonkb/k-diffusion
git clone https://github.com/salesforce/BLIP
mv stablediffusion stable-diffusion-stability-ai
```

**Download Checkpoints and MLLMs configuration**

Here, we use [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) , [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo) , [Playground v2](https://huggingface.co/playgroundai/playground-v2-1024px-aesthetic) in our most experiments to achieve SOTA level generation, to further generate high-fidelity image in different style (e.g., photorealistic, cartoon, anime ) , we select some models from [CIVITA](https://civitai.com/). For photorealistic image, we recommand [AlbedoBase XL](https://civitai.com/models/140737/albedobase-xl?modelVersionId=281176) , and [DreamShaper XL](https://civitai.com/models/112902/dreamshaper-xl?modelVersionId=251662) . We also select models based on SD v1.5 and SD v2.1 to cater for different needs. The checkpoints are all available at our  [huggingface spaces](https://huggingface.co/BitStarWalkin/RPG_models) , you can see the model card for detail.

For MLLMs, we strongly suggest you to use GPT-4 or Gemini-Pro which is more powerful and save graphics memory. Our experiments are conducted on A100 80GB but should work on other cards with at least 12GB VRAM . If you want to try MLLMs locally, we suggest you to use [miniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) or you can directly use large local LLMs like  [Llama2-13b-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) and [Llama2-70b-chat](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf). For [Llama2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), we find it perform poorly and cannot get correct split ratio. 

## Text-to-Image Generation

#### 1. Quick Start

If you have limited computing device, we presents a double-version demo which split the image into two equally sized subregions . By simply modified some functions in diffusers library, we can get satisfactory results by using base diffusion models like SD v1.4/1.5/2.0/2.1 mentioned in our paper. You can also use your personalized settings to give it a try with a card of 8GB VRAM. See details in  our [Example_Notebook](RegionalDiffusion_playground.ipynb).

#### **2. Demo** 

Note that we have uploaded detailed parameters of some examples in our paper, to make perfect reproduction, the only thing is to download the models we specify in [demo.py](template/demo.py) and run

```
python main.py --demo
```

You can find the results in outputs/txt2img-images which caches the generated history, or directly in generated_imgs/demo_imgs/

#### **3. Regional Generation with GPT-4**

One of the highlights of our work is that we don't need to cache the MLLMs/LLMs response in advance, our generation task is totally automatically conducted without even the need to copy and paste from MLLMs by leveraging our Chain-of-Thought and well-formated high quality in-context examples strategy. The only thing we need to do is to figure out  the function of each parameters. For example, to use GPT-4 as the planner, we can run

```bash
python main.py --user_prompt 'A blonde hair girl with black suit and white skirt' --model_name 'input your model name here' --version_number 0 --api_key 'put your api_key here' --use_gpt
```

**--user_prompt** is the original prompt that roughly summarize the content contained in the image

**--model_name** is the name of the model in the directory models/Stable-diffusion/

**--version_number** is the class of our in-context examples used in generation. We find that in different scenarios, if we use the relevant in-context examples as few-shot sample, the planing ability of the MLLMs can be further boost. In this example, we want to generate a girl with multiple attributes, here we choose 0 that fits for multi-attribute binding plan.

**--api_key** is needed if you use GPT-4.

#### **4. Regional Generation with local LLMs**

We recommend to use base model with more than 13B parameters to achieve satisfactory results. But it will take more time to load the model and inference, with the significant increase in graphic memory usage. We conduct experiments on theses three base models. For 13B model, the recommended device is A100 80GB, for 70B model, the recommended device is 8*A100 80GB, here we take llama2-13b-chat as an example, we can run

```bash
python RPG.py --user_prompt 'A blonde hair girl with black suit and white skirt' --model_name 'input your model name here' --version_number 0 --use_local --llm_path 'local_llms/ your llm name' 
```

In local version, we only need to clarify the local llm_path to use llm locally.

Here we can also specify other usual parameters in diffusion model like:

**--cfg** which is the context-free guidance scale

**--steps** the steps to generate an image

**--seed** control the seed to make the generation reproducible

It should be noted that we also introduce some new parameters into diffusion generation:

**--use_base** the function of this bool variable is to activate the base prompt in diffusion process. When we use base prompt, it means that we don't just directly take the combination of subregions as the latent, we use a base prompt that summarize the content contained in the image to get another latent and use the weighted sum of them to be the final results, which could help to solve the common entity missing issues in complex prompt generation tasks, and help to make the boundary of each subregions smooth and harmonious.

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
