from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DiffusionPipeline, StableDiffusionXLPipeline
from diffusers import StableDiffusionPipeline,  UNet2DConditionModel
import torch
import json
import os
from tqdm import tqdm
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default="stabilityai/stable-diffusion-2")
parser.add_argument('--cache_dir', type=str, default="pretrained_model")
parser.add_argument('--prompt_path', type=str, default="../llm/countbench_prompt_expansion_5k.json")
parser.add_argument('--output_path', type=str, default="datasets/countbench_sdxl_1k")
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args()


# model_id = "stabilityai/stable-diffusion-2"
# model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# model_id = "CompVis/stable-diffusion-v1-4"
model_id = args.model_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if "xl" in model_id:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, cache_dir=args.cache_dir, safety_checker = None
    )
else:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir=args.cache_dir, safety_checker = None)

pipe.to(device)


data_all = dict()

with open(args.prompt_path) as f:
    data = json.load(f)
f.close()


output_path = args.output_path
batch_size = args.batch_size
prompts = []
names = []
for i, item in enumerate(tqdm(data)):
    prompt = item
    name = f"sd-sample{sample}-{i}"
    prompts.append(prompt)
    names.append(name)

    if len(prompts) == batch_size:
        images = pipe(prompts).images
        print(len(images))
        for j, img in enumerate(images):
            img.save(os.path.join(output_path, f"{names[j]}.jpg" ))

        prompts = []
        names = []

