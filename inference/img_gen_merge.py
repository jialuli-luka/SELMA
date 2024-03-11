from diffusers import StableDiffusionPipeline,  UNet2DConditionModel, StableDiffusionXLPipeline, DiffusionPipeline
import torch
import json
import csv
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default=None, nargs='+')
parser.add_argument('--model_id', type=str, default=None)
parser.add_argument('--output_path', type=str, default="data")
parser.add_argument('--steps', type=int, default=3000)
parser.add_argument('--cache_dir', type=str, default="pretrained_models")
parser.add_argument('--eval_benchmark', type=str, default="DSG")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

base_model = args.model_id
ckpt = args.ckpt
output_path = args.output_path
cache_dir = args.cache_dir
load_ckpt_step = args.steps
load_lora = True
scale = 100
adapter_names = []
adapter_weights = []

if "xl" in base_model:
    pipe = StableDiffusionXLPipeline.from_pretrained(base_model, torch_dtype=torch.float16,cache_dir=cache_dir)
else:
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16, cache_dir=cache_dir)

for model in ckpt:
    pipe.load_lora_weights(model + f"/checkpoint-{load_ckpt_step}/",
                           weight_name="pytorch_lora_weights.safetensors", adapter_name=model.split("/")[-1])
    adapter_names.append(model.split("/")[-1])
    adapter_weights.append(1/len(args.model_id))

pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
print(pipe.get_list_adapters())
pipe.to("cuda")

if not os.path.exists(output_path):
    os.mkdir(output_path)


if args.eval_benchmark == "DSG":
    data_all = dict()
    with open("../DSG/dsg-1k-anns.csv") as f:
        data = csv.reader(f, delimiter=",")
        next(data)
        for row in data:
            if row[0] in data_all:
                continue
            data_all[row[0]] = row[1]
    f.close()
    print(len(data_all))

    for k, v in tqdm(data_all.items()):
        name = k
        prompt = v
        images = pipe(prompt, num_images_per_prompt=1, cross_attention_kwargs={"scale": scale / 100}).images[0]
        images.save(os.path.join(output_path, f"{name}.jpg"))

elif args.eval_benchmark == "TIFA":
    with open("../TIFA/tifa_v1.0/tifa_v1.0_text_inputs.json", "r") as f:
        data_all = json.load(f)
    f.close()

    for item in tqdm(data_all):
        prompt = item["caption"]
        name = item["id"]
        images = pipe(prompt, num_images_per_prompt=1, cross_attention_kwargs={"scale": scale / 100}).images[0]
        images.save(os.path.join(output_path, f"{name}.jpg"))
