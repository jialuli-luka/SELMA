import os
import jsonlines
import json
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default="data")
parser.add_argument('--prompt_path', type=str, default="llm")
args = parser.parse_args()


img_path = args.img_path
output_path = "datasets/cb_sd_unfiltered_1k"

with open(args.prompt_path, "r") as f:
    captions = json.load(f)
f.close()

results = []
for file in tqdm(os.listdir(img_path)):
    item = dict()
    item["file_name"] = file
    item["image"] = os.path.join(img_path, file)
    i = file.split(".")[0].split("-")[-1]
    item["text"] = captions[int(i)]
    results.append(item)

with jsonlines.open(os.path.join(img_path, "metadata.jsonl"), 'w') as writer:
    writer.write_all(results)
writer.close()