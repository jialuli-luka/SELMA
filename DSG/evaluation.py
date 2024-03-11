import csv
import os
from PIL import Image
from vqa_utils import MPLUG
import json
from tqdm import tqdm
import numpy as np
import argparse
os.environ["TRANSFORMERS_CACHE"] = 'pretrained_models'
os.environ['MODELSCOPE_CACHE'] = 'pretrained_models'

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default=None)
args = parser.parse_args()

data_all = dict()
with open("dsg-1k-anns.csv") as f:
    data = csv.reader(f, delimiter=",")

    next(data)
    for row in data:
        if row[0] in data_all:
            data_all[row[0]].append(row)
        else:
            data_all[row[0]] = [row]
f.close()

print("Loading mPLUG-large")
vqa_model = MPLUG()

image_path = args.img_path

results = dict()

for k, v in tqdm(data_all.items()):
    text = v[0][1]
    generated_image = Image.open(os.path.join(image_path, f"{k}.jpg"))
    VQA = vqa_model.vqa
    id2scores = dict()
    id2dependency = dict()
    for item in v:
        id = item[3]
        answer = VQA(generated_image, item[-1])
        id2scores[str(id)] = float(answer == 'yes')

        id2dependency[str(id)] = str(item[4]).split(",")

    for id, parent_ids in id2dependency.items():
        any_parent_answered_no = False
        for parent_id in parent_ids:
            if parent_id == '0':
                continue
            if parent_id in id2scores and id2scores[parent_id] == 0:
                any_parent_answered_no = True
                break
        if any_parent_answered_no:
            id2scores[id] = 0

    average_score = sum(id2scores.values()) / len(id2scores)

    results[k] = [average_score, id2scores]

with open(f"evaluation.json", "w") as f:
    json.dump(results, f)
f.close()

with open(f"evaluation.json", "r") as f:
    score_data = json.load(f)
f.close()

results_dict = dict()

for k, v in score_data.items():
    item = data_all[k]
    for q in item:
        id = q[3]
        score_cat = q[6]
        score = v[1][id]
        if score_cat not in results_dict:
            results_dict[score_cat] = [score]
        else:
            results_dict[score_cat].append(score)

output = dict()
all = 0
count = 0
for k,v in results_dict.items():
    output[k] = np.mean(v)
    all += np.sum(v)
    count += len(v)

print("Average", all / count)
print(output)

