import json
import json, os
from collections import defaultdict
from tqdm import tqdm
from tifascore.vqa_models import VQAModel
from statistics import mean, stdev
import argparse
os.environ["TRANSFORMERS_CACHE"] = 'pretrained_models'
os.environ['MODELSCOPE_CACHE'] = 'pretrained_models'
os.environ["TORCH_HOME"] = 'pretrained_models'
os.environ['HF_HOME'] = 'pretrained_models'

def tifa_score_benchmark(vqa_model_name, question_answer_path, caption_id2img_fn):
    # load the questions and answers
    with open(question_answer_path) as f:
        question_answer_pairs = json.load(f)

    # load the VQA model
    vqa_model = VQAModel(vqa_model_name)

    tifa_statistics = {"scores": defaultdict(list),
                       "type_scores": defaultdict(list)}
    question_logs = defaultdict(dict)

    for question_answer_pair in tqdm(question_answer_pairs):
        # get text input id
        caption_id = question_answer_pair['id']

        # read the question, choices, and answers
        if question_answer_pair['question'] not in question_logs[caption_id]:
            question_logs[caption_id][question_answer_pair['question']] = question_answer_pair
        choices = question_answer_pair['choices']

        img_fn = caption_id2img_fn[str(caption_id)]

        # get VQA answer
        vqa_answer = vqa_model.multiple_choice_vqa(img_fn, question_answer_pair['question'], choices=choices)

        free_form_answer, mc_answer = vqa_answer["free_form_answer"], vqa_answer["multiple_choice_answer"]
        question_logs[caption_id][question_answer_pair['question']]['free_form_vqa'] = free_form_answer
        question_logs[caption_id][question_answer_pair['question']]['multiple_choice_vqa'] = mc_answer

        # compute multiple choice score
        score = int(mc_answer == question_answer_pair['answer'])
        question_logs[caption_id][question_answer_pair['question']]['scores'] = score

        # statistics of the scores
        tifa_statistics['scores'][caption_id].append(score)
        tifa_statistics['type_scores'][question_answer_pair['element_type']].append(score)

    question_logs = dict(question_logs)
    result_dict = {}

    # compute the average score
    averaged_scores = [mean(scores) for caption_id, scores in tifa_statistics["scores"].items()]

    result_dict = {"tifa_average": mean(averaged_scores),
                   "tifa_stdev": stdev(averaged_scores),
                   "accuracy_by_type": {type_: mean(scores) for type_, scores in tifa_statistics["type_scores"].items()}
                   }

    print(f"Average TIFA is {result_dict['tifa_average']}")

    # record the details of each question
    result_dict["question_details"] = question_logs

    # record the scores averaged by captions
    result_dict["caption_scores"] = {caption_id: mean(scores) for caption_id, scores in
                                     tifa_statistics["scores"].items()}

    return result_dict


if __name__ == "__main__":

    #####################################
    ## Test TIFA score on benchmark
    #####################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default=None)
    args = parser.parse_args()


    # test tifa benchmarking
    with open("tifa_v1.0/tifa_v1.0_question_answers.json", "r") as f:
        all_answer = json.load(f)
    f.close()

    file_dir = args.img_path
    vqa_model = "blip2-flant5xl"

    imgs = os.listdir(file_dir)

    results = dict()

    for img in imgs:
        id = img.split(".")[0]
        results[id] = os.path.join(file_dir, img)

    print(len(results))
    ids = set()
    for key, value in results.items():
        ids.add(key)
    new_answer = []
    for item in all_answer:
        if item["id"] in ids:
            new_answer.append(item)

    with open("sample_question_answers.json", "w") as f:
        json.dump(new_answer, f)
    f.close()

    results = tifa_score_benchmark(vqa_model, "sample_question_answers.json", results)

    with open(f"evaluation.json", "w") as f:
        json.dump(results, f, indent=4)
