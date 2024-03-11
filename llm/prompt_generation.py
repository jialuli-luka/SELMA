import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm
from rouge_score import rouge_scorer
import utils
import argparse


def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    if SKILL == "all":
        prompt = '''
                    Please create 20 more diverse image descriptions. Here are several rules for creating the image description: 
                    1. The image should be used to evaluate one of the skills of an image generator: spatial, count, animal, human, object, location, activity, color, attribute, food, material, shape.
                    3. The image descriptions need to mention diverse objects that frequently appear in daily life.
                    4. The image descriptions should be less than 30 words. 
                    5. The text style of the image descriptions should be different from each other.  

                    Here're some image description examples: 
                    '''
    elif SKILL == "Localized_Narrative":
        prompt = '''
                    Please create 20 more diverse image descriptions. Here are several rules for creating the image description: 
                    1. The image should be used to evaluate one of the skills of an image generator: spatial, count, animal, human, object, location, activity, color, attribute, food, material, shape.
                    3. The image descriptions need to mention diverse objects that frequently appear in daily life.
                    4. The image descriptions should be less than 50 words. 
                    5. The text style of the image descriptions should be different from each other, but generally in the style of paragraph captions. 

                    Here're some image description examples: 
                    '''
    elif SKILL == "DiffusionDB" or SKILL == "Midjourney":
        prompt = '''
                    Please create 20 more diverse image descriptions. Here are several rules for creating the image description: 
                    1. The image should be used to evaluate one of the skills of an image generator: spatial, count, animal, human, object, location, activity, color, attribute, food, material, shape.
                    3. The image descriptions need to mention diverse objects that frequently appear in daily life.
                    4. The image descriptions should be less than 50 words. 
                    5. The text style of the image descriptions should be different from each other, but describe the image in details.

                    Here're some image description examples: 
                    '''
    elif SKILL == "CountBench":
        prompt = '''
                    Please create 20 more diverse image descriptions. Here are several rules for creating the image description: 
                    1. The image descriptions should be able to evaluate model's capability to count objects.
                    2. The count should not be larger than 3.
                    3. The image descriptions need to mention diverse objects that frequently appear in daily life.
                    4. The image descriptions should be less than 30 words. 
                    5. The text style of the image descriptions should be different from each other.  

                    Here're some image description examples: 
                    '''
    elif SKILL == "Coco":
        prompt = '''
                    Please create 20 more diverse image descriptions. Here are several rules for creating the image description: 
                    1. The image should be used to evaluate one of the skills of an image generator: spatial, count, animal, human, object, location, activity, color, attribute, food, material, shape.
                    3. The image descriptions need to mention diverse objects that frequently appear in daily life.
                    4. The image descriptions should be less than 30 words. 
                    5. The text style of the image descriptions should be different from each other. 

                    Here're some image description examples: 
                    '''
    elif SKILL == "Whoops":
        prompt = '''
                    Please create 20 more diverse image descriptions. Here are several rules for creating the image description: 
                    1. The image should be used to evaluate one of the skills of an image generator: spatial, count, animal, human, object, location, activity, color, attribute, food, material, shape.
                    3. The image descriptions need to mention diverse objects that frequently appear in daily life, but have some commonsense-defying combination.
                    4. The image descriptions should be less than 30 words. 
                    5. The text style of the image descriptions should be different from each other. 

                    Here're some image description examples: 
                    '''

    for idx, caption in enumerate(prompt_instructions):
        prompt += f"\n"
        prompt += f"{idx + 1}. {caption}\n"
    return prompt


def post_process_gpt4_response(num_prompt_instructions, response):
    if response is None:
        return []
    # print(response.text.split("\n"))
    captions = []
    idx = num_prompt_instructions
    for prompt in response.text.split("\n"):
        if prompt == "":
            continue
        if f"{idx + 1}. " not in prompt:
            continue
        caption = prompt.split(f"{idx + 1}. ")[1]
        captions.append(caption)
        idx += 1

    return captions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="")
    parser.add_argument('--skills', type=str, default="Whoops")
    args = parser.parse_args()

    output_dir = args.output_dir
    num_instructions_to_generate = 1000
    model_name = "gpt-3.5-turbo-instruct"
    num_prompt_instructions = 3
    request_batch_size = 5
    temperature = 1.0
    top_p = 1.0
    num_cpus = 16
    SKILL = args.skills

    if SKILL == "Localized_Narrative":
        seeds = ["In this image we can see a pie on a large spatula. On the backside we can see some fire.",
             "Here we can see a woman having a football in her hand and behind her we can see group of people standing and it is snowy.",
            "This image is taken in a store where we can see mirrors, chairs, wall, lights, ceiling, bottles, tripod stands, frames, jar, table and a cupboard."]
    elif SKILL == "DiffusionDB":
        seeds = ["a beautiful painting of a pyramid on the moon by nekro and pascal blanche and syd mead and greg rutkowski and sachin teng and victo ngai and simon stalenhag and chris voy and tsutomu nihei. in style of cg art. ray tracing, cel shading, 3 d. ue 5. hyper detailed. realistic. maya. octane render.",
                 "a highly detailed digital illustration of a cute corgi puppy floating in outer space, hyperrealistic, stunning, beautiful, cinematic, sci - fi, greg rutkowski, artgerm, moebius, ruan jia, makoto shinkai, simon stalenhag.",
                 "portrait of a beautiful, elegant winged goddess with horns by tom bagshaw."]
    elif SKILL == "Midjourney":
        seeds = [
            "high priestess tarot card, black background, moon theme, witchery, high detail, sketch art with intricate background, dynamic pose, closeup",
            "A generic comic book location, African Jungle, comic book panel",
            "cinematic shot of a Walter White themed like game of thrones, space explosion, action, cinematic, photorealstic, 8k, cinematic lighting"
        ]
    elif SKILL == "CountBench":
        seeds = [
            "background photo of three light bulbs",
            "A look at style queen Maxima's four inauguration outfits",
            "Image showing four steps of preparing homemade goulash in a pot."
        ]
    elif SKILL ==  "Coco":
        seeds = [
            "Dogs stick their heads out of car windows.",
            "A brown purse is sitting on a green bench.",
            "A group of motorcycle riders driving past buildings."
        ]
    elif SKILL == "Whoops":
        seeds = [
            "A man sleeps in the middle of the highway.",
            "Wheat is covered by snow.",
            "A slice of chocolate cake served with cherry tomatoes."
        ]

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = list(seeds)
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            # prompt_instructions = random.sample(seeds, num_prompt_instructions)
            prompt_instructions = random.sample(all_instructions, num_prompt_instructions)
            prompt = encode_prompt(prompt_instructions)
            batch_inputs.append(prompt)
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=top_p,
        )
        request_start = time.time()
        results = utils.openai_completion(
            prompts=batch_inputs,
            model_name=model_name,
            batch_size=request_batch_size,
            decoding_args=decoding_args,
        )
        request_duration = time.time() - request_start

        process_start = time.time()
        caption_data = []
        for result in results:
            new_captions = post_process_gpt4_response(num_prompt_instructions, result)
            # print(new_captions)
            caption_data += new_captions

        total = len(caption_data)
        keep = 0
        for instruction_data_entry in caption_data:
            # computing similarity with the pre-tokenzied instructions
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry)
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > 0.8:
                continue
            else:
                keep += 1
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry)
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(machine_instruction_data, os.path.join(output_dir, f"{SKILL}_prompt_expansion_1k_gpt35.json"))
