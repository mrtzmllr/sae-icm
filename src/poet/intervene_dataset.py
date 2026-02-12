import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.poet.config import load_config
from src.poet.argparse import parse_args
from src.poet.directories import write_feature_storage_dir, write_intervenability_dir
from src.poet.prompt_template import create_prompt_dataset, jerrys_dataset, name_map, alternative_names, evaluate_generated_texts
from src.poet.model_config import retrieve_config
from src.poet.dataset import retrieve_dataset
from src.poet.insert_sae import Gemma2SAEForCausalLM, insert_sae


all_args = parse_args()
conf = load_config(all_args, run_eval=True)

# assert conf["intervenability"]["do_intervene"]

model_name = conf["model"]["name"]
model_path = conf["eval"]["model_path"]

model_config = retrieve_config(conf)


model = Gemma2SAEForCausalLM.from_pretrained(model_path, config=model_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = retrieve_dataset(conf, tokenizer, split="test")
doc_idxs = [211] 
# Jacob [7574, 11058, 9236, 6431, 15110, 1192]
# Jerry [14154, 379, 6431, 10614]

### GSM8K
# Jerry [100, 595, 1243, 211, 975, 184, 887]

# 1e-06 42983 (Jacob) 44109 (Jerry / Jerome) 9394 (sq footage) 21904 (animal-related / breed) 62849 (waiting / just a comma (,)) 31472 (counting in set / INCT) 58633 (percentage change / https) 7253 (commute or travel / to find out how many) 8912 (time and distance / computes 20/3) 21074 (doubling / emotions) 6664 (fill containers / this one actually changes the computation) 49707 (group outings / ', then') 39177 (limits and boundaries / 'on average') 12547 (geometric and vector operations or Pythagoras / changes calculation profoundly) 30395 (doubling / 'of course') 27922 (exponentiation / złoty and exponentiates at later position) 7608 (percentage-based discounts / ضايا) 23951 (iterative grid construction / taglio and wrong computation) 16697 (financial compensation / every 4 sheep Charleston has) 60844 (grouped into cartons or crates / styleType) 64650 (durations / thenГеографияCharleston)

# 59158 (Jason) 11681 (Mike) 42983 (Jacob / Jake) 44109 (Jerry / Jerome) 37532 (James) 22710 (Robert) 28504 (Jordan) 38384 (Jackson) 26534 (Paul) 28951 (David) 26342 (Andrew) 17911 (Gary / Garre)

model = insert_sae(model=model, conf=conf)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.model.sae.intervention_indices = {
    "include":  [27076], # 1e-04 27076 47299 59158 (Jason) 11681 (Mike) # 1e-08 33368 40885 9155 57573 
    "drop": [44109],
}


# example = dataset["test"][doc_idxs[0]]

dataset_dict = jerrys_dataset("Jermey")

prompts = []
for jerry_doc in dataset_dict.keys():
    prompt = dataset_dict[jerry_doc]["sentence"][0]
    pp = f"Problem:\n{prompt}\n\nSolution:\n"
    prompts.append(pp)

enc = tokenizer(
    prompts,
    truncation=True,
    padding="max_length",
    padding_side = "left",
    max_length=512,
    return_tensors="pt",
)

input_ids = enc.input_ids
attention_mask = enc.attention_mask

assert input_ids.shape[0] == len(dataset_dict)

with torch.no_grad():
    output_ids = model.generate(
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        max_new_tokens=400,
        do_sample=False,
    )

texts = tokenizer.batch_decode(
    output_ids,
    skip_special_tokens=True,
)

for text in texts:
    print(text)
    print("\n\n\n")

# prompt = "David puts $25 in his piggy bank every month for 2 years to save up for a vacation. He had to spend $400 from his piggy bank savings last week to repair his car. How many dollars are left in his piggy bank?"

# pp = f"Problem:\n{prompt}\n\nSolution:\n"

# enc = tokenizer(
#     prompt,
#     # truncation=True,
#     # padding="max_length",
#     # padding_side = "left",
#     # max_length=512,
#     return_tensors="pt",
# )

# input_ids = enc.input_ids
# attention_mask = enc.attention_mask

# # assert input_ids.shape[0] == len(dataset_dict)

# # input_ids = torch.tensor([example["input_ids"]])
# # attention_mask = torch.tensor([example["attention_mask"]])

# with torch.no_grad():
#     output_ids = model.generate(
#         input_ids=input_ids.to(model.device),
#         attention_mask=attention_mask.to(model.device),
#         max_new_tokens=200,
#         do_sample=False,
#     )

# texts = tokenizer.decode(
#     output_ids[0],
#     skip_special_tokens=True,
# )

# print(texts)