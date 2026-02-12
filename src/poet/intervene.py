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

assert conf["intervenability"]["do_intervene"]

model_name = conf["model"]["name"]
model_path = conf["eval"]["model_path"]

model_config = retrieve_config(conf)


model = Gemma2SAEForCausalLM.from_pretrained(model_path, config=model_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# dataset = retrieve_dataset(conf, tokenizer, split="test")
# doc_idxs = [595] 
# Jacob [7574, 11058, 9236, 6431, 15110, 1192]
# Jerry [14154, 379, 6431, 10614]

### GSM8K
# Jerry [100, 595, 1243, 211, 975, 184, 887]

# 1e-06 42983 (Jacob) 44109 (Jerry / Jerome) 9394 (sq footage) 21904 (animal-related / breed) 62849 (waiting / just a comma (,)) 31472 (counting in set / INCT) 58633 (percentage change / https) 7253 (commute or travel / to find out how many) 8912 (time and distance / computes 20/3) 21074 (doubling / emotions) 6664 (fill containers / this one actually changes the computation) 49707 (group outings / ', then') 39177 (limits and boundaries / 'on average') 12547 (geometric and vector operations or Pythagoras / changes calculation profoundly) 30395 (doubling / 'of course') 27922 (exponentiation / złoty and exponentiates at later position) 7608 (percentage-based discounts / ضايا) 23951 (iterative grid construction / taglio and wrong computation) 16697 (financial compensation / every 4 sheep Charleston has) 60844 (grouped into cartons or crates / styleType) 64650 (durations / thenГеографияCharleston)

# 59158 (Jason) 11681 (Mike) 42983 (Jacob / Jake) 44109 (Jerry / Jerome) 37532 (James) 22710 (Robert) 28504 (Jordan) 38384 (Jackson) 26534 (Paul) 28951 (David) 26342 (Andrew) 17911 (Gary / Garre)

model = insert_sae(model=model, conf=conf)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

name_dict = name_map()
alt_dict = alternative_names()

total_count = 0
total_drop = 0
total_include = 0

step = 0
for drop_name in tqdm(name_dict.keys()):
    for include_name in name_dict.keys():
        if include_name == drop_name:
            continue

        drop_feature_idx = name_dict[drop_name]
        include_feature_idx = name_dict[include_name]

        model.model.sae.intervention_indices = {
            "include": [include_feature_idx], # 1e-04 27076 47299 59158 (Jason) 11681 (Mike) # 1e-08 33368 40885 9155 57573 
            "drop": [drop_feature_idx],
        }

        dataset_dict = jerrys_dataset(drop_name)

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

        count, drop_names, include_names = evaluate_generated_texts(texts, dataset_dict, drop_name, include_name)

        total_count += count
        total_drop += drop_names
        total_include += include_names

        step += 1

total_evaluations = step * len(dataset_dict)

print(total_evaluations)
print(total_count)
print(total_drop)
print(total_include)
mean_count = total_count / total_evaluations
mean_drop = total_drop / total_evaluations
mean_include = total_include / total_evaluations
print(mean_count)
print(mean_drop)
print(mean_include)

if conf["eval"]["write_eval_file"]:
    interv_dir = write_intervenability_dir(conf)
    interv_path = interv_dir + "/metrics.csv"
    print(interv_path)
    orthogonality_lambda = conf["sae"]["finetuning"]["orthogonality_lambda"]
    new_row = {"orthogonality_lambda": orthogonality_lambda, "total_evaluations": total_evaluations, "total_steps": step, 
               "total_count": total_count, "total_drop": total_drop, "total_include": total_include,
               "mean_count": mean_count, "mean_drop": mean_drop, "mean_include": mean_include}
    df_new = pd.DataFrame([new_row])

    if os.path.exists(interv_path):
        df_new.to_csv(interv_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(interv_path, mode='w', header=True, index=False)

assert total_evaluations == len(dataset_dict) * len(name_dict.keys()) * (len(name_dict.keys()) - 1)
assert total_evaluations == 30 * 12 * 11