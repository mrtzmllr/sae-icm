import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.poet.config import load_config
from src.poet.argparse import parse_args
from src.poet.directories import write_feature_storage_dir, write_intervenability_dir, write_interv_tokens_dir, write_interv_features_dir
from src.poet.prompt_template import create_prompt_dataset, name_map, alternative_names, evaluate_generated_texts, group_dataset_map
from src.poet.dataset import retrieve_dataset
from src.poet.load_finetuned import load_finetuned_model, load_finetuned_tokenizer


all_args = parse_args()
conf = load_config(all_args, run_eval=True)

assert conf["intervenability"]["do_intervene"]

model_name = conf["model"]["name"]
model_path = conf["eval"]["model_path"]
interv_group = conf["intervenability"]["group"]
replacement_group = conf["intervenability"]["replacement_group"]
assert interv_group in ["names", "locations", "animals"]
assert replacement_group in ["names", "locations", "animals"]

model = load_finetuned_model(conf)
tokenizer = load_finetuned_tokenizer(conf)

name_dict = name_map(conf)[interv_group]
replacement_dict = name_map(conf)[replacement_group]
group_dataset = group_dataset_map(conf)

print(name_dict)

total_count = 0
total_drop = 0
total_include = 0

step = 0

for drop_name in tqdm(name_dict.keys()):
    for include_name in replacement_dict.keys():
        
        if include_name == drop_name:
            continue

        drop_feature_idx = name_dict[drop_name]
        include_feature_idx = replacement_dict[include_name]

        assert drop_feature_idx is not None and include_feature_idx is not None, (
            f"name_map has no SAE feature index for drop='{drop_name}' / include='{include_name}'. "
            "Fill in the Llama SAE feature indices in prompt_template.name_map()."
        )

        model.model.sae.intervention_indices = {
            "include": [include_feature_idx],
            "drop": [drop_feature_idx],
        }

        dataset_dict = group_dataset(drop_name)
        
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

        model.model.sae.indices = []
        
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

        # write tokens
        tokens_dir = write_interv_tokens_dir(conf, drop_name, include_name)
        tokens_pth = tokens_dir + "/tokens.pt"
        generated_ids = output_ids[:, input_ids.shape[1]:]
        torch.save(generated_ids, tokens_pth)

        feature_ids = model.model.sae.indices
        indices = torch.stack(feature_ids, dim=1)
        features_dir = write_interv_features_dir(conf, drop_name, include_name)
        features_pth = features_dir + "/features.pt"
        torch.save(feature_ids, features_pth)

        model.model.sae.step = 0

        count, drop_names, include_names = evaluate_generated_texts(conf, texts, dataset_dict, drop_name, include_name)

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

if interv_group == replacement_group: assert total_evaluations == len(dataset_dict) * len(name_dict.keys()) * (len(name_dict.keys()) - 1)
else: assert total_evaluations == len(dataset_dict) * len(name_dict.keys()) * len(replacement_dict.keys())