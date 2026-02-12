import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.poet.config import load_config
from src.poet.argparse import parse_args
from src.poet.directories import write_feature_storage_dir, write_overarching_statistics_dir
from src.poet.prompt_template import dataset_to_inputs, dataset_score, add_to_accuracy_dictionary


all_args = parse_args()
conf = load_config(all_args, run_eval=True)

llama_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_name)
model = AutoModelForCausalLM.from_pretrained(
    llama_name,
    device_map="auto",
    use_cache=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prompt_dataset, correct_indices_dict = dataset_score(conf)

llama_enc = dataset_to_inputs(llama_tokenizer, prompt_dataset)

input_ids_tensor = llama_enc["input_ids"].to(model.device)
attention_mask_tensor = llama_enc["attention_mask"].to(model.device)


accuracy_dict = {}
total_overlap = 0
if len(input_ids_tensor) > 25:
    assert (len(input_ids_tensor) / 25).is_integer()
    num_batches = int(len(input_ids_tensor) / 25)
    for b in range(num_batches):
        input_sub_tensor = input_ids_tensor[(b * 25):((b+1) * 25), :]
        attention_sub_tensor = attention_mask_tensor[(b * 25):((b+1) * 25), :]
        correct_sub_keys = list(correct_indices_dict.keys())[(b * 25):((b+1) * 25)]
        correct_sub_dict = {k: correct_indices_dict[k] for k in correct_sub_keys}
        assert len(correct_sub_dict) == 25
        assert len(list(correct_sub_dict.keys())) == len(set(correct_sub_dict.keys())) == 25
        accuracy_dict, batch_overlap = add_to_accuracy_dictionary(conf, model, llama_tokenizer, input_sub_tensor, attention_sub_tensor, correct_sub_dict, accuracy_dict)
        total_overlap += batch_overlap

else: accuracy_dict, total_overlap = add_to_accuracy_dictionary(conf, model, llama_tokenizer, input_ids_tensor, attention_mask_tensor, correct_indices_dict, accuracy_dict)

average_overlap = total_overlap / len(input_ids_tensor)

print("lambda")
print(conf["sae"]["finetuning"]["orthogonality_lambda"])
print("average_overlap")
print(average_overlap)


if conf["eval"]["write_eval_file"]:
    num_total_snippets = conf["interpretability"]["total_snippets"]
    num_correct_snippets = conf["interpretability"]["correct_snippets"]
    total_evaluations = conf["interpretability"]["total_evaluations"]

    stats_dir = write_overarching_statistics_dir()
    stats_dir += "/interpretability/"
    stats_dir += "score/"
    stats_dir += f"total_evaluations{total_evaluations}/"
    stats_dir += f"num_total_snippets{num_total_snippets}/"
    stats_dir += f"num_correct_snippets{num_correct_snippets}/"
    stats_path = stats_dir + "metrics.csv"

    print(stats_dir)
    if not os.path.exists(stats_dir): os.makedirs(stats_dir, exist_ok=True)

    orthogonality_lambda = conf["sae"]["finetuning"]["orthogonality_lambda"]
    new_row = {"orthogonality_lambda": orthogonality_lambda, "average_overlap": average_overlap}
    df_new = pd.DataFrame([new_row])

    if os.path.exists(stats_path):
        df_new.to_csv(stats_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(stats_path, mode='w', header=True, index=False)


    json_dir = write_feature_storage_dir(conf)
    json_path = json_dir + "/accuracy_dict.jsonl"

    with open(json_path, "w") as f:
        for ftr_index, entries in accuracy_dict.items():
            choice_list = entries["choice_list"]
            correct_indices = entries["correct_indices"]
            overlap = entries["overlap"]
            json_line = {
                "feature_idx": ftr_index,
                "choice_list": choice_list,
                "correct_indices": correct_indices,
                "overlap": overlap,
            }
            f.write(json.dumps(json_line) + "\n")
    print(json_path)
    print("Overlap statistics written to .jsonl file!")