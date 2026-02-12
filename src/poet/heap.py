import os
import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
from heapq import heappush, heappushpop, nsmallest
import json

from src.poet.directories import write_feature_storage_dir, write_overarching_statistics_dir
from src.poet.config import load_config
from src.poet.insert_sae import Gemma2SAEForCausalLM, insert_sae
from src.poet.argparse import parse_args
from src.poet.dataset import retrieve_dataset
from src.poet.model_config import retrieve_config

all_args = parse_args()

conf = load_config(all_args, run_eval=True)
model_name = conf["model"]["name"]
dataset_name = conf["eval"]["dataset"]

model_path = conf["eval"]["model_path"]

model_config = retrieve_config(conf)

model = Gemma2SAEForCausalLM.from_pretrained(model_path, config=model_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = insert_sae(model=model, conf=conf)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

d_sae = conf["sae"]["d_sae"]
max_elements = conf["eval"]["max_features_stored"]
top_k = conf["sae"]["top_k"]

seq_len = 512

feature_dict = {
    f: []
    for f in range(d_sae)
}

print("hi")

dataset = retrieve_dataset(conf, tokenizer, split="test")
print(len(dataset["test"]))
print("hi")

active_table = torch.zeros(d_sae).to(model.device)

for test_set_idx, ex in enumerate(tqdm(dataset["test"])):
    with torch.no_grad():
        outs = model(
            input_ids = torch.tensor([ex["input_ids"]]).to(model.device),
            attention_mask = torch.tensor([ex["attention_mask"]]).to(model.device),
            labels = torch.tensor([ex["labels"]]).to(model.device),
        )

    feature_indices = model.model.sae.indices[0]
    feature_values = model.model.sae.values[0]

    # active_features stuff
    feature_indices_set = list(set([x for row in feature_indices for x in row]))
    active_table += torch.ones_like(active_table)
    active_table[feature_indices_set] = 0
    
    assert len(feature_indices) == 512
    assert len(feature_indices[0]) == top_k

    for token_idx in range(seq_len):
        indices_row = feature_indices[token_idx]
        values_row  = feature_values[token_idx]
        for sae_index in range(top_k):
            ftr_index = indices_row[sae_index]
            entry = (feature_values[token_idx][sae_index], test_set_idx, token_idx)
            heap = feature_dict[ftr_index]
            if len(heap) < max_elements:
                heappush(heap, entry)
            else:
                heappushpop(heap, entry)

    model.model.sae.indices = []
    model.model.sae.values = []

json_dict = {}

json_dir = write_feature_storage_dir(conf)
json_path = json_dir + "/feature_dict.jsonl"

# active_features stuff
if conf["interpretability"]["compute_active"]:
    stats_dir = write_overarching_statistics_dir()
    if conf["sae"]["binary"]["use_binary"]: 
        stats_path = stats_dir + "/dead_features_binary.csv"
    else:
        stats_path = stats_dir + "/dead_features.csv"

    dataset_length = len(dataset["test"])
    dead = 0
    sum_active = torch.relu(active_table - torch.full_like(active_table, dataset_length - 1))
    dead_features_fraction = (torch.sum(sum_active) / d_sae).detach().item()
    orthogonality_lambda = conf["sae"]["finetuning"]["orthogonality_lambda"]
    new_row = {"orthogonality_lambda": orthogonality_lambda, "dead_features_fraction": dead_features_fraction}
    df_new = pd.DataFrame([new_row])
    print("lambda")
    print(orthogonality_lambda)
    print("dead_features_fraction")
    print(dead_features_fraction)

    if os.path.exists(stats_path):
        df_new.to_csv(stats_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(stats_path, mode='w', header=True, index=False)


with open(json_path, "w") as f:
    for ftr_index, heap in feature_dict.items():
        if len(heap) == 0:
            continue
        sorted_entries = nsmallest(len(heap), heap)
        json_line = {
            "feature_idx": ftr_index,
            "entries": [list(e) for e in sorted_entries]
        }
        f.write(json.dumps(json_line) + "\n")