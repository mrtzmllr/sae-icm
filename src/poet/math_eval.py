import os
import torch
import pandas as pd

from src.poet.directories import write_overarching_statistics_dir, insertion_layer_tag
from src.poet.config import load_config
from src.poet.argparse import parse_args
from src.poet.dataset import retrieve_dataset
from src.poet.load_finetuned import load_finetuned_model, load_finetuned_tokenizer
from src.poet.compare_answers import generate_answers, extract_floats

import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")
print("set sharing strategy")

all_args = parse_args()

conf = load_config(all_args, run_eval=True)

model_name = conf["model"]["name"]
model_path = conf["eval"]["model_path"]

model = load_finetuned_model(conf)
tokenizer = load_finetuned_tokenizer(conf)

print(model_path)

dataset = retrieve_dataset(conf, tokenizer, split="test")

texts, solutions = generate_answers(conf, model, tokenizer, dataset)

count = 0
for idx, sol in enumerate(solutions):
    pred, gt = extract_floats(conf, texts[idx], sol)
    if pred == gt:
        count += 1

acc = count / len(solutions)
print(f"accuracy: {acc}  ({count}/{len(solutions)})")


if conf["eval"]["write_eval_file"]:
    eval_dataset = conf["eval"]["dataset"]
    eval_length = conf["dataset"]["eval_length"]
    stats_dir = write_overarching_statistics_dir(conf)
    stats_dir += "/math_eval/"
    stats_dir += eval_dataset
    stats_dir += f"/eval_length{eval_length}/"
    tag = insertion_layer_tag(conf)
    if tag is not None:
        stats_dir += f"{tag}/"

    stats_path = stats_dir + "metrics.csv"

    print(stats_dir)
    if not os.path.exists(stats_dir): os.makedirs(stats_dir, exist_ok=True)

    orthogonality_lambda = conf["sae"]["finetuning"]["orthogonality_lambda"]
    new_row = {"orthogonality_lambda": orthogonality_lambda, "accuracy": acc}
    df_new = pd.DataFrame([new_row])

    if os.path.exists(stats_path):
        df_new.to_csv(stats_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(stats_path, mode='w', header=True, index=False)
    
    print(f"Evaluation dataset metrics written successfully at accuracy {acc}!")
