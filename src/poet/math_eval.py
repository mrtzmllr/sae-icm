import os
import torch
import pandas as pd
from transformers import AutoTokenizer

from src.poet.directories import write_overarching_statistics_dir
from src.poet.config import load_config
from src.poet.insert_sae import Gemma2SAEForCausalLM, insert_sae
from src.poet.argparse import parse_args
from src.poet.dataset import retrieve_dataset
from src.poet.model_config import retrieve_config
from src.poet.compare_answers import generate_answers, extract_floats

all_args = parse_args()

conf = load_config(all_args, run_eval=True)

model_name = conf["model"]["name"]
model_path = conf["eval"]["model_path"]

model_config = retrieve_config(conf)

model = Gemma2SAEForCausalLM.from_pretrained(model_path, config=model_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = insert_sae(model=model, conf=conf)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(model_path)

dataset = retrieve_dataset(conf, tokenizer, split="test")

texts, solutions = generate_answers(conf, model, tokenizer, dataset)

count = 0
for idx, sol in enumerate(solutions):
    pred, gt = extract_floats(conf, texts[idx], sol)
    if pred == gt:
        count += 1

acc = count / len(solutions)

if conf["eval"]["write_eval_file"]:
    eval_dataset = conf["eval"]["dataset"]
    eval_length = conf["dataset"]["eval_length"]
    stats_dir = write_overarching_statistics_dir()
    stats_dir += "/math_eval/"
    stats_dir += eval_dataset
    stats_dir += f"/eval_length{eval_length}/"
    
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
