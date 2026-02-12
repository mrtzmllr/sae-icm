from transformers import AutoTokenizer, Trainer, TrainingArguments, Gemma2ForCausalLM
import os
import json
from pathlib import Path
from src.poet.config import load_config
from src.poet.dataset import retrieve_dataset
from src.poet.directories import write_output_dir
from src.poet.insert_sae import Gemma2SAEForCausalLM, insert_sae
from src.poet.argparse import parse_args

from src.poet.pretrained import retrieve_pretrained
from src.poet.model_config import retrieve_config

all_args = parse_args()

conf = load_config(all_args, run_eval=True)
model_name = conf["model"]["name"]

model_path = conf["eval"]["model_path"]
print(model_path)

model_config = retrieve_config(conf)

model = Gemma2SAEForCausalLM.from_pretrained(model_path, config=model_config) # Gemma2SAEForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_name)

### sanity
# model = Gemma2SAEForCausalLM.from_pretrained(model_name)
# _, tokenizer = retrieve_pretrained(conf)

model = insert_sae(model=model, conf=conf)

modls = [k for k, _ in model.named_modules() if "lora" in k.lower()]

print("LoRA modules found:", len(modls))

dataset = retrieve_dataset(conf, tokenizer, split="test")

output_dir = write_output_dir(conf, train=False)

test_args = TrainingArguments(
    output_dir=output_dir,
    per_device_eval_batch_size=1
)

trainer = Trainer(
    model=model,
    args=test_args,
)

if conf["training"]["include_sae"]:
    train_W_enc = conf['sae']['train_W_enc']
    train_b_enc = conf['sae']['train_b_enc']
    train_b_dec = conf['sae']['train_b_dec']

    for name, p in model.named_parameters():
        if "sae" in name: 
            if train_W_enc and 'W_enc.weight' in name: assert p.requires_grad
            elif train_b_enc and 'W_enc.bias' in name: assert p.requires_grad
            elif train_b_dec and 'W_dec.bias' in name: assert p.requires_grad
            else: assert not p.requires_grad

metrics = trainer.evaluate(dataset, metric_key_prefix="eval")
print(metrics)


if conf["eval"]["write_eval_file"]:
    eval_path = Path(output_dir) / "metrics.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved evaluation metrics to {eval_path}")