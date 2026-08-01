from transformers import Trainer, TrainingArguments
import json
from pathlib import Path
from src.poet.config import load_config
from src.poet.dataset import retrieve_dataset
from src.poet.directories import write_output_dir
from src.poet.argparse import parse_args
from src.poet.load_finetuned import load_finetuned_model, load_finetuned_tokenizer

all_args = parse_args()

conf = load_config(all_args, run_eval=True)

model_path = conf["eval"]["model_path"]
print(model_path)

model = load_finetuned_model(conf)
tokenizer = load_finetuned_tokenizer(conf)

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

metrics = trainer.evaluate(dataset, metric_key_prefix="eval")
print(metrics)


if conf["eval"]["write_eval_file"]:
    eval_path = Path(output_dir) / "metrics.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved evaluation metrics to {eval_path}")
