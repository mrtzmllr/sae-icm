print("python started")
from transformers import Trainer, TrainingArguments, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from dotenv import load_dotenv
import os

import torch.optim as optim
from src.poet.config import load_config
from src.poet.utils import check_hardware
from src.poet.directories import write_sae_dir
from src.poet.sae_preprocessing import preprocessing
from src.poet.argparse import parse_args
from src.poet.insert_sae import insert_sae
from src.poet.callbacks import OrthogonalityCallback
print("loaded imports")

load_dotenv()
print("loaded env")

hf_token = os.getenv("HF_TOKEN")
cache_dir = os.getenv("CACHE_DIR")
print("got token")

all_args = parse_args()

conf = load_config(all_args)

assert not conf["model"]["finetune_language_model"]

model, tokenized_datasets = preprocessing(conf)

model = insert_sae(model, conf)

num_gpus, num_cpus = check_hardware()

sae_dir = write_sae_dir(conf)

print("sae dir:", sae_dir)

######## Fine-tuning setup ########

lr = float(conf["training"]["learning_rate"])
warmup_steps=int(conf["training"]["warmup_steps"])
optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
num_epochs = int(conf["training"]["num_train_epochs"])
per_device_train_batch_size = int(conf["training"]["per_device_train_batch_size"])
num_training_steps = int(len(tokenized_datasets["train"]) * num_epochs / (num_gpus * per_device_train_batch_size))
scheduler_function = get_cosine_schedule_with_warmup if conf["training"]["lr_scheduler_type"] == "cosine" else get_linear_schedule_with_warmup
scheduler = scheduler_function(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
gradient_checkpointing = conf["training"]["gradient_checkpointing"]
assert not gradient_checkpointing

training_args = TrainingArguments(
    output_dir=sae_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    logging_steps=int(conf["training"]["logging_steps"]),
    eval_strategy="steps",
    eval_steps=int(conf["training"]["logging_steps"]),
    report_to="wandb",
    logging_dir="./logs",
    save_steps=int(conf["training"]["save_steps"]),
    num_train_epochs=int(conf["training"]["num_train_epochs"]),
    dataloader_num_workers=num_cpus,
    push_to_hub=False,
    run_name=conf["wandb"]["name"],
    max_grad_norm=conf["training"]["max_grad_norm"],
    gradient_checkpointing=gradient_checkpointing,
    remove_unused_columns=False,
    save_safetensors=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    callbacks=[OrthogonalityCallback(validation_set=tokenized_datasets["validation"], conf=conf)],
    optimizers=(optimizer, scheduler),
)

for name, p in model.named_parameters():
    if "sae" in name: 
        print(name)
        assert p.requires_grad
    else: assert not p.requires_grad

trainer.train()

