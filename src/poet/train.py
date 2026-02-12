print("python started")
from transformers import Trainer, TrainingArguments, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from dotenv import load_dotenv
import os
# import torch
# import wandb # for logging specific arguments
import torch.optim as optim
from src.poet.pretrained import retrieve_pretrained
from src.poet.dataset import retrieve_dataset
from src.poet.lora import apply_lora
from src.poet.config import load_config
from src.poet.utils import check_hardware
from src.poet.directories import write_output_dir
from src.poet.insert_sae import insert_sae
from src.poet.callbacks import SAEMetricsCallback, SaveSAECallback, TopKScheduleCallback
from src.poet.argparse import parse_args
print("loaded imports")

load_dotenv()
print("loaded env")

hf_token = os.getenv("HF_TOKEN")
cache_dir = os.getenv("CACHE_DIR")
print("got token")

all_args = parse_args()

conf = load_config(all_args)

assert conf["model"]["finetune_language_model"]

model, tokenizer = retrieve_pretrained(conf)

model = insert_sae(model, conf)

model.config.use_cache = False
if conf["training"]["gradient_checkpointing"]: model.gradient_checkpointing_enable()

print("loaded model")

tokenized_datasets = retrieve_dataset(conf, tokenizer)
print("loaded dataset")

######## Fine-tuning setup ########

if conf["model"]["use_lora"]: model = apply_lora(model, conf)
print("applied LoRA")

num_gpus, num_cpus = check_hardware()

output_dir = write_output_dir(conf)

print("output dir:", output_dir)

include_sae = conf["training"]["include_sae"]

lr = float(conf["training"]["learning_rate"])
warmup_steps=int(conf["training"]["warmup_steps"])
optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
num_epochs = int(conf["training"]["num_train_epochs"])
per_device_train_batch_size = int(conf["training"]["per_device_train_batch_size"])
num_training_steps = int(len(tokenized_datasets["train"]) * num_epochs / (num_gpus * per_device_train_batch_size))
scheduler_function = get_cosine_schedule_with_warmup if conf["training"]["lr_scheduler_type"] == "cosine" else get_linear_schedule_with_warmup
scheduler = scheduler_function(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    logging_steps=int(conf["training"]["logging_steps"]),
    eval_strategy="steps",
    eval_steps=int(conf["training"]["logging_steps"]),
    report_to="wandb",
    logging_dir="./logs",
    save_steps=int(conf["training"]["save_steps"]),
    num_train_epochs=int(conf["training"]["num_train_epochs"]),
    dataloader_num_workers=num_cpus,
    ddp_find_unused_parameters=False,
    push_to_hub=False,
    run_name=conf["wandb"]["name"],
    gradient_checkpointing=conf["training"]["gradient_checkpointing"],
    max_grad_norm=conf["training"]["max_grad_norm"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    callbacks=[SAEMetricsCallback(tokenized_datasets["validation"], conf), SaveSAECallback(conf), TopKScheduleCallback(conf)] if include_sae else [],
    optimizers=(optimizer, scheduler),
)

if include_sae:
    train_W_enc = conf['sae']['train_W_enc']
    train_b_enc = conf['sae']['train_b_enc']
    train_b_dec = conf['sae']['train_b_dec']

    for name, p in model.named_parameters():
        if "sae" in name: 
            if train_W_enc and 'W_enc.weight' in name: assert p.requires_grad
            elif train_b_enc and 'W_enc.bias' in name: assert p.requires_grad
            elif train_b_dec and 'W_dec.bias' in name: assert p.requires_grad
            else: assert not p.requires_grad
        if "lora" in name: assert p.requires_grad

else:
    for name, p in model.named_parameters():
        if "lora" in name: assert p.requires_grad


trainer.train()
