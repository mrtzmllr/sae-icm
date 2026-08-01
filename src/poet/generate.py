import torch
from transformers import AutoTokenizer
from peft import PeftModel

from src.poet.config import load_config
from src.poet.insert_sae import Gemma2SAEForCausalLM, insert_sae
from src.poet.insert_sae_llama import LlamaSAEForCausalLM
from src.poet.argparse import parse_args
from src.poet.model_config import Gemma2SAEConfig, LlamaSAEConfig

all_args = parse_args()

conf = load_config(all_args, run_eval=True)
model_name = conf["model"]["name"]
dataset_name = conf["eval"]["dataset"]

model_path = conf["eval"]["model_path"]

# SAE-class based on model
is_llama = "llama" in model_name.lower()
sae_cls = LlamaSAEForCausalLM if is_llama else Gemma2SAEForCausalLM
sae_config_cls = LlamaSAEConfig if is_llama else Gemma2SAEConfig

model_config = sae_config_cls.from_pretrained(
    model_name,
    sae_layer=conf["sae"]["sae_layer"],
    return_z=conf["sae"]["finetuning"]["return_z"],
)

# retrieve stage-1 pre-trained model
model = sae_cls.from_pretrained(model_name, config=model_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if is_llama and tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = insert_sae(model=model, conf=conf)              # combine stage-1 decoder and stage-2 encoder/biases
model = PeftModel.from_pretrained(model, model_path)    # attach low-rank adapter to LM
model = model.merge_and_unload()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(model_path)

p = "What's the capital of Germany?"
prompts = [f"Problem:\n{p}\n\nSolution:\n"]

for prompt in prompts:
    pp = f"Problem:\n{p}\n\nSolution:\n"

    enc = tokenizer(
        pp,
        truncation=True,
        padding="max_length",
        padding_side = "left",
        max_length=512,
        return_tensors="pt",
    )

    input_ids = enc.input_ids
    attention_mask = enc.attention_mask

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_new_tokens=200,
            do_sample=False,
        )
    
    text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )
    
    print(text)