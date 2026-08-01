import torch

from poet.config import load_config
from poet.argparse import parse_args
from poet.prompt_template import jerrys_dataset
from poet.dataset import retrieve_dataset
from poet.load_finetuned import load_finetuned_model, load_finetuned_tokenizer


all_args = parse_args()
conf = load_config(all_args, run_eval=True)

# assert conf["intervenability"]["do_intervene"]

model_name = conf["model"]["name"]
model_path = conf["eval"]["model_path"]

model = load_finetuned_model(conf)
tokenizer = load_finetuned_tokenizer(conf)

dataset = retrieve_dataset(conf, tokenizer, split="test")
doc_idxs = [211] 

model.model.sae.intervention_indices = {
    "include": [7925],
    "drop": [161],
}

example = dataset["test"][doc_idxs[0]]

dataset_dict = jerrys_dataset("John")

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


print(texts[2])
print(texts[9])