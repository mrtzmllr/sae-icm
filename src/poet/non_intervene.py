import torch
from tqdm import tqdm

from src.poet.config import load_config
from src.poet.argparse import parse_args
from src.poet.directories import write_interv_tokens_dir, write_interv_features_dir
from src.poet.prompt_template import name_map, group_dataset_map
from src.poet.load_finetuned import load_finetuned_model, load_finetuned_tokenizer

all_args = parse_args()
conf = load_config(all_args, run_eval=True)

assert not conf["intervenability"]["do_intervene"]

model_name = conf["model"]["name"]
model_path = conf["eval"]["model_path"]
interv_group = conf["intervenability"]["group"]
assert interv_group in ["names", "locations", "animals"]

model = load_finetuned_model(conf)
tokenizer = load_finetuned_tokenizer(conf)

# load_finetuned_model already inserts the SAE, merges the LoRA adapter, and moves to device.

name_dict = name_map(conf)[interv_group]
group_dataset = group_dataset_map(conf)

print(name_dict)

total_count = 0
total_drop = 0
total_include = 0

step = 0
for drop_name in tqdm(name_dict.keys()):
    dataset_dict = group_dataset(drop_name)
    
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

    model.model.sae.indices = []
    
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

    tokens_dir = write_interv_tokens_dir(conf, drop_name, include_name=None)
    tokens_pth = tokens_dir + "/tokens.pt"
    generated_ids = output_ids[:, input_ids.shape[1]:]
    torch.save(generated_ids, tokens_pth)
    
    feature_ids = model.model.sae.indices
    indices = torch.stack(feature_ids, dim=1)
    features_dir = write_interv_features_dir(conf, drop_name, include_name=None)
    features_pth = features_dir + "/features.pt"
    torch.save(feature_ids, features_pth)

    model.model.sae.step = 0