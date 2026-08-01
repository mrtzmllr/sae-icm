import torch

from src.poet.config import load_config
from src.poet.argparse import parse_args
from src.poet.prompt_template import jerrys_dataset, name_map, alternative_names
from src.poet.load_finetuned import load_finetuned_model, load_finetuned_tokenizer


all_args = parse_args()
conf = load_config(all_args, run_eval=True)

assert conf["intervenability"]["do_intervene"]

model_name = conf["model"]["name"]
model_path = conf["eval"]["model_path"]

model = load_finetuned_model(conf)
tokenizer = load_finetuned_tokenizer(conf)

# Some found GSM8K features
# Jerry [100, 595, 1243, 211, 975, 184, 887]
# 59158 (Jason) 11681 (Mike) 42983 (Jacob / Jake) 44109 (Jerry / Jerome) 37532 (James) 22710 (Robert) 28504 (Jordan) 38384 (Jackson) 26534 (Paul) 28951 (David) 26342 (Andrew) 17911 (Gary / Garre)
# 1e-04 27076 (aqua) 47299 59158 (Jason) 11681 (Mike) 
# 1e-06 42983 (Jacob) 44109 (Jerry / Jerome) 9394 (sq footage) 21904 (animal-related / breed) 62849 (waiting / just a comma (,)) 31472 (counting in set / INCT) 58633 (percentage change / https) 7253 (commute or travel / to find out how many) 8912 (time and distance / computes 20/3) 21074 (doubling / emotions) 6664 (fill containers / this one actually changes the computation) 49707 (group outings / ', then') 39177 (limits and boundaries / 'on average') 12547 (geometric and vector operations or Pythagoras / changes calculation profoundly) 30395 (doubling / 'of course') 27922 (exponentiation / złoty and exponentiates at later position) 7608 (percentage-based discounts / ضايا) 23951 (iterative grid construction / taglio and wrong computation) 16697 (financial compensation / every 4 sheep Charleston has) 60844 (grouped into cartons or crates / styleType) 64650 (durations / thenГеографияCharleston)
# 1e-08 33368 40885 9155 57573 

name_dict = name_map(conf)
alt_dict = alternative_names()

drop_name = "Mike"
include_name = "John"

drop_feature_idx = name_dict["names"][drop_name]
include_feature_idx = 27076 # aqua feature # name_dict["names"][include_name]

model.model.sae.intervention_indices = {
    "include": [include_feature_idx],
    "drop": [drop_feature_idx],
}

dataset_dict = jerrys_dataset(drop_name)

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