import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.poet.config import load_config
from src.poet.argparse import parse_args
from src.poet.directories import write_feature_storage_dir
from src.poet.prompt_template import create_prompt_dataset, dataset_to_inputs, add_to_explanations_dictionary


all_args = parse_args()
conf = load_config(all_args, run_eval=True)

assert conf["interpretability"]["write_gsm8k_heap"], "We require this for the right GSM8K dataset to be retrieved during interp."

llama_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_name)
model = AutoModelForCausalLM.from_pretrained(
    llama_name,
    device_map="auto",
    # reduce from fp32 to reduce required computed
    dtype=torch.bfloat16,
    use_cache=True,
)

prompt_dataset, final_feature_indices = create_prompt_dataset(conf)

llama_enc = dataset_to_inputs(llama_tokenizer, prompt_dataset)

input_ids_tensor = llama_enc["input_ids"].to(model.device)
attention_mask_tensor = llama_enc["attention_mask"].to(model.device)

explanations_dict = {}

if len(input_ids_tensor) > 100:
    print(len(input_ids_tensor))
    assert (len(input_ids_tensor) / 100).is_integer()
    num_batches = int(len(input_ids_tensor) / 100)
    for b in tqdm(range(num_batches)):
        input_sub_tensor = input_ids_tensor[(b * 100):((b+1) * 100), :]
        attention_sub_tensor = attention_mask_tensor[(b * 100):((b+1) * 100), :]
        feature_sub_indices = final_feature_indices[(b * 100):((b+1) * 100)]
        explanations_dict = add_to_explanations_dictionary(model, llama_tokenizer, input_sub_tensor, attention_sub_tensor, feature_sub_indices, explanations_dict)

else: explanations_dict = add_to_explanations_dictionary(model, llama_tokenizer, input_ids_tensor, attention_mask_tensor, final_feature_indices, explanations_dict)


mathematical_system_prompt = conf["interpretability"]["mathematical_system_prompt"]
json_dir = write_feature_storage_dir(conf)
if mathematical_system_prompt:
    json_path = json_dir + "/feature_explanations.jsonl"
else: 
    json_path = json_dir + "/real_world_feature_explanations.jsonl"


if conf["eval"]["write_eval_file"]:
    with open(json_path, "w") as f:
        for ftr_index, decoded in explanations_dict.items():
            json_line = {
                "feature_idx": ftr_index,
                "explanation": decoded
            }
            f.write(json.dumps(json_line) + "\n")
    print("Explanations written to .jsonl file!")