import os

import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from peft import PeftModel
from safetensors.torch import load_file

from src.poet.argparse import parse_args
from src.poet.config import load_config
from src.poet.insert_sae import Gemma2SAEForCausalLM
from src.poet.insert_sae_llama import LlamaSAEForCausalLM
from src.poet.load_finetuned import _is_llama, load_finetuned_tokenizer
from src.poet.model_config import Gemma2SAEConfig, LlamaSAEConfig
from src.poet.prompt_template import jerrys_dataset, name_map
from src.poet.sae import TopKSAE

hf_name = "moritzmiller"


def hub_repo_id(conf):
    slug = conf["model"]["name"].split("/")[-1].lower()
    orthogonality_lambda = conf["sae"]["finetuning"]["orthogonality_lambda"]
    return hf_name + "/sae-icm-final-checkpoint-" + slug + "-" + orthogonality_lambda


def load_hub_model(conf, to_device=True):
    model_name = conf["model"]["name"]
    repo = hub_repo_id(conf)
    is_llama = _is_llama(conf)
    hf_token = os.getenv("HF_TOKEN")

    sae_cls = LlamaSAEForCausalLM if is_llama else Gemma2SAEForCausalLM
    sae_config_cls = LlamaSAEConfig if is_llama else Gemma2SAEConfig

    insertion_layer = conf["sae"]["insertion_layer"]
    splice_layer = insertion_layer if insertion_layer is not None else conf["sae"]["sae_layer"] # not required for COLM experiments

    model_config = sae_config_cls.from_pretrained(
        model_name,
        sae_layer=splice_layer,
        return_z=conf["sae"]["finetuning"]["return_z"],
        token=hf_token,
    )

    model = sae_cls.from_pretrained(model_name, config=model_config, token=hf_token)

    sae = TopKSAE(conf=conf)
    sae.to(model.device, model.dtype)
    state_path = hf_hub_download(repo_id=repo, filename="sae_state.safetensors", token=hf_token)
    sae.load_state_dict(load_file(state_path))
    model.model.sae = sae

    if conf["model"]["use_lora"]:
        model = PeftModel.from_pretrained(model, repo, token=hf_token)
        model = model.merge_and_unload()

    if to_device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    return model


if __name__ == "__main__":
    load_dotenv()

    conf = load_config(parse_args(), run_eval=True)
    print(f"loading https://huggingface.co/{hub_repo_id(conf)}")

    model = load_hub_model(conf)
    model.eval()
    tokenizer = load_finetuned_tokenizer(conf)

    # Figure 1: swap the feature of the first name "Mike" for the aqua feature
    is_llama = _is_llama(conf)
    drop_name = "Mike" if not is_llama else "John"
    drop_feature_idx = name_map(conf)["names"][drop_name]
    include_feature_idx = conf["interpretability"]["feature_index"] if not is_llama else name_map(conf)["names"]["James"]
    # for llama-3.2-1b experiment swap "John" for "James"
    # note that for inference on the llama models, the optimal insertion value is 10-20
    # set the flag --intervenability.insertion_value 20

    sentence = jerrys_dataset(drop_name)["2"]["sentence"][0]
    prompt = f"Problem:\n{sentence}\n\nSolution:\n"
    enc = tokenizer(prompt, return_tensors="pt")

    def run_generation():
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=enc.input_ids.to(model.device),
                attention_mask=enc.attention_mask.to(model.device),
                max_new_tokens=400,
                do_sample=False,
            )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("\nStandard Generation:\n" + run_generation())

    model.model.sae.do_intervene = True
    model.model.sae.intervention_indices = {
        "drop": [drop_feature_idx],
        "include": [include_feature_idx],
    }

    print("\nPost-Intervention Generation:\n" + run_generation())
