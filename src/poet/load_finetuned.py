import torch
from transformers import AutoTokenizer
from peft import PeftModel

from src.poet.insert_sae import Gemma2SAEForCausalLM, insert_sae
from src.poet.insert_sae_llama import LlamaSAEForCausalLM
from src.poet.model_config import Gemma2SAEConfig, LlamaSAEConfig


def _is_llama(conf):
    return "llama" in conf["model"]["name"].lower()


def load_finetuned_tokenizer(conf):
    tokenizer = AutoTokenizer.from_pretrained(conf["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_finetuned_model(conf, to_device=True):
    model_name = conf["model"]["name"]
    model_path = conf["eval"]["model_path"]
    is_llama = _is_llama(conf)

    sae_cls = LlamaSAEForCausalLM if is_llama else Gemma2SAEForCausalLM
    sae_config_cls = LlamaSAEConfig if is_llama else Gemma2SAEConfig

    insertion_layer = conf["sae"]["insertion_layer"]
    splice_layer = insertion_layer if insertion_layer is not None else conf["sae"]["sae_layer"]

    model_config = sae_config_cls.from_pretrained(
        model_name,
        sae_layer=splice_layer,
        return_z=conf["sae"]["finetuning"]["return_z"],
    )

    model = sae_cls.from_pretrained(model_name, config=model_config)
    model = insert_sae(model=model, conf=conf)

    if conf["model"]["use_lora"]:
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()

    if to_device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    return model
