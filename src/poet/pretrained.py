from transformers import AutoTokenizer, AutoModelForCausalLM
from src.poet.insert_sae import Gemma2SAEForCausalLM
from src.poet.insert_sae_llama import LlamaSAEForCausalLM
from src.poet.model_config import Gemma2SAEConfig, LlamaSAEConfig
from src.poet.register import register_automodel
import os

hf_token = os.getenv("HF_TOKEN")

def retrieve_pretrained(conf):
    """Retrieve a pretrained model and tokenizer from Hugging Face."""
    model_name = conf["model"]["name"]
    is_llama = "llama" in model_name.lower()

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              token=hf_token,
                                              )

    if is_llama and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if conf["model"]["finetune_language_model"]:
        sae_cls = LlamaSAEForCausalLM if is_llama else Gemma2SAEForCausalLM
        sae_config_cls = LlamaSAEConfig if is_llama else Gemma2SAEConfig

        model_config = sae_config_cls.from_pretrained(
            model_name,
            sae_layer=conf["sae"]["sae_layer"],
            return_z=conf["sae"]["finetuning"]["return_z"],
            token=hf_token,
        )
        model_config.use_cache = False

        model = sae_cls.from_pretrained(
            model_name,
            config=model_config,
            token=hf_token,
        )

    else:
        register_automodel()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_cache=False,
            token=hf_token,
        )

    return model, tokenizer
