from transformers import AutoTokenizer, AutoModelForCausalLM
from src.poet.insert_sae import Gemma2SAEForCausalLM
from src.poet.register import register_automodel
import os

hf_token = os.getenv("HF_TOKEN")

def retrieve_pretrained(conf):
    """Retrieve a pretrained model and tokenizer from Hugging Face."""
    model_name = conf["model"]["name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              token=hf_token,
                                              )
    
    if conf["model"]["finetune_language_model"]:
        # requires_trust = model_name.startswith(("Qwen", "Qwen2", "Yi", "DeepSeek"))
        # model = AutoModelForCausalLM.from_pretrained(
        model = Gemma2SAEForCausalLM.from_pretrained(
            model_name,
            # trust_remote_code=requires_trust,
            use_cache=False,
            token=hf_token,
        )

    else:
        register_automodel()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # trust_remote_code=requires_trust,
            use_cache=False,
            token=hf_token,
        )

    return model, tokenizer