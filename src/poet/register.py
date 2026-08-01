from transformers import AutoConfig, AutoModelForCausalLM

from src.poet.model_config import Gemma2SAEConfig, LlamaSAEConfig
from src.poet.insert_sae import Gemma2SAEForCausalLM
from src.poet.insert_sae_llama import LlamaSAEForCausalLM

def register_automodel():
    # we need to call this when using AutoModelForCausalLM
    AutoModelForCausalLM.register(Gemma2SAEConfig, Gemma2SAEForCausalLM)
    AutoModelForCausalLM.register(LlamaSAEConfig, LlamaSAEForCausalLM)

def register_autoconfig():
    AutoConfig.register("gemma2-sae", Gemma2SAEConfig)
    AutoConfig.register("llama-sae", LlamaSAEConfig)
