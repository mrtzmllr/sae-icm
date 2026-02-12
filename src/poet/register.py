from transformers import AutoConfig, AutoModelForCausalLM

from src.poet.model_config import Gemma2SAEConfig
from src.poet.insert_sae import Gemma2SAEForCausalLM

def register_automodel():
    # we need to call this when using AutoModelForCausalLM
    AutoModelForCausalLM.register(Gemma2SAEConfig, Gemma2SAEForCausalLM)

def register_autoconfig():
    AutoConfig.register("gemma2-sae", Gemma2SAEConfig)