from transformers import Gemma2Config, LlamaConfig

class Gemma2SAEConfig(Gemma2Config):
    model_type = "gemma2-sae"

    def __init__(
        self,
        sae_layer: int | None = None,
        return_z: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sae_layer = sae_layer
        self.return_z = return_z


class LlamaSAEConfig(LlamaConfig):
    model_type = "llama-sae"

    def __init__(
        self,
        sae_layer: int | None = None,
        return_z: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sae_layer = sae_layer
        self.return_z = return_z


def _is_llama(conf):
    return "llama" in conf["model"]["name"].lower()


def retrieve_config(conf):
    model_path = conf["eval"]["model_path"]
    sae_layer = conf["sae"]["sae_layer"]
    return_z = conf["sae"]["finetuning"]["return_z"]
    config_class = LlamaSAEConfig if _is_llama(conf) else Gemma2SAEConfig
    return config_class.from_pretrained(model_path, sae_layer=sae_layer, return_z=return_z)
