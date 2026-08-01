import os

from src.poet.directories import write_output_dir
from src.poet.register import register_autoconfig


def adapt_curriculum(conf):
    if conf["sae"]["top_k_curriculum"]: 
        assert conf["curriculum"]["top_k"]["start"] > conf["curriculum"]["top_k"]["end"]
        assert conf["sae"]["type"] == "top-k"
        conf["sae"]["top_k"] = conf["curriculum"]["top_k"]["start"]
    else:
        conf["curriculum"]["top_k"]["start"] = conf["sae"]["top_k"]
        conf["curriculum"]["top_k"]["end"] = conf["sae"]["top_k"]
    return conf


def assign_datasets(conf):
    if not conf["dataset"]["for_generation"] and not conf["interpretability"]["write_gsm8k_heap"]:
        conf["training"]["dataset"] = conf["dataset"]["name"]
        conf["eval"]["dataset"] = conf["dataset"]["name"]
    return conf


def supported_dataset_pairs(conf):
    if conf["interpretability"]["write_gsm8k_heap"] or conf["dataset"]["for_generation"]:
        pair = (conf["training"]["dataset"], conf["eval"]["dataset"])
        # Currently only runs on meta-math training and gsm8k evaluation
        if pair not in [("meta-math", "gsm8k")]:
            raise NotImplementedError(
                f"The interpretability and generation-eval pipelines are only implemented for (training.dataset, eval.dataset) == ('meta-math', 'gsm8k'), got {pair}."
            )
    return conf


def model_specific_parameters(conf):
    model_name = conf["model"]["name"]
    assert model_name in ["google/gemma-2-2b", "meta-llama/Llama-3.2-1B"]
    d_model = conf["model"]["d_model"]
    d_sae = conf["sae"]["d_sae"]
    sae_layer = conf["sae"]["sae_layer"]
    if model_name == "google/gemma-2-2b":
        assert d_model == 2304 and d_sae == 65536 and sae_layer == 12
    if model_name == "meta-llama/Llama-3.2-1B":
        assert d_model == 2048 and d_sae == 14336 and sae_layer == 11
    return conf


def turn_off_trainable(conf):
    if not conf["training"]["include_sae"]:
        conf["sae"]["use_orthogonal"] = False

    if not (conf["training"]["include_sae"] and conf["sae"]["any_trainable"]):
        conf['sae']['train_W_enc'] = False
        conf['sae']['train_b_enc'] = False
        conf['sae']['train_b_dec'] = False

    return conf


def assert_finetune_orthogonal(conf):
    assert not (conf["sae"]["use_finetuned"] and conf["sae"]["use_orthogonal"])
    return conf


def gradient_checkpointing(conf):
    if not conf["model"]["finetune_language_model"]:
        conf["training"]["gradient_checkpointing"] = False
    return conf


def mathematical_dataset(conf):
    if conf["dataset"]["name"] in ["meta-math", "gsm8k"]:
        conf["dataset"]["is_mathematical"] = True
    else:
        conf["dataset"]["is_mathematical"] = False
    return conf


def meta_math_logging(conf):
    if conf["dataset"]["name"] == "meta-math":
        conf["training"]["logging_steps"] = 1000
        conf["training"]["save_steps"] = 5000
    return conf


def meta_math_evaluation(conf):
    if conf["dataset"]["name"] == "meta-math":
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return conf


def model_path(conf):
    checkpoint = conf["eval"]["checkpoint"]
    model_dir = write_output_dir(conf)
    model_path = model_dir + f"/checkpoint-{checkpoint}"
    conf["eval"]["model_path"] = model_path
    return conf


def register(conf):
    # registers the Gemma2SAEConfig
    register_autoconfig()
    return conf


def return_z(conf):
    # we only return_z if running SAEonGemma2
    conf["sae"]["finetuning"]["return_z"] = not conf["model"]["finetune_language_model"]
    return conf


def set_eval_length_dataset(conf):
    if not conf["dataset"]["for_generation"] or conf["intervenability"]["do_intervene"]:
        conf["dataset"]["eval_length"] = None
    return conf