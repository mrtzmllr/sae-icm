import yaml

from src.poet.argparse import handle_unknown_args
from src.poet.config_adaptations import adapt_curriculum, assign_datasets, turn_off_trainable, assert_finetune_orthogonal, gradient_checkpointing, mathematical_dataset, meta_math_logging, meta_math_evaluation, model_path, register, return_z, set_eval_length_dataset

def load_config(all_args, run_eval = False):
    args, unknown = all_args
    
    yaml_file = args.conf

    with open(yaml_file, "r") as f:
        conf = yaml.safe_load(f)

    conf = handle_unknown_args(unknown, conf)
    conf = adapt_config(conf)

    if run_eval: conf["model"]["run_eval"] = True
    return conf


def adapt_config(conf):
    conf = adapt_curriculum(conf)
    conf = assign_datasets(conf)
    conf = mathematical_dataset(conf)
    conf = turn_off_trainable(conf)
    conf = assert_finetune_orthogonal(conf)
    conf = gradient_checkpointing(conf)
    conf = meta_math_logging(conf)
    conf = meta_math_evaluation(conf)
    conf = model_path(conf)
    conf = register(conf)
    conf = return_z(conf)
    conf = set_eval_length_dataset(conf)
    return conf