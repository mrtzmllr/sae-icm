import os


def write_output_dir(conf, train = True):
    parts = _initial_parts(conf)
    
    include_sae = conf["training"]["include_sae"]
    sae_type = conf['sae']['type']
    lr = conf["training"]["learning_rate"]
    top_k = conf["sae"]["top_k"]
    if include_sae:
        parts.append(sae_type)
        if conf["sae"]["disturb"]:
            parts.append("disturb")
        if conf["sae"]["use_orthogonal"]:
            parts.append("orthogonal")
            sae_weights = conf["sae"]["sae_weights"]
            parts.append(sae_weights)
            parts = _include_weights_in_pathname(conf, parts)
            parts.append(f"lr{lr}")
            parts.append(f"top_k{top_k}")
            if conf["sae"]["top_k_curriculum"]:
                parts.append("curriculum")
                end = conf["curriculum"]["top_k"]["end"]
                steps = conf["curriculum"]["top_k"]["steps"]
                red = conf["curriculum"]["top_k"]["reduction_factor"]
                parts.append(f"end{end}")
                parts.append(f"steps{steps}")
                parts.append(f"red{red}")
        if conf["sae"]["use_finetuned"]:
            parts.append("lm_around_ft_sae")
            parts.append(f"top_k{top_k}")
            orthogonality_lambda = conf["sae"]["finetuning"]["orthogonality_lambda"]
            parts.append(f"ortho{orthogonality_lambda}")
            parts = _include_weights_in_pathname(conf, parts)
            parts.append(f"lr{lr}")
        if conf["sae"]["binary"]["use_binary"]:
            parts.append("binary")
            temp = conf["sae"]['binary']['temperature']
            parts.append(f"temp{temp}")
    else:
        parts.append("no-sae")
    if train:
        parts.append("train")
    else:
        parts.append("eval")
    output_dir = "/".join(parts)

    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _include_weights_in_pathname(conf, parts: list):
    train_W_enc = conf["sae"]["train_W_enc"]
    train_b_enc = conf["sae"]["train_b_enc"]
    train_b_dec = conf["sae"]["train_b_dec"]
    if train_W_enc:
        if train_b_enc:
            if train_b_dec:
                parts.append("trainWencbencbdec")
            else:
                parts.append("trainWencbenc")
        elif train_b_dec:
            parts.append("trainWencbdec")
    else:
        if train_b_enc:
            if train_b_dec:
                parts.append("trainbencbdec")
            else:
                parts.append("trainbenc")
        elif train_b_dec:
            parts.append("trainbdec")
    return parts


def _initial_parts(conf):
    model_name = conf["model"]["name"]
    dataset_name = conf["training"]["dataset"]
    
    cache_dir = os.getenv("CACHE_DIR")
    assert conf["sae"]["type"] == "top-k"
    parts = [
        cache_dir, 
        model_name.replace("/", "-"), 
        dataset_name,
        ]
    return parts


def write_sae_dir(conf, train = True):
    # train determines which directory we choose
    # we cannot set a global flag as during any evaluation run we need to first call the trained model (/train/) and then save metrics to /eval/
    parts = _initial_parts(conf)
   
    sae_type = conf['sae']['type']
    parts.append(sae_type)
    parts.append("sae_finetuning")
    
    orthogonality_lambda = conf["sae"]["finetuning"]["orthogonality_lambda"]
    parts.append(orthogonality_lambda)

    lr = conf["training"]["learning_rate"]
    top_k = conf["sae"]["top_k"]
    parts.append(f"lr{lr}")
    parts.append(f"top_k{top_k}")
    
    if conf["sae"]["binary"]["use_binary"]:
        parts.append("binary")
        temp = conf["sae"]['binary']['temperature']
        parts.append(f"temp{temp}")

    if train:
        parts.append("train")
    else:
        parts.append("eval")
    sae_dir = "/".join(parts)

    if not os.path.exists(sae_dir): os.makedirs(sae_dir, exist_ok=True)
    return sae_dir


def finetuned_sae_states_dir(conf):
    sae_type = conf['sae']['type']
    dataset = conf["dataset"]["name"]
    project_path = os.getenv('PROJECT_PATH')
    parts = [
        project_path, 
        "orthogonal-weights",
        dataset,
        sae_type,
        "finetuned"
        ]
    top_k = conf["sae"]["top_k"]
    lr = conf["training"]["learning_rate"]
    
    if conf["sae"]["use_finetuned"]:
        parts.append("lm_around_ft_sae")
        parts.append(f"top_k{top_k}")
        orthogonality_lambda = conf["sae"]["finetuning"]["orthogonality_lambda"]
        parts.append(f"ortho{orthogonality_lambda}")
        parts = _include_weights_in_pathname(conf, parts)
        parts.append(f"lr{lr}")
    else: parts.append("ft_during_lm_ft")
    if conf["sae"]["binary"]["use_binary"]:
        parts.append("binary")
        temp = conf["sae"]['binary']['temperature']
        parts.append(f"temp{temp}")
    else: parts.append("continuous")

    ft_dir = "/".join(parts)

    assert not conf["sae"]["top_k_curriculum"]

    if not os.path.exists(ft_dir): os.makedirs(ft_dir, exist_ok=True)
    return ft_dir


def off_the_shelf_sae_states_dir(conf):
    sae_type = conf['sae']['type']
    project_path = os.getenv('PROJECT_PATH')
    weights_dir = project_path + "/orthogonal-weights/" + sae_type + "/off-the-shelf/"
    if not os.path.exists(weights_dir): os.makedirs(weights_dir)
    return weights_dir


def plotting_dir():
    project_path = os.getenv('PROJECT_PATH')
    return os.path.join(project_path, "plots")


def write_feature_storage_dir(conf):
    parts = []
    output_dir = write_output_dir(conf, train=False)
    parts.append(output_dir)
    parts.append("interpretability")
    parts.append("feature_storage")
    max_elements = conf["eval"]["max_features_stored"]
    parts.append(f"maxelem{max_elements}")

    interp_dir = "/".join(parts)
    if not os.path.exists(interp_dir): os.makedirs(interp_dir, exist_ok=True)
    return interp_dir


def write_overarching_statistics_dir():
    stats_dir = "statistics"
    return stats_dir


def write_bootstrap_dir():
    stats_dir = write_overarching_statistics_dir()
    return stats_dir + "/bootstrap"


def write_intervenability_dir(conf):
    parts = []
    output_dir = write_output_dir(conf, train = False)
    parts.append(output_dir)
    parts.append("intervenability")
    insertion_value = conf["intervenability"]["insertion_value"]
    parts.append(f"insertion_value{insertion_value}")

    interv_dir = "/".join(parts)
    if not os.path.exists(interv_dir): os.makedirs(interv_dir, exist_ok=True)
    return interv_dir
