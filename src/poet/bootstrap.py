import os
import random
import numpy as np
import pandas as pd

from src.poet.config import load_config
from src.poet.argparse import parse_args
from src.poet.directories import write_overarching_statistics_dir, write_bootstrap_dir, _initial_parts
from src.poet.embeddings import preprocess_embedding, write_embeddings, center_embeddings, similarities, tril_list
from src.poet.orthogonality import bootstrap_orthogonality

all_args = parse_args()
conf = load_config(all_args, run_eval=True)


def preprocess_mean(mean_value, num_values):
    hits = num_values * mean_value
    assert len(hits) == 1
    if not num_values == 15799:
        assert round(hits.iloc[0], 8).is_integer()
    return int(round(hits.iloc[0], 8))


def preprocess_hits(hits, num_values):
    diff = num_values - hits
    column = [1] * hits + [0] * diff 
    return column


def bootstrap(conf, column):
    iterations = conf["bootstrap"]["iterations"]
    num_sampled_observations = len(column)

    bootstrapped_means = []

    for _ in range(iterations):
        sampled = random.choices(list(column), k = num_sampled_observations)
        mean_sampled = sum(sampled) / len(sampled)
        bootstrapped_means.append(mean_sampled)
    
    q025 = np.quantile(bootstrapped_means, 0.025)
    q975 = np.quantile(bootstrapped_means, 0.975)
    mn = sum(column) / len(column)

    lower, upper = 2 * mn - q975, 2 * mn - q025 

    return lower, mn, upper


def intervention_metric_path(conf, ortho):
    insertion_value = conf["intervenability"]["insertion_value"]
    interv_group = conf["intervenability"]["group"]
    replacement_group = conf["intervenability"]["replacement_group"]

    parts = _initial_parts(conf)
    assert conf["training"]["dataset"] == "meta-math", "For consistency, we store the metrics under the training dataset path."
    
    parts.append("top-k/lm_around_ft_sae/top_k20")
    parts.append(f"ortho{ortho}")
    parts.append("trainWencbencbdec/lr5e-05/eval/intervenability")
    parts.append(f"insertion_value{insertion_value}")
    if interv_group != "names":
        parts.append(f"group{interv_group}")
    if interv_group != replacement_group:
        parts.append(f"replacement{replacement_group}")
    
    metric_dir = "/".join(parts)
    metric_path = metric_dir + "/" + "metrics.csv"
    return metric_path


def preprocess_bootstrap(conf, ortho):
    metric = conf["bootstrap"]["metric"]
    
    assert isinstance(metric, str)
    assert metric in ["dead_features", "math_eval", "interp_score", "interv_eval", "interv_include", "interv_drop", "embeddings", "orthogonality_raw", "orthogonality_norm", "orthogonality_mean_cos", "orthogonality_max_cos", "orthogonality_loss"]

    stats_dir = write_overarching_statistics_dir(conf)

    if metric == "dead_features":
        metric_path = stats_dir + "/dead_features.csv"
        df = pd.read_csv(metric_path)
        df["orthogonality_lambda"] = df["orthogonality_lambda"].astype(float)
        mean_value = df["dead_features_fraction"][df["orthogonality_lambda"] == float(ortho)]
        num_values = 15799 # len(evalset)
        hits = preprocess_mean(mean_value, num_values)
        column = preprocess_hits(hits, num_values)

    elif metric == "math_eval":
        eval_length = conf["dataset"]["eval_length"]
        metric_path = stats_dir + f"/math_eval/gsm8k/eval_length{eval_length}/metrics.csv"
        
        df = pd.read_csv(metric_path)
        df["orthogonality_lambda"] = df["orthogonality_lambda"].astype(float)
        mean_value = df["accuracy"][df["orthogonality_lambda"] == float(ortho)]
        num_values = 500
        hits = preprocess_mean(mean_value, num_values)
        column = preprocess_hits(hits, num_values)

    elif metric == "interp_score":
        num_total_evaluations = conf["interpretability"]["total_evaluations"]
        num_total_snippets = conf["interpretability"]["total_snippets"]
        num_correct_snippets = conf["interpretability"]["correct_snippets"]
        metric_path = stats_dir + f"/interpretability/score/total_evaluations{num_total_evaluations}/num_total_snippets{num_total_snippets}/num_correct_snippets{num_correct_snippets}/metrics.csv"
        
        df = pd.read_csv(metric_path)
        df["orthogonality_lambda"] = df["orthogonality_lambda"].astype(float)
        mean_value = df["average_overlap"][df["orthogonality_lambda"] == float(ortho)]

        hits = preprocess_mean(mean_value, num_values = num_total_evaluations)
        column = preprocess_hits(hits, num_values = num_total_evaluations)
        
    elif metric == "interv_eval":
        metric_path = intervention_metric_path(conf, ortho)
        df = pd.read_csv(metric_path)
        num_values = df["total_evaluations"].iloc[0]
        hits = df["total_count"].iloc[0]
        column = preprocess_hits(hits, num_values)

    elif metric == "interv_include":
        
        metric_path = intervention_metric_path(conf, ortho)
        df = pd.read_csv(metric_path)
        num_values = df["total_evaluations"].iloc[0]
        hits = df["total_include"].iloc[0]
        column = preprocess_hits(hits, num_values)

    elif metric == "interv_drop":
        metric_path = intervention_metric_path(conf, ortho)
        df = pd.read_csv(metric_path)
        num_values = df["total_evaluations"].iloc[0]
        hits = df["total_drop"].iloc[0]
        column = preprocess_hits(hits, num_values)

    elif metric == "embeddings":
        conf["sae"]["finetuning"]["orthogonality_lambda"] = ortho
        model, explanations, feature_indices = preprocess_embedding(conf)

        random.shuffle(feature_indices)

        embeddings = write_embeddings(conf, model, explanations, feature_indices)
        embeddings = center_embeddings(conf, embeddings)
        tril = similarities(model, embeddings)
        nonzero_list = tril_list(tril)
        column = [element[0] for element in nonzero_list]

    elif metric == "orthogonality_raw":
        conf["bootstrap"]["cos"] = False
        conf["bootstrap"]["welch"] = True
        conf["bootstrap"]["loss"] = False
        conf["sae"]["finetuning"]["orthogonality_lambda"] = ortho
        metrics = bootstrap_orthogonality(conf)
        print("bound:", metrics["bound"])
        column = metrics["mean_raw"]

    elif metric == "orthogonality_norm":
        conf["bootstrap"]["cos"] = False
        conf["bootstrap"]["welch"] = True
        conf["bootstrap"]["loss"] = False
        conf["sae"]["finetuning"]["orthogonality_lambda"] = ortho
        metrics = bootstrap_orthogonality(conf)
        print("bound:", metrics["bound"])
        column = metrics["mean_norm"]

    elif metric == "orthogonality_mean_cos":
        conf["bootstrap"]["cos"] = True
        conf["bootstrap"]["welch"] = False
        conf["bootstrap"]["loss"] = False
        conf["sae"]["finetuning"]["orthogonality_lambda"] = ortho
        metrics = bootstrap_orthogonality(conf)
        column = metrics["mean_cos"]

    elif metric == "orthogonality_max_cos":
        conf["bootstrap"]["cos"] = True
        conf["bootstrap"]["welch"] = False
        conf["bootstrap"]["loss"] = False
        conf["sae"]["finetuning"]["orthogonality_lambda"] = ortho
        metrics = bootstrap_orthogonality(conf)
        column = metrics["max_cos"]
    
    elif metric == "orthogonality_loss":
        conf["bootstrap"]["cos"] = False
        conf["bootstrap"]["welch"] = False
        conf["bootstrap"]["loss"] = True
        conf["sae"]["finetuning"]["orthogonality_lambda"] = ortho
        losses = bootstrap_orthogonality(conf)
        loss = losses["loss"]
        column = [l.detach().numpy() for l in loss]
        
    return column


def bootstrap_pipeline(conf):
    orthogonalities = conf["bootstrap"]["orthogonalities"]
    interv_group = conf["intervenability"]["group"]
    replacement_group = conf["intervenability"]["replacement_group"]

    for ortho in orthogonalities:
        column = preprocess_bootstrap(conf, ortho)
        lwr, mn, upr = bootstrap(conf, column)
        
        if conf["eval"]["write_eval_file"]:
            bootstrap_dir = write_bootstrap_dir(conf)
            metric = conf["bootstrap"]["metric"]
            insertion_value = conf["intervenability"]["insertion_value"]
            file_path = bootstrap_dir + f"/{metric}.csv"
            if insertion_value != 200:
                file_path = bootstrap_dir + f"/{metric}_insertion{insertion_value}.csv"

            if "interv" in metric and interv_group != "names":
                file_path = bootstrap_dir + f"/{metric}_{interv_group}.csv"
                if insertion_value != 200:
                    file_path = bootstrap_dir + f"/{metric}_{interv_group}_insertion{insertion_value}.csv"

            # metric_parts connect everything after the final /, i.e., the _
            metric_parts = [f"{metric}"]
            if "interv" in metric and interv_group != "names":
                metric_parts.append(f"{interv_group}")
            if "interv" in metric and interv_group != replacement_group:
                metric_parts.append(f"replace{replacement_group}")
            if insertion_value != 200:
                metric_parts.append(f"insertion{insertion_value}")
            metric_path = "_".join(metric_parts)

            file_path = bootstrap_dir + "/" + metric_path + ".csv"

            new_row = {"orthogonality_lambda": ortho, "lower": lwr, "mean": mn, "upper": upr}
            df_new = pd.DataFrame([new_row])

            if os.path.exists(file_path):
                df_new.to_csv(file_path, mode='a', header=False, index=False)
            else:
                df_new.to_csv(file_path, mode='w', header=True, index=False)
            
            print(f"Confidence interval written successfully for {metric} at orthogonality {ortho}: {lwr}, {mn}, {upr}!")

        else:
            print( ortho , lwr , mn , upr )


if __name__ == "__main__":
    bootstrap_pipeline(conf)
