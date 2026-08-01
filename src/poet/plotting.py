from src.poet.directories import write_output_dir, plotting_dir, write_bootstrap_dir
from src.poet.argparse import parse_args
from src.poet.config import load_config

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import seaborn as sns
import matplotlib.pyplot as plt


def lambda_to_scientific(lam):
    lam = float(lam)
    if lam == 0:
        return r"$0$"
    exponent = int(round(np.log10(lam)))
    return rf"$10^{{{exponent}}}$"


def plot_orthogonality(conf):
    d_sub_decoder = conf["sae"]["finetuning"]["d_sub_decoder"]
    welch_k = conf["sae"]["finetuning"]["welch_k"]

    ortho_lambdas = list(reversed(conf["bootstrap"]["orthogonalities"]))
    means = []

    for o in ortho_lambdas:
        conf["sae"]["finetuning"]["orthogonality_lambda"] = o

        ortho_dir = write_output_dir(conf, train = False)
        ortho_dir += "/orthogonality/"
        ortho_dir += f"d_sub_dec{d_sub_decoder}/"
        ortho_dir += f"welch_k{welch_k}/"
        ortho_file = ortho_dir + "metrics.csv"

        df = pd.read_csv(ortho_file)

        means.append(df["mean_cos"].iloc[0])

    plt_dir = plotting_dir(conf)
    plt_file = plt_dir + "/orthogonality.pdf"

    labels = [lambda_to_scientific(o) for o in ortho_lambdas]
    plt.plot(labels, means)
    plt.xlabel(r"$\mathbf{\lambda}$")
    plt.ylabel("Mean Cosine Similarity")
    plt.savefig(plt_file, bbox_inches="tight")
    plt.close()

    print("Orthogonality plot created!")



def colors(num_colors = 3):
    if num_colors > 4: return ["#DF232C"] * num_colors
    if num_colors == 4: return ["#F1B50E", "#92BCEA", "#0061AC", "#DF232C"]
    return ["#F1B50E", "#0061AC", "#DF232C"]


def pre_processing_plotting(conf):
    
    bootstrap_dir = write_bootstrap_dir(conf)
    metric = conf["bootstrap"]["metric"]
    interv_group = conf["intervenability"]["group"]
    replacement_group = conf["intervenability"]["replacement_group"]
    insertion_value = conf["intervenability"]["insertion_value"]

    metric_parts = [f"{metric}"]
    if "interv" in metric and interv_group != "names":
        metric_parts.append(f"{interv_group}")
    if "interv" in metric and interv_group != replacement_group:
        metric_parts.append(f"replace{replacement_group}")
    if insertion_value != 200:
        metric_parts.append(f"insertion{insertion_value}")
    file_path = bootstrap_dir + "/" + "_".join(metric_parts) + ".csv"

    FONT_SIZE = 14
    TICK_SIZE = 12

    sns.set_style("white")
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "axes.linewidth": 2,
        "xtick.major.width": 2,
        "ytick.major.width": 2,
    })

    df = pd.read_csv(file_path)

    df = df.iloc[::-1].reset_index(drop=True)
    
    print(df)

    mean = df["mean"].values
    lower = df["lower"].values
    upper = df["upper"].values
        
    yerr_upper = upper - mean
    yerr_lower = mean - lower

    lambdas = df["orthogonality_lambda"].astype(float).values

    if conf["plotting"]["mask"]:
        mask_values = conf["plotting"]["mask_values"]
        mask = np.array(mask_values)
        mask = mask.astype(int)

        x = np.arange(len(mask))

        return x, mean[mask], (lower[mask], upper[mask]), (yerr_lower[mask], yerr_upper[mask]), lambdas[mask]

    else:
        x = np.arange(len(lambdas))

        return x, mean, (lower, upper), (yerr_lower, yerr_upper), lambdas


def y_label(conf):
    metric = conf["bootstrap"]["metric"]

    label_dict = {
        "dead_features": "Fraction of Dead Features",
        "math_eval": "Accuracy",
        "interp_score": "Interpretability Score",
        "interv_eval": "Accuracy",
        "interv_include": "Correctly Included Indices",
        "interv_drop": "Mistakenly Non-Dropped Indices",
        "embeddings": "Average Cosine Similarity",
        "orthogonality_raw": "Mean Similarity",
        "orthogonality_norm": "Mean Similarity",
        "orthogonality_mean_cos": "Mean Cosine Similarity",
        "orthogonality_max_cos": "Max Cosine Similarity",
        "orthogonality_loss": "Orthogonality Evaluation Loss"
    }
    return label_dict[metric]


def bar_plot(conf, x, mean, yerr, lambdas):

    yerr_lower, yerr_upper = yerr
    metric = conf["bootstrap"]["metric"]
    bar_width = 0.7
    colours= colors(len(x))

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        x,
        mean,
        width=bar_width,
        color=colours[:len(x)],
        edgecolor="black",
        linewidth=1.2,
        yerr=[yerr_lower, yerr_upper],
        capsize=6,
        error_kw=dict(lw=2, capthick=2)
    )

    ax.set_xticks(x)

    labels = [lambda_to_scientific(l) for l in lambdas]
    ax.set_xticklabels(labels)

    ax.set_xlabel(r"$\mathbf{\lambda}$")
    print(conf["bootstrap"]["metric"])
    
    ax.set_ylabel(y_label(conf))
    ax.tick_params(axis="both", width=2)

    if conf["bootstrap"]["metric"] == "interp_score":
        ax.set_ylim(0, 0.6)
    elif conf["bootstrap"]["metric"] == "embeddings":
        ax.set_ylim(0, 0.7)
    elif "interv" in conf["bootstrap"]["metric"]:
        if conf["model"]["name"] == "google/gemma-2-2b":
            if conf["intervenability"]["group"] == "names":
                ax.set_ylim(0.4, 0.8)
                ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
            else:
                ax.set_ylim(0.3, 1)
        elif conf["model"]["name"] == "meta-llama/Llama-3.2-1B":
            ax.set_ylim(0, 0.6)
    elif "orthogonality" not in conf["bootstrap"]["metric"]:
        ax.set_ylim(0, 0.8)


    sns.despine(ax=ax)
    plt.tight_layout()

    plt_dir = plotting_dir(conf)
    interv_group = conf["intervenability"]["group"]
    replacement_group = conf["intervenability"]["replacement_group"]
    insertion_value = conf["intervenability"]["insertion_value"]

    # mirrors the writer in bootstrap.py: metric_parts connect everything after the final /
    metric_parts = [f"{metric}"]
    if "interv" in metric and interv_group != "names":
        metric_parts.append(f"{interv_group}")
    if "interv" in metric and interv_group != replacement_group:
        metric_parts.append(f"replace{replacement_group}")
    if insertion_value != 200:
        metric_parts.append(f"insertion{insertion_value}")
    metric_parts.append("bar")
    plt_file = plt_dir + "/" + "_".join(metric_parts) + ".pdf"

    plt.savefig(plt_file, bbox_inches="tight")
    plt.close()



def line_plot(conf, x, mean, confidence, lambdas):
    lower, upper = confidence
    metric = conf["bootstrap"]["metric"]
    line_width = 2.5
    colours= colors()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        x,
        mean,
        color=colours[1],
        linewidth=line_width
    )

    ax.fill_between(
        x,
        lower,
        upper,
        color=colours[1],
        alpha=0.25
    )

    ax.set_xticks(x)

    if "orthogonality_raw" in conf["bootstrap"]["metric"]:
        ax.set_yticks([0.0012, 0.0014, 0.0016, 0.0018, 0.002, 0.0022])
        ax.set_yticklabels(["1.2", "1.4", "1.6", "1.8", "2.0", "2.2"])
        ax.set_ylabel(y_label(conf) + " " + r"$(10^{-3})$")
    else:
        ax.set_ylabel(y_label(conf))
    

    labels = [lambda_to_scientific(l) for l in lambdas]
    ax.set_xticklabels(labels)

    ax.set_xlabel(r"$\mathbf{\lambda}$")
    ax.tick_params(axis="both", width=2)

    sns.despine(ax=ax)
    plt.tight_layout()
    plt_dir = plotting_dir(conf)
    plt_file = plt_dir + f"/{metric}_line.pdf"
    plt.savefig(plt_file, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    all_args = parse_args()
    conf = load_config(all_args, run_eval=True)

    if conf["training"]["run_tag"] != "resid-post":
        conf["plotting"]["mask"] = True

    if conf["plotting"]["type"] == "orthogonality":
        plot_orthogonality(conf)
    else:
        x, mean, confidence, yerr, lambdas = pre_processing_plotting(conf)
        if conf["plotting"]["type"] == "bar":
            bar_plot(conf, x, mean, yerr, lambdas)
        elif conf["plotting"]["type"] == "line":
            line_plot(conf, x, mean, confidence, lambdas)
        else: raise NotImplementedError