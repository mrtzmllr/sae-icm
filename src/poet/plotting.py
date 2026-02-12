from src.poet.directories import write_output_dir, plotting_dir, write_bootstrap_dir
from src.poet.argparse import parse_args
from src.poet.config import load_config
from src.poet.prompt_template import retrieve_feature_counts

import numpy as np
import pandas as pd
from itertools import cycle
from collections import Counter

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def plot_orthogonality(conf):
    d_sub_decoder = conf["sae"]["finetuning"]["d_sub_decoder"]
    welch_k = conf["sae"]["finetuning"]["welch_k"]

    ortho_lambdas = ["1e-01", "1e-02", "1e-03", "1e-04", "1e-05", "1e-06", "1e-07", "1e-08"]
    means = []

    for o in ortho_lambdas:
        conf["sae"]["finetuning"]["orthogonality_lambda"] = o

        ortho_dir = write_output_dir(conf, train = False)
        ortho_dir += "/orthogonality/"
        ortho_dir += f"d_sub_dec{d_sub_decoder}/"
        ortho_dir += f"welch_k{welch_k}/"
        ortho_file = ortho_dir + "metrics.csv"

        df = pd.read_csv(ortho_file)

        means.append(df["mean_cos"])
    
    plt_dir = plotting_dir()
    plt_file = plt_dir + "/orthogonality.pdf"

    plt.plot(ortho_lambdas, means)
    plt.xlabel("Orthogonality lambda")
    plt.ylabel("Mean")
    plt.savefig(plt_file, bbox_inches="tight")
    plt.close()

    print("Orthogonality plot created!")


def plot_counter_histogram(conf):
    feature_counts = retrieve_feature_counts(conf)
    values = feature_counts.values()
    counts_of_values = Counter(values)
    x = list(counts_of_values.keys())
    y = list(counts_of_values.values())
    
    bauhausblue = (0/255, 97/255, 172/255)
    bauhausyellow = (241/255, 181/255, 14/255)
    bauhausred = (223/255, 35/255, 44/255)

    palette = [bauhausblue, bauhausyellow, bauhausred]

    cmap = LinearSegmentedColormap.from_list(
        "bauhaus_gradient",
        palette
    )

    plt.figure(figsize=(30, 6))
    plt.hist(
        y, bins = 100
    )

    plt.xlabel("Index")
    plt.ylabel("Count")
    plt.title("Histogram of Index Counts")

    sns.despine()
    plt.tight_layout()

    plt_dir = plotting_dir()
    orthogonality_lambda = f"ortho{conf["sae"]["finetuning"]["orthogonality_lambda"]}"
    plt_file = plt_dir + f"/histogram_counter_{orthogonality_lambda}.pdf"
    plt.savefig(plt_file, bbox_inches="tight")
    plt.close()


def colors(num_colors = 3):
    if num_colors > 4: return ["#DF232C"] * num_colors
    if num_colors == 4: return ["#F1B50E", "#92BCEA", "#0061AC", "#DF232C"]
    return ["#F1B50E", "#0061AC", "#DF232C"]


def pre_processing_plotting(conf):
    
    bootstrap_dir = write_bootstrap_dir()
    metric = conf["bootstrap"]["metric"]
    file_path = bootstrap_dir + f"/{metric}.csv"

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

    df = df.iloc[::-1].reset_index(drop=True) # df[list(reversed(df.columns))]

    print(df)

    mean = df["mean"].values
    lower = df["lower"].values
    upper = df["upper"].values

    # if conf["bootstrap"]["metric"] == "orthogonality_loss":
    #     mean = np.log(mean)
    #     lower = np.log(lower)
    #     upper = np.log(upper)
        
    yerr_upper = upper - mean
    yerr_lower = mean - lower

    if conf["plotting"]["mask"]:
        mask_values = conf["plotting"]["mask_values"]
        mask = np.array(mask_values)
        mask = mask.astype(int)
        
        # labels = df["orthogonality_lambda"].astype(str).values
        x = np.arange(len(mask))

        return x, mean[mask], (lower[mask], upper[mask]), (yerr_lower[mask], yerr_upper[mask]), mask
        # return x, mean[mask], (lower[mask], upper[mask]), (yerr_lower[mask], yerr_upper[mask]), labels[mask], mask

    else:
        labels = df["orthogonality_lambda"].astype(str).values
        x = np.arange(len(labels))
        
        return x, mean, (lower, upper), (yerr_lower, yerr_upper), None
        # return x, mean, (lower, upper), (yerr_lower, yerr_upper), labels, None


def mask_to_scientific(conf):
    if "interv" in conf["bootstrap"]["metric"]:
        mask_dict = {
            0: r"$0$",
            1: r"$10^{-9}$",
            2: r"$10^{-8}$",
            3: r"$10^{-6}$",
            4: r"$10^{-5}$",
            5: r"$10^{-4}$",
            6: r"$10^{-3}$",
            7: r"$10^{-2}$",
            8: r"$10^{-1}$",
        }
    else:
        mask_dict = {
            0: r"$0$",
            1: r"$10^{-10}$",
            2: r"$10^{-9}$",
            3: r"$10^{-8}$",
            4: r"$10^{-7}$",
            5: r"$10^{-6}$",
            6: r"$10^{-5}$",
            7: r"$10^{-4}$",
            8: r"$10^{-3}$",
            9: r"$10^{-2}$",
            10: r"$10^{-1}$",
        }
    return mask_dict

def y_label(conf):
    metric = conf["bootstrap"]["metric"]
    # ["dead_features", "math_eval", "interp_score", "interv_eval", "interv_include", "embeddings", "orthogonality_raw", "orthogonality_norm", "orthogonality_mean_cos", "orthogonality_max_cos"]
    label_dict = {
        "dead_features": "Fraction of Dead Features",
        "math_eval": "Accuracy",
        "interp_score": "Interpretability Score",
        "interv_eval": "Accuracy",
        "interv_include": "Correctly Included Indices",
        "embeddings": "Average Cosine Similarity",
        "orthogonality_raw": "Mean Similarity",
        "orthogonality_norm": "Mean Similarity",
        "orthogonality_mean_cos": "Mean Cosine Similarity",
        "orthogonality_max_cos": "Max Cosine Similarity",
        "orthogonality_loss": "Orthogonality Evaluation Loss"# + r"$(\log)$",
    }
    return label_dict[metric]


def bar_plot(conf, x, mean, yerr, mask):

    yerr_lower, yerr_upper = yerr
    metric = conf["bootstrap"]["metric"]
    BAR_WIDTH = 0.7
    COLORS = colors(len(x))

    fig, ax = plt.subplots(figsize=(7, 5))

    # colors = [c for _, c in zip(x, cycle(COLORS))]

    ax.bar(
        x,
        mean,
        width=BAR_WIDTH,
        color=COLORS[:len(x)],
        edgecolor="black",
        linewidth=1.2,
        yerr=[yerr_lower, yerr_upper],
        capsize=6,
        error_kw=dict(lw=2, capthick=2)
    )

    mask_dict = mask_to_scientific(conf)

    ax.set_xticks(x)

    labels = [mask_dict[m] for m in mask]
    ax.set_xticklabels(labels)

    ax.set_xlabel(r"$\mathbf{\lambda}$")
    print(conf["bootstrap"]["metric"])
    
    ax.set_ylabel(y_label(conf))
    ax.tick_params(axis="both", width=2)

    if conf["bootstrap"]["metric"] == "interp_score":
        ax.set_ylim(0, 0.5)
    elif conf["bootstrap"]["metric"] == "embeddings":
        ax.set_ylim(0, 0.7)
    elif "interv" in conf["bootstrap"]["metric"]:
        ax.set_ylim(0.4, 0.8)
    elif "orthogonality" not in conf["bootstrap"]["metric"]:
        ax.set_ylim(0, 0.8)


    sns.despine(ax=ax)
    plt.tight_layout()

    plt_dir = plotting_dir()
    plt_file = plt_dir + f"/{metric}_bar.pdf"
    plt.savefig(plt_file, bbox_inches="tight")
    plt.close()



def line_plot(conf, x, mean, confidence, mask):
    lower, upper = confidence
    metric = conf["bootstrap"]["metric"]
    LINE_WIDTH = 2.5
    COLORS = colors()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        x,
        mean,
        color=COLORS[1],
        linewidth=LINE_WIDTH
    )

    ax.fill_between(
        x,
        lower,
        upper,
        color=COLORS[1],
        alpha=0.25
    )

    mask_dict = mask_to_scientific(conf)

    ax.set_xticks(x)

    if "orthogonality_raw" in conf["bootstrap"]["metric"]:
        ax.set_yticks([0.0012, 0.0014, 0.0016, 0.0018, 0.002, 0.0022])
        ax.set_yticklabels(["1.2", "1.4", "1.6", "1.8", "2.0", "2.2"])
        ax.set_ylabel(y_label(conf) + " " + r"$(10^{-3})$")
    else:
        ax.set_ylabel(y_label(conf))
    

    labels = [mask_dict[m] for m in mask]
    ax.set_xticklabels(labels)
    
    ax.set_xlabel(r"$\mathbf{\lambda}$")
    # ax.set_ylabel(y_label(conf))
    ax.tick_params(axis="both", width=2)

    sns.despine(ax=ax)
    plt.tight_layout()
    plt_dir = plotting_dir()
    plt_file = plt_dir + f"/{metric}_line.pdf"
    plt.savefig(plt_file, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    all_args = parse_args()
    conf = load_config(all_args)

    # plot_counter_histogram(conf=conf)
    # if conf["plotting"]["type"] == "bar":
    #     conf["plotting"]["mask"] = True
    # elif conf["plotting"]["type"] == "line":
    #     conf["plotting"]["mask"] = True #False   
    conf["plotting"]["mask"] = True

    x, mean, confidence, yerr, mask = pre_processing_plotting(conf)
    if conf["plotting"]["type"] == "bar":
        bar_plot(conf, x, mean, yerr, mask)
    elif conf["plotting"]["type"] == "line":
        line_plot(conf, x, mean, confidence, mask)
    else: raise NotImplementedError