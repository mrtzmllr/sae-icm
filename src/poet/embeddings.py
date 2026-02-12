import os
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.poet.config import load_config
from src.poet.argparse import parse_args
from src.poet.math_utils import _binomial
from src.poet.prompt_template import retrieve_feature_explanations, relevant_prefixes, remove_prefix
from src.poet.directories import write_overarching_statistics_dir


def preprocess_embedding(conf):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    explanations = retrieve_feature_explanations(conf)
    feature_indices = list(explanations.keys())
    return model, explanations, feature_indices

# random.seed(21012026)


def write_embeddings(conf, model, explanations, feature_indices):
    sentences = []

    prefixes = relevant_prefixes()

    max_explanations = conf["interpretability"]["num_explanations"]

    for ftr_index in feature_indices:
        explanation = explanations[ftr_index]
        for p in prefixes:
            if p in explanation:
                neat = remove_prefix(explanation)
                sentences.append(neat)
                break
        if len(sentences) == max_explanations:
            break

    assert len(sentences) == max_explanations

    embeddings = model.encode(sentences)
    assert embeddings.shape[0] == max_explanations

    return embeddings

def center_embeddings(conf, embeddings):
    centered_embeddings = conf["interpretability"]["centered_embeddings"]
    if centered_embeddings:
        embeddings = embeddings - np.mean(embeddings, axis = 0, keepdims= True)
        assert np.mean(embeddings, axis = 0).shape[0] == 384
    return embeddings


def similarities(model, embeddings):
    similarities = model.similarity(embeddings, embeddings)
    tril = torch.tril(similarities, diagonal=-1)
    return tril


def tril_list(tril):
    nonzero_tril = tril.reshape(-1)[tril.reshape(-1).nonzero()]
    nonzero_list = nonzero_tril.tolist()
    return nonzero_list


def compute_total(conf, tril):
    max_explanations = conf["interpretability"]["num_explanations"]
    total = torch.sum(tril)
    den = _binomial(max_explanations, 2)

    print((total / den).item())

    return total, den


if __name__ == "__main__":
    all_args = parse_args()
    conf = load_config(all_args, run_eval=True)

    model, explanations, feature_indices = preprocess_embedding(conf)

    random.shuffle(feature_indices)

    embeddings = write_embeddings(model, explanations, feature_indices)
    embeddings = center_embeddings(conf, embeddings)
    tril = similarities(model, embeddings)
    total, den = compute_total(conf, tril)

    max_explanations = conf["interpretability"]["num_explanations"]
    centered_embeddings = conf["interpretability"]["centered_embeddings"]

    if conf["eval"]["write_eval_file"]:
        stats_dir = write_overarching_statistics_dir()
        stats_dir += "/embeddings/"
        stats_dir += f"num_explanations{max_explanations}/"
        if centered_embeddings:
            stats_dir += "centered/"
        stats_path = stats_dir + "metrics.csv"
        realizations_path = stats_dir + "realizations.jsonl"

        print(stats_dir)
        if not os.path.exists(stats_dir): os.makedirs(stats_dir, exist_ok=True)

        orthogonality_lambda = conf["sae"]["finetuning"]["orthogonality_lambda"]
        average = (total / den).item()
        new_row = {"orthogonality_lambda": orthogonality_lambda, "average": average, "total": total, "denominator": den}
        df_new = pd.DataFrame([new_row])

        if os.path.exists(stats_path):
            df_new.to_csv(stats_path, mode='a', header=False, index=False)
        else:
            df_new.to_csv(stats_path, mode='w', header=True, index=False)
        
        print(f"Embedding metrics written successfully at orthogonality {orthogonality_lambda} mean embedding {average}!")
