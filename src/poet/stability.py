import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from rapidfuzz.distance import LCSseq

from src.poet.config import load_config
from src.poet.argparse import parse_args
from src.poet.directories import write_feature_storage_dir, write_rouge_dir, write_interv_tokens_dir, write_interv_features_dir, write_jsds_dir
from src.poet.prompt_template import create_prompt_dataset, name_map, group_dataset_map
from src.poet.bootstrap import bootstrap
from src.poet.dataset import retrieve_dataset


all_args = parse_args()
conf = load_config(all_args, run_eval=True)

interv_group = conf["intervenability"]["group"]
assert interv_group in ["names", "locations", "animals"]

name_dict = name_map(conf)[interv_group]
group_dataset = group_dataset_map(conf)

print(name_dict)


def strip_zeros(seq, pad_id=0):
    i = 0
    while i < len(seq) and seq[i] == pad_id:
        i += 1
    return seq[i:]


def trim_tail(seq, max_period=40, min_repeats=4, keep_cycles=0):
    n = len(seq)
    best_start, best_p = n, None
    for p in range(1, max_period + 1):
        i = n - p - 1
        while i >= 0 and seq[i] == seq[i + p]:
            i -= 1
        start = i + 1
        if n - start >= p * min_repeats and start < best_start:
            best_start, best_p = start, p
    if best_p is None:
        return seq
    return seq[: best_start + keep_cycles * best_p]


def _as_idx_tensor(x):
    if torch.is_tensor(x):
        t = x
    elif len(x) and torch.is_tensor(x[0]):
        t = torch.stack(x)
    else:
        t = torch.as_tensor(x)
    return t.long()


def feature_jsd(idx_base, idx_interv,
                mask_base=None, mask_interv=None, eps=1e-12):

    idx_base = _as_idx_tensor(idx_base)
    idx_interv = _as_idx_tensor(idx_interv)

    B, T, k = idx_base.shape

    assert idx_base.shape == idx_interv.shape

    def counts(idx, mask):
        flat = idx.reshape(B, -1)

        d_sae = conf["sae"]["d_sae"]
        src = (torch.ones_like(flat, dtype=torch.float) if mask is None
               else mask.unsqueeze(-1).expand(B, T, k).reshape(B, -1).float())
        c = torch.zeros(B, d_sae, device=idx.device)
        c.scatter_add_(1, flat, src)
        return c

    p = counts(idx_base, mask_base)
    q = counts(idx_interv, mask_interv)
    p = p / p.sum(1, keepdim=True).clamp_min(eps)
    q = q / q.sum(1, keepdim=True).clamp_min(eps)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.add(eps).log() - b.add(eps).log())).sum(1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


replacement_group = conf["intervenability"]["replacement_group"]
if interv_group != replacement_group: raise NotImplementedError



all_rouges = []
jsds = []
drop_mean_rouges = []
drop_mean_jsds = []

for drop_name in name_dict.keys():
    non_dir = write_interv_tokens_dir(conf, drop_name, include_name=None)
    non_pth = non_dir + "/" + "tokens.pt"
    baseline = torch.load(non_pth)

    non_ftrs_dir = write_interv_features_dir(conf, drop_name, include_name=None)
    non_ftrs_pth = non_ftrs_dir + "/" + "features.pt"
    baseline_features = torch.load(non_ftrs_pth)

    drop_rouges = []
    name_jsds = []
    for include_name in name_dict.keys():
        if include_name == drop_name:
            continue

        tokens_dir = write_interv_tokens_dir(conf, drop_name, include_name)
        tokens_pth = tokens_dir + "/" + "tokens.pt"
        interv = torch.load(tokens_pth)

        rouges = []
        for b, v in zip(baseline, interv):
            b = trim_tail(strip_zeros(b.tolist()))
            v = trim_tail(strip_zeros(v.tolist()))
            if len(b) == 0:
                continue
            rouges.append(LCSseq.similarity(b, v) / len(b))
        if not rouges:
            continue
        pair_rouge = sum(rouges) / len(rouges)

        features_dir = write_interv_features_dir(conf, drop_name, include_name)
        features_pth = features_dir + "/" + "features.pt"
        interv_features = torch.load(features_pth)
        pair_jsd = feature_jsd(baseline_features, interv_features).mean().item()

        drop_rouges.append(pair_rouge); all_rouges.append(pair_rouge)
        name_jsds.append(pair_jsd);     jsds.append(pair_jsd)

    if drop_rouges:
        mean_drop = sum(drop_rouges) / len(drop_rouges)
        name_jsd  = sum(name_jsds) / len(name_jsds)
        drop_mean_rouges.append(mean_drop)
        drop_mean_jsds.append(name_jsd)
        print(drop_name, mean_drop, name_jsd)


def compute_descriptive(column: list):
    return bootstrap(conf, column)

lower_rouge, mn_rouge, upper_rouge = compute_descriptive(all_rouges)
lower_jsd, mn_jsd, upper_jsd = compute_descriptive(jsds)

orthogonality_lambda = conf["sae"]["finetuning"]["orthogonality_lambda"]
print(f"results for the orthogonality lambda {orthogonality_lambda}")
print(lower_rouge, mn_rouge, upper_rouge)
print(lower_jsd, mn_jsd, upper_jsd)

if conf["eval"]["write_eval_file"]:
    rouge_dir = write_rouge_dir(conf)
    rouge_path = rouge_dir + f"/rouge_{interv_group}.csv"
    orthogonality_lambda = conf["sae"]["finetuning"]["orthogonality_lambda"]
    new_row = {"orthogonality_lambda": orthogonality_lambda, 
               "mean_rouge": mn_rouge, 
               "lower_rouge": lower_rouge, 
               "upper_rouge": upper_rouge}
    df_new = pd.DataFrame([new_row])

    if os.path.exists(rouge_path):
        df_new.to_csv(rouge_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(rouge_path, mode='w', header=True, index=False)

    jsds_dir = write_jsds_dir(conf)
    jsds_path = jsds_dir + f"/jsds_{interv_group}.csv"
    orthogonality_lambda = conf["sae"]["finetuning"]["orthogonality_lambda"]
    new_row = {"orthogonality_lambda": orthogonality_lambda, 
               "mean_jsd": mn_jsd, 
               "lower_jsd": lower_jsd, 
               "upper_jsd": upper_jsd}
    df_new = pd.DataFrame([new_row])

    if os.path.exists(jsds_path):
        df_new.to_csv(jsds_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(jsds_path, mode='w', header=True, index=False)