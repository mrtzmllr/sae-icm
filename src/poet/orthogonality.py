from src.poet.directories import write_output_dir
from src.poet.config import load_config
from src.poet.argparse import parse_args
from src.poet.sae import TopKSAE
from src.poet.math_utils import _binomial
from src.poet.prompt_template import retrieve_feature_dict

from tqdm import tqdm
from dotenv import load_dotenv
import os
import csv
import torch
import random
from math import factorial


def _normalize_orthogonal_features(D_norm):
    D_sub = D_norm / torch.norm(D_norm, dim = 0, keepdim=True)
    return D_sub



def _retrieve_decoder_weight(conf, sae):
    if conf["sae"]["any_trainable"]: D_full = sae.W_dec.weight
    else: D_full = sae.W_dec.T
    return D_full


def _sampled_orthogonal_features(conf, D_full):
    active = conf["sae"]["finetuning"]["active_features_only"]
    d_sub_decoder = conf["sae"]["finetuning"]["d_sub_decoder"]
    d_model = conf["model"]["d_model"]
    d_sae = conf["sae"]["d_sae"]
    if active:
        D_sub = _active_features_decoder(conf, D_full)
        assert D_sub.shape[0] == d_model
    else:
        indices = torch.randperm(d_sae, device=D_full.device)[:d_sub_decoder]
        D_sub = D_full[:,indices]
    assert D_sub.shape == (d_model, d_sub_decoder)
    return D_sub


def _active_features_decoder(conf, D_full):
    d_sub_decoder = conf["sae"]["finetuning"]["d_sub_decoder"]
    feature_dict = retrieve_feature_dict(conf)
    feature_indices = list(feature_dict.keys())
    active_sub = random.sample(feature_indices, d_sub_decoder)
    D_active = D_full[:, active_sub]
    return D_active


def _return_tril(D_sub):
    tp = D_sub.T @ D_sub
    return torch.tril(tp, diagonal=-1)


def _orthogonality_loss(conf, sae):
    D_full = _retrieve_decoder_weight(conf, sae)
    D_sub = _sampled_orthogonal_features(conf, D_full)
    tril = _return_tril(D_sub)
    orthogonality = tril.square().sum()
    return orthogonality


def _orthogonality_metrics(conf, sae):
    D_full = _retrieve_decoder_weight(conf, sae)
    D_norm = _sampled_orthogonal_features(conf, D_full)
    D_sub = _normalize_orthogonal_features(D_norm)
    tril = _return_tril(D_sub)
    off_diagonal = tril.abs()
    sum_cos = off_diagonal.sum()
    mean_cos = sum_cos / _binomial(tril.shape[0], 2)
    
    return {
        "max_cos": off_diagonal.max().item(),
        "sum_cos": sum_cos.item(),
        "mean_cos": mean_cos.item(),
    }


def _welch(conf, sae):
    welch_k = conf["sae"]["finetuning"]["welch_k"]
    d_sub_decoder = conf["sae"]["finetuning"]["d_sub_decoder"]
    d_model = conf["model"]["d_model"]
    D_full = _retrieve_decoder_weight(conf, sae)
    D_samp = _sampled_orthogonal_features(conf, D_full)
    exponent = 2 * welch_k

    D_sub = _normalize_orthogonal_features(D_samp)
    D_norm = D_sub.T @ D_sub
    sum_norm = (D_norm.abs() ** exponent).sum()
    mean_norm = sum_norm / (d_sub_decoder ** 2)

    D_raw = D_samp.T @ D_samp
    sum_raw = (D_raw.abs() ** exponent).sum()
    
    normalizing_raw = torch.norm(D_samp, dim = 0) ** exponent
    assert normalizing_raw.shape == torch.Size([d_sub_decoder])
    normalizing_raw = normalizing_raw.sum().square()
    mean_raw = sum_raw / normalizing_raw

    binomial = _binomial(d_model + welch_k - 1, welch_k)
    if welch_k == 1: assert binomial == d_model
    bound = 1 / binomial

    return {
        "sum_norm": sum_norm.item(),
        "mean_norm": mean_norm.item(),
        "sum_raw": sum_raw.item(),
        "mean_raw": mean_raw.item(),
        "normalizing_raw": normalizing_raw.item(),
        "bound": bound,
    }



def bootstrap_orthogonality(conf):
    sae = TopKSAE(conf=conf)
    if conf['sae']['use_orthogonal']: sae.from_orthogonal(conf=conf)
    elif conf["sae"]["use_finetuned"]: sae.from_finetuned_2(conf=conf)
    else: sae.from_huggingface(conf=conf)

    assert conf["sae"]["use_finetuned"] and not conf['sae']['use_orthogonal']
    assert conf["sae"]["finetuning"]["active_features_only"]
    if not conf["bootstrap"]["loss"]: assert conf["sae"]["finetuning"]["d_sub_decoder"] == 4096
    assert conf["sae"]["finetuning"]["welch_k"] == 1

    if conf["bootstrap"]["welch"]:
        sum_norms = []
        mean_norms = []
        sum_raws = []
        mean_raws = []
        normalizing_raws = []

        for _ in tqdm(range(conf["bootstrap"]["iterations"])):
            metrics = _welch(conf, sae)
            sum_norm = metrics["sum_norm"]
            mean_norm = metrics["mean_norm"]
            sum_raw = metrics["sum_raw"]
            mean_raw = metrics["mean_raw"]
            normalizing_raw = metrics["normalizing_raw"]
            sum_norms.append(sum_norm)
            mean_norms.append(mean_norm)
            sum_raws.append(sum_raw)
            mean_raws.append(mean_raw)
            normalizing_raws.append(normalizing_raw)

        bound = metrics["bound"]
        
        return {
            "sum_norm": sum_norms,
            "mean_norm": mean_norms,
            "sum_raw": sum_raws,
            "mean_raw": mean_raws,
            "normalizing_raw": normalizing_raws,
            "bound": bound,
        }
    
    
    elif conf["bootstrap"]["cos"]:
        mean_coss = []
        max_coss = []
        
        for _ in tqdm(range(conf["bootstrap"]["iterations"])):
            metrics = _orthogonality_metrics(conf, sae)
            mean_cos = metrics["mean_cos"]
            max_cos = metrics["max_cos"]
            mean_coss.append(mean_cos)
            max_coss.append(max_cos)
            
        return {
            "mean_cos": mean_coss,
            "max_cos": max_coss,
        }
    

    elif conf["bootstrap"]["loss"]:
        losses = []
        
        for _ in tqdm(range(conf["bootstrap"]["iterations"])):
            loss = _orthogonality_loss(conf, sae)
            losses.append(loss)
            
        return {
            "loss": losses,
        }

    else: raise NotImplementedError



if __name__ == "__main__":
    load_dotenv()
    print("loaded env")

    hf_token = os.getenv("HF_TOKEN")
    cache_dir = os.getenv("CACHE_DIR")
    print("got token")

    all_args = parse_args()

    conf = load_config(all_args)

    any_trainable = conf["sae"]["any_trainable"]

    sae = TopKSAE(conf=conf)
    if conf['sae']['use_orthogonal']: sae.from_orthogonal(conf=conf)
    elif conf["sae"]["use_finetuned"]: sae.from_finetuned_2(conf=conf)
    else: sae.from_huggingface(conf=conf)

    d_sub_decoder = conf["sae"]["finetuning"]["d_sub_decoder"]
    welch_k = conf["sae"]["finetuning"]["welch_k"]

    if any_trainable: print(sae.W_enc.bias)
    else: print(sae.b_enc)


    metrics = _orthogonality_metrics(conf, sae)
    welch = _welch(sae)
    loss = _orthogonality_loss(conf, sae)

    max_cos = metrics["max_cos"]
    mean_cos = metrics["mean_cos"]
    sum_norm = welch["sum_norm"]
    mean_norm = welch["mean_norm"]
    sum_raw = welch["sum_raw"]
    mean_raw = welch["mean_raw"]
    normalizing_raw = welch["normalizing_raw"]
    bound = welch["bound"]
    loss = loss.item()

    print(metrics)
    print(welch)

    if conf["eval"]["write_eval_file"]:
        ortho_dir = write_output_dir(conf, train=False)
        ortho_dir += "/orthogonality/"
        ortho_dir += f"d_sub_dec{d_sub_decoder}/"
        ortho_dir += f"welch_k{welch_k}/"
        if conf["sae"]["finetuning"]["active_features_only"]:
            ortho_dir += f"active/"
        ortho_file = ortho_dir + "metrics.csv"

        print(ortho_dir)
        if not os.path.exists(ortho_file): os.makedirs(ortho_dir, exist_ok=True)

        with open(ortho_file, 'w', newline='') as csvfile:
            f = csv.writer(csvfile)
            f.writerow(["max_cos", "mean_cos", "sum_norm", "mean_norm", "sum_raw", "mean_raw", "normalizing_raw", "bound", "loss"])
            f.writerow([max_cos, mean_cos, sum_norm, mean_norm, sum_raw, mean_raw, normalizing_raw, bound, loss])
        
        print("Orthogonality metrics written successfully!")

