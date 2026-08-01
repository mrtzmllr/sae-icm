import json
import torch
from collections import Counter

from src.poet.prompt_template import retrieve_feature_dict
from src.poet.config import load_config
from src.poet.argparse import parse_args
from src.poet.directories import write_feature_storage_dir


all_args = parse_args()
conf = load_config(all_args, run_eval=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features_tensor = torch.full((conf["sae"]["d_sae"], 100), -1).to(device)

feature_dict = retrieve_feature_dict(conf)

features_list = []
for ftr_index in sorted(feature_dict.keys()):
    heap = feature_dict[ftr_index]
    
    test_set_list = []
    test_set_idx_set = set()
    for ftr in heap:
        _, test_set_idx, _ = ftr
        if test_set_idx not in test_set_idx_set:
            test_set_list.append(test_set_idx)
        test_set_idx_set.add(test_set_idx)
        
    assert len(test_set_list) == len(list(test_set_idx_set))

    features_list.extend(test_set_list)

cnt = Counter(features_list)

json_dir = write_feature_storage_dir(conf)
json_path = json_dir + "/feature_counter.jsonl"

counter_dict = {}
for ftr, count in cnt.most_common():
    counter_dict[ftr] = count


if conf["eval"]["write_eval_file"]:
    with open(json_path, "w") as f:
        for ftr, count in counter_dict.items():
            json_line = {
                "feature_idx": ftr,
                "count": count
            }
            f.write(json.dumps(json_line) + "\n")
    print("Feature counter written to .jsonl file!")