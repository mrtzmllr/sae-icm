from src.poet.prompt_template import spans_for_feature

from src.poet.config import load_config
from src.poet.argparse import parse_args

all_args = parse_args()
conf = load_config(all_args, run_eval=True)

assert conf["interpretability"]["write_gsm8k_heap"], "Pass --interpretability.write_gsm8k_heap true; span contexts are looked up by the heap's gsm8k test-set indices."

ftr_index = conf["interpretability"]["feature_index"]

spans = spans_for_feature(conf, ftr_index)
print(spans)