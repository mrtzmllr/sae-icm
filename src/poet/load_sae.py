from src.poet.directories import write_sae_dir
from src.poet.config import load_config
from src.poet.argparse import parse_args
from src.poet.sae import TopKSAE, retrieve_sae_configs

from dotenv import load_dotenv
import os
import torch

load_dotenv()
print("loaded env")

hf_token = os.getenv("HF_TOKEN")
cache_dir = os.getenv("CACHE_DIR")
print("got token")

all_args = parse_args()

conf = load_config(all_args)

# for k in [20, 40]:
#     for s in [16384, 65536]:
#         conf["sae"]["top_k"] = k
#         conf["sae"]["d_sae"] = s
#         retrieve_sae_configs(conf)



checkpoints = []
checkpoints += [5000, 10000, 15000, 20000, 23453]

# orthos = ["1e-09", "1e-10", "0e-00"]
# orthos += ["1e-03", "3e-03", "5e-03", "8e-03"]
# orthos += ["1e-04", "3e-04", "5e-04", "8e-04"]
# orthos += ["1e-05", "3e-05", "5e-05", "8e-05"]
# orthos += ["1e-06", "3e-06", "5e-06", "8e-06"]
# orthos += ["5e+00", "8e+00"]

print(checkpoints)
# print(orthos)

for checkpoint in checkpoints:
    # for ortho in orthos:
    conf["eval"]["checkpoint"] = checkpoint
    # conf["sae"]["finetuning"]["orthogonality_lambda"] = ortho
    ortho = conf["sae"]["finetuning"]["orthogonality_lambda"]

    sae = TopKSAE(conf=conf)
    sae.from_finetuned(conf=conf)

    sae_dir = write_sae_dir(conf)

    sae_file = sae_dir + f"/checkpoint-{checkpoint}/" + "sae_state.pt"

    torch.save(sae.state_dict(), sae_file)

    sae2 = TopKSAE(conf=conf)
    sae2.from_finetuned_2(conf=conf)

    assert torch.allclose(sae.W_enc.weight, sae2.W_enc.weight)
    assert torch.allclose(sae.W_dec.weight, sae2.W_dec.weight)
    assert torch.allclose(sae.W_enc.bias, sae2.W_enc.bias)
    assert torch.allclose(sae.W_dec.bias, sae2.W_dec.bias)

    print(f"file transcribed for lambda {ortho} at checkpoint {checkpoint}!")