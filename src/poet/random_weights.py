from src.poet.sae import retrieve_sae_configs
from src.poet.config import load_config
from src.poet.directories import off_the_shelf_sae_states_dir
import torch
import json
import os 
from dotenv import load_dotenv

load_dotenv()

conf = load_config()

config_path, params_path = retrieve_sae_configs(conf=conf)
pt_params = torch.load(params_path)

with open(config_path) as f:
    sae_conf = json.load(f)

assert sae_conf['trainer']['layer'] == conf['sae']['sae_layer']
assert sae_conf['trainer']['k'] == conf['sae']['top_k']

sae_type = conf['sae']['type']

key_mapping = {
    "encoder.weight": "W_enc",
    "decoder.weight": "W_dec",
    "encoder.bias": "b_enc",
    "bias": "b_dec",
    "k": "top_k",
}

renamed_params = {key_mapping.get(k, k): v for k, v in pt_params.items()}

# due to the way torch uses nn.Linear, we need to transpose the weight matrices
renamed_params["W_enc"] = renamed_params["W_enc"].T
renamed_params["W_dec"] = renamed_params["W_dec"].T

random_W_dec = torch.randn_like(renamed_params["W_dec"])
random_W_dec = random_W_dec / torch.norm(random_W_dec, dim = 0)
random_b_dec = renamed_params["b_dec"] # torch.zeros_like(renamed_params["b_dec"])
corresponding_W_enc = renamed_params["W_enc"] # torch.linalg.pinv(random_W_dec)
# random_W_dec = torch.linalg.pinv(corresponding_W_enc)
corresponding_b_enc = torch.zeros_like(renamed_params["b_enc"]) # torch.zeros_like(renamed_params["b_enc"])

sae_weights_name = "Wencfixd_benczero_Wdecrndm_bdecfixd"

weights_dir = off_the_shelf_sae_states_dir(conf)
weights_path = weights_dir + sae_weights_name + ".pt"


# Wenc  # Wdec  # benc  # bdec  # done
#       #       #       #       #
# pinv  # rndm  # zero  # zero  # 1
# pinv  # rndm  # zero  # rndm  # 1
# pinv  # rndm  # zero  # fixd  #
# pinv  # rndm  # rndm  # zero  #
# pinv  # rndm  # rndm  # rndm  #
# pinv  # rndm  # rndm  # fixd  #
# pinv  # rndm  # fixd  # zero  #
# pinv  # rndm  # fixd  # rndm  #
# pinv  # rndm  # fixd  # fixd  # 1
# fixd  # pinv  # zero  # zero  #
# fixd  # pinv  # zero  # rndm  #
# fixd  # pinv  # zero  # fixd  # 1
# fixd  # pinv  # rndm  # zero  #
# fixd  # pinv  # rndm  # rndm  # 1
# fixd  # pinv  # rndm  # fixd  #
# fixd  # pinv  # fixd  # zero  #
# fixd  # pinv  # fixd  # rndm  #
# fixd  # pinv  # fixd  # fixd  # 1
# fixd  # rndm  # fixd  # fixd  # 1
# fixd  # rndm  # zero  # fixd  # 1

torch.save({
    "W_enc": corresponding_W_enc,
    "W_dec": random_W_dec,
    "b_enc": corresponding_b_enc,
    "b_dec": random_b_dec,
}, weights_path
)

print(sae_weights_name)