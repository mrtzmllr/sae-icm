import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import json
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from src.poet.directories import write_output_dir, finetuned_sae_states_dir, off_the_shelf_sae_states_dir, write_sae_dir


class BaseSAE(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()

        d_model = conf['model']['d_model']
        d_sae = conf['sae']['d_sae']
        disturb = conf['sae']['disturb']
        
        self.train_W_enc = conf['sae']['train_W_enc']
        self.train_b_enc = conf['sae']['train_b_enc']
        self.train_b_dec = conf['sae']['train_b_dec']
        self.any_trainable = conf['sae']['any_trainable']

        self.do_intervene = conf["intervenability"]["do_intervene"]

        assert self.any_trainable == self.train_W_enc or self.train_b_enc or self.train_b_dec

        self.d_model = d_model
        self.d_sae = d_sae
        self.disturb = disturb
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else: device = torch.device("cpu")
        self.device = device

        
        if self.any_trainable:
            self.W_enc = nn.Linear(
                in_features=d_model,
                out_features=d_sae,
                bias=True,
            )
            self.W_dec = nn.Linear(
                in_features=d_sae,
                out_features=d_model,
                bias=True,
            )
        else: 
            self.W_enc = nn.Parameter(torch.empty(d_model, d_sae, device=device), requires_grad=False)
            self.W_dec = nn.Parameter(torch.empty(d_sae, d_model, device=device), requires_grad=False)
            self.b_enc = nn.Parameter(torch.zeros(d_sae, device=device), requires_grad=False)
            self.b_dec = nn.Parameter(torch.zeros(d_model, device=device), requires_grad=False)
        
            print("W_enc", self.W_enc.device)


    def encoder(self, x_flat):
        """
        x_flat:     [batch * sequence_length, d_model]
        z_flat:     [batch * sequence_length, d_sae]
        """
        raise NotImplementedError("Need to implement SAE encoder")
        

    def decoder(self, z_flat):
        """
        z_flat:     [batch * sequence_length, d_sae]
        x_hat_flat: [batch * sequence_length, d_model]
        """
        if self.any_trainable: x_hat_flat = self.W_dec(z_flat)
        else: x_hat_flat = z_flat @ self.W_dec + self.b_dec

        ### sanity
        if self.disturb:
            # disturb = torch.ones(self.b_dec) * 1000
            # x_hat_flat += F.relu(disturb)
            assert not self.any_trainable
            x_hat_flat = z_flat @ self.W_dec + self.b_dec + 10000 * torch.cos(F.relu(self.b_dec))

        return x_hat_flat

    
    def forward(self, x, return_z=False):
        """
        x: [batch, sequence_length, d_model]
        """
        x = x.to(self.device)
        b, s, d = x.shape
        x_flat = x.reshape(b * s, d)

        z_flat = self.encoder(x_flat)

        ### sanity
        if not self.do_intervene:
            bottoms = self.d_sae - self.top_k
            botks = z_flat.topk(k = bottoms, largest = False, dim = -1).indices
            assert torch.gather(z_flat, dim=-1, index=botks).abs().sum() == 0.0
        elif self.draw == 1:
            topks = z_flat.topk(k = self.top_k + 1, dim = -1).indices
            gathered = torch.gather(z_flat, dim=-1, index=topks)
            assert len(gathered.shape) == 2
            elements = gathered.shape[0] * gathered.shape[1]
            if len(torch.nonzero(gathered)) != elements:
                print(gathered)

        x_hat_flat = self.decoder(z_flat)
        x_hat = x_hat_flat.reshape(b, s, d)
        
        if return_z: 
            z = z_flat.reshape(b, s, -1)
            return x_hat, {"sae_latents": z}
        return x_hat



class JumpReLUSAE(BaseSAE):
    def __init__(self, conf):  # 65536 # 16384
        super().__init__(conf = conf)

        self.threshold = nn.Parameter(torch.zeros(self.d_sae, device=self.device), requires_grad=False)

        assert not self.any_trainable, "JumpReLUSAE does not support trainable weights yet"

    
    def encoder(self, x_flat):
        """
        x_flat:     [batch * sequence_length, d_model]
        z_flat:     [batch * sequence_length, d_sae]
        """
        pre = x_flat @ self.W_enc + self.b_enc
        z = (self.threshold < pre) * pre
        return z
    
    
    def from_huggingface(self, conf):
        config_path, _ = retrieve_sae_configs(conf=conf)
        npz = np.load(config_path)

        for key in npz.files:
            arr = npz[key]
            print(f"{key}: shape={arr.shape} dtype={arr.dtype} min={np.nanmin(arr)} max={np.nanmax(arr)} mean={np.nanmean(arr)}")

        with torch.no_grad():
            self.W_enc.copy_(
                torch.from_numpy(npz["W_enc"])
                    .to(self.W_enc.dtype)
                    .to(self.W_enc.device)
            )
            self.b_enc.copy_(
                torch.from_numpy(npz["b_enc"])
                    .to(self.b_enc.dtype)
                    .to(self.b_enc.device)
            )
            self.W_dec.copy_(
                torch.from_numpy(npz["W_dec"])
                    .to(self.W_dec.dtype)
                    .to(self.W_dec.device)
            )
            self.b_dec.copy_(
                torch.from_numpy(npz["b_dec"])
                    .to(self.b_dec.dtype)
                    .to(self.b_dec.device)
            )
            self.threshold.copy_(
                torch.tensor(npz["threshold"])
                    .to(self.threshold.dtype)
                    .to(self.threshold.device)
            )
            print(self.W_enc.device)
            print(self.b_enc.device)
            print(self.W_dec.device)
            print(self.b_dec.device)
            print(self.threshold.device)
            
        return self



class TopKSAE(BaseSAE):
    def __init__(self, conf):  # 65536 # 16384
        super().__init__(conf=conf)

        self.top_k = conf['sae']['top_k']
        self.binary = conf["sae"]["binary"]["use_binary"]
        self.binary_temp = conf["sae"]["binary"]["temperature"]
        self.run_eval = conf["model"]["run_eval"]
        # self.intervention_token = conf["intervenability"]["intervention_token_position"]
        self.insertion_value = conf["intervenability"]["insertion_value"]

        self.stat = 0
        self.indices = []
        self.values = []

        self.step = 0

        self.intervention_indices = {
            "drop": [],
            "include": [],
        }


    def set_top_k(self, updated_top_k):
        self.top_k = updated_top_k


    def intervene(self, post):
        if self.do_intervene:
            drops = self.intervention_indices["drop"]
            includes = self.intervention_indices["include"]
        
        assert len(drops) == 1

        if includes != []:
            indices_to_include = post.indices.clone()
            values_to_include = post.values.clone()
            insert_value = self.insertion_value # torch.max(post.values) * 100
            self.draw = 0
            for incl in includes:
                if self.step >= 0:
                    mask = (post.indices == drops[0])
                    rows, cols = mask.nonzero(as_tuple=True)
                    indices_to_include[rows, cols] = incl
                    values_to_include[rows, cols] = insert_value
            return indices_to_include, values_to_include


    def encoder(self, x_flat, return_pre = False):
        # tied weights between encoder and decoder
        if self.any_trainable: 
            pre = F.relu(self.W_enc(x_flat - self.W_dec.bias))
        else: 
            pre = F.relu((x_flat - self.b_dec) @ self.W_enc + self.b_enc)
        
        if self.binary:
            pre = torch.sigmoid(pre / self.binary_temp)

        post = pre.topk(k = self.top_k, dim = -1, sorted = False)

        zeros = torch.zeros_like(pre)
        
        if self.do_intervene:
            drops = self.intervention_indices["drop"]
            includes = self.intervention_indices["include"]

            new_indices, new_values = self.intervene(post)
            assert new_indices.shape == post.indices.shape
            assert new_values.shape == post.values.shape
            
            z = zeros.scatter_(
                dim=-1, index=new_indices, src=new_values
            )
            Dec = self.W_dec.weight[:, new_indices[0]]

        else:
            z = zeros.scatter_(
                dim=-1, index=post.indices, src=post.values
            )

        ##################
        ### MODIFY
            Dec = self.W_dec.weight[:, post.indices[0]]
        
        if self.run_eval:
            self.indices.append(post.indices.tolist())
            self.values.append(post.values.tolist())
        ##################

        if self.run_eval and self.binary:
            z[z > 0] = 1
            z[z <= 0] = 0

        self.step += 1

        assert z.shape[0] == x_flat.shape[0]
        assert z.shape[1] == self.d_sae
        assert len(z.shape) == 2
        
        if return_pre: 
            # this returns two-dimensional pre of dimension [batch * sequence_length, d_sae]
            return z, pre
        return z
    

    def from_huggingface(self, conf):
        config_path, params_path = retrieve_sae_configs(conf=conf)
        pt_params = torch.load(params_path) #, map_location=torch.device("cpu"))
        
        with open(config_path) as f:
            sae_conf = json.load(f)

        assert sae_conf['trainer']['layer'] == conf['sae']['sae_layer']
        assert sae_conf['trainer']['k'] == conf['sae']['top_k']
        
        key_mapping = {
            "encoder.weight": "W_enc",
            "decoder.weight": "W_dec",
            "encoder.bias": "b_enc",
            "bias": "b_dec",
            "k": "top_k",
        }

        renamed_params = {key_mapping.get(k, k): v for k, v in pt_params.items()}

        with torch.no_grad():
            if self.any_trainable:
                self.W_enc.weight.copy_(renamed_params["W_enc"].to(self.W_enc.weight.dtype).to(self.W_enc.weight.device))
                self.W_enc.bias.copy_(renamed_params["b_enc"].to(self.W_enc.bias.dtype).to(self.W_enc.bias.device))
                self.W_dec.weight.copy_(renamed_params["W_dec"].to(self.W_dec.weight.dtype).to(self.W_dec.weight.device))
                self.W_dec.bias.copy_(renamed_params["b_dec"].to(self.W_dec.bias.dtype).to(self.W_dec.bias.device))
            
            else:
                # due to the way torch uses nn.Linear, we need to transpose the weight matrices
                renamed_params["W_enc"] = renamed_params["W_enc"].T
                renamed_params["W_dec"] = renamed_params["W_dec"].T

                self.W_enc.copy_(renamed_params["W_enc"].to(self.W_enc.dtype).to(self.W_enc.device))
                self.W_dec.copy_(renamed_params["W_dec"].to(self.W_dec.dtype).to(self.W_dec.device))
                self.b_enc.copy_(renamed_params["b_enc"].to(self.b_enc.dtype).to(self.b_enc.device))
                self.b_dec.copy_(renamed_params["b_dec"].to(self.b_dec.dtype).to(self.b_dec.device))

                print(self.W_enc.device)
                print(self.b_enc.device)
                print(self.W_dec.device)
                print(self.b_dec.device)

        return self
    

    def from_orthogonal(self, conf):
        sae_weights_name = conf['sae']['sae_weights']
        weights_dir = off_the_shelf_sae_states_dir(conf)
        weights_path = weights_dir + sae_weights_name + ".pt"
        pt_params = torch.load(weights_path)

        run_eval = conf["model"]["run_eval"]
        checkpoint = conf["eval"]["checkpoint"]
        if run_eval:
            state_dir = finetuned_sae_states_dir(conf)
            state_path = state_dir + f"/sae_state_{checkpoint}.pt"
            ft_params = torch.load(state_path, map_location="cpu")

        with torch.no_grad():
            print("any_trainable", self.any_trainable)
            if self.any_trainable:
                if not run_eval:
                    self.W_enc.weight.copy_(pt_params["W_enc"].T.to(self.W_enc.weight.dtype).to(self.W_enc.weight.device))
                    self.W_enc.bias.copy_(pt_params["b_enc"].to(self.W_enc.bias.dtype).to(self.W_enc.bias.device))
                    self.W_dec.weight.copy_(pt_params["W_dec"].T.to(self.W_dec.weight.dtype).to(self.W_dec.weight.device))
                    self.W_dec.bias.copy_(pt_params["b_dec"].to(self.W_dec.bias.dtype).to(self.W_dec.bias.device))
                
                else:
                    self.W_enc.weight.copy_(ft_params["W_enc.weight"].to(self.W_enc.weight.dtype).to(self.W_enc.weight.device))
                    self.W_enc.bias.copy_(ft_params["W_enc.bias"].to(self.W_enc.bias.dtype).to(self.W_enc.bias.device))
                    self.W_dec.weight.copy_(ft_params["W_dec.weight"].to(self.W_dec.weight.dtype).to(self.W_dec.weight.device))
                    self.W_dec.bias.copy_(ft_params["W_dec.bias"].to(self.W_dec.bias.dtype).to(self.W_dec.bias.device))                

                    sanity_W_dec = pt_params["W_dec"].T.to(self.W_dec.weight.dtype).to(self.W_dec.weight.device)
                    sanity_W_enc = pt_params["W_enc"].T.to(self.W_enc.weight.dtype).to(self.W_enc.weight.device)
                    sanity_b_dec = pt_params["b_dec"].to(self.W_dec.bias.dtype).to(self.W_dec.bias.device)
                    sanity_b_enc = pt_params["b_enc"].to(self.W_enc.bias.dtype).to(self.W_enc.bias.device)

                    assert torch.allclose(self.W_dec.weight, sanity_W_dec, atol=1e-6), "Finetuned W_dec does not match initial W_dec"
                    if self.train_W_enc: assert not torch.allclose(self.W_enc.weight, sanity_W_enc, atol=1e-6), "Finetuned W_enc matches initial W_enc but should be trainable"
                    else: assert torch.allclose(self.W_enc.weight, sanity_W_enc, atol=1e-6), "Finetuned W_enc does not match initial W_enc"
                    if self.train_b_dec: assert not torch.allclose(self.W_dec.bias, sanity_b_dec, atol=1e-6), "Finetuned b_dec matches initial b_dec but should be trainable"
                    else: assert torch.allclose(self.W_dec.bias, sanity_b_dec, atol=1e-6), "Finetuned b_dec does not match initial b_dec"
                    if self.train_b_enc: assert not torch.allclose(self.W_enc.bias, sanity_b_enc, atol=1e-6), "Finetuned b_enc matches initial b_enc but should be trainable"
                    else: assert torch.allclose(self.W_enc.bias, sanity_b_enc, atol=1e-6), "Finetuned b_enc does not match initial b_enc"

                
            else:
                self.W_enc.copy_(pt_params["W_enc"].to(self.W_enc.dtype).to(self.W_enc.device))
                self.b_enc.copy_(pt_params["b_enc"].to(self.b_enc.dtype).to(self.b_enc.device))
                self.W_dec.copy_(pt_params["W_dec"].to(self.W_dec.dtype).to(self.W_dec.device))
                self.b_dec.copy_(pt_params["b_dec"].to(self.b_dec.dtype).to(self.b_dec.device))

                print(self.W_enc.device)
                print(self.b_enc.device)
                print(self.W_dec.device)
                print(self.b_dec.device)

        return self


    def from_finetuned(self, conf):
        sae_dir = write_sae_dir(conf)
        checkpoint = conf["eval"]["checkpoint"]
        state_path = sae_dir + f"/checkpoint-{checkpoint}/"
        state_file = state_path + "pytorch_model.bin"
        # state_file = state_path + "sae_state.pt"

        ft_params = torch.load(state_file, map_location="cpu")
        with torch.no_grad():
            # self.load_state_dict(ft_params)
            self.W_enc.weight.copy_(ft_params["sae.W_enc.weight"].to(self.W_enc.weight.dtype).to(self.W_enc.weight.device))
            self.W_enc.bias.copy_(ft_params["sae.W_enc.bias"].to(self.W_enc.bias.dtype).to(self.W_enc.bias.device))
            self.W_dec.weight.copy_(ft_params["sae.W_dec.weight"].to(self.W_dec.weight.dtype).to(self.W_dec.weight.device))
            self.W_dec.bias.copy_(ft_params["sae.W_dec.bias"].to(self.W_dec.bias.dtype).to(self.W_dec.bias.device))
            
        return self
    

    def from_finetuned_2(self, conf):
        ### MODIFY
        # For writing the SAE weights during SAE fine-tuning Option 1 is used
        # For writing the SAE weights during LM  fine-tuning Option 2 is used

        # Option 1
        # sae_dir = write_sae_dir(conf)
        # checkpoint = conf["eval"]["checkpoint"]
        # state_path = sae_dir + f"/checkpoint-{checkpoint}/"
        # state_file = state_path + "sae_state.pt"
        # print(sae_dir)
        
        # Option 2
        sae_dir = finetuned_sae_states_dir(conf)
        state_file = sae_dir + "/sae_state_23453.pt"

        # Load paramaeters
        ft_params = torch.load(state_file, map_location="cpu")
        with torch.no_grad():
            self.load_state_dict(ft_params)

        return self

    
def retrieve_sae_configs(conf):
    d_sae = conf["sae"]["d_sae"]
    sae_layer = conf["sae"]["sae_layer"]
    d_sae_in_thsnd = d_sae // 1000

    if conf["sae"]["type"] == "jump-relu":
        assert conf["model"]["name"] == "google/gemma-2-2b"
        repo_id = "google/gemma-scope-2b-pt-res"
        gemma_scope_map = {
            16: {10: "77"},
            65: {10: "66"},
        }
        subfolder = f"layer_{sae_layer}/width_{d_sae_in_thsnd}k/average_l0_{gemma_scope_map[d_sae_in_thsnd][sae_layer]}"
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{subfolder}/params.npz"
        )
        return config_path, None

    elif conf["sae"]["type"] == "top-k":
        assert conf["model"]["name"] == "google/gemma-2-2b"

        log_d_sae = np.log2(conf["sae"]["d_sae"])
        assert log_d_sae.is_integer()
        log_d_sae = int(log_d_sae)
        assert log_d_sae in [14, 16]

        repo_id = f"canrager/saebench_gemma-2-2b_width-2pow{log_d_sae}_date-0107"
        assert sae_layer == 12

        top_k = conf["sae"]["top_k"]
        assert (top_k / 10).is_integer()

        t = np.log2(top_k // 10) - 1
        assert t.is_integer()
        t = int(t)
        assert t in range(6)

        # using no checkpoint for now
        subfolder = f"gemma-2-2b_top_k_width-2pow{log_d_sae}_date-0107/resid_post_layer_{sae_layer}/trainer_{t}"
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{subfolder}/config.json"
        )
        params_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{subfolder}/ae.pt"
        )
        return config_path, params_path
    
    else:
        raise NotImplementedError(f"Unknown SAE type: {conf['sae']['type']}")
    

def save_final_sae_state(model, conf):
    param_dict = {}

    for name, p in model.named_parameters():
        if "sae" in name:
            param_dict[name] = p

    states_dir = finetuned_sae_states_dir(conf)
    states_path = states_dir + "/sae_state.pt"

    torch.save(
        param_dict,
        states_path,
    )
    print(f"Saved final SAE state to {states_path}")
