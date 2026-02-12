from pyexpat import model
from transformers import Gemma2ForCausalLM, Gemma2Model, Gemma2Config
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2DecoderLayer,
    Gemma2RMSNorm,
    Gemma2RotaryEmbedding,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging
from transformers.cache_utils import Cache, DynamicCache
import torch
import torch.nn as nn
from typing import Optional, Union
from src.poet.sae import JumpReLUSAE, TopKSAE
from src.poet.model_config import Gemma2SAEConfig
import torch.nn.functional as F

logger = logging.get_logger(__name__)

"""
Gemma2 model with SAE included.
Based on the Gemma2 model from Hugging Face Transformers
Using transformers version 4.57.1.
In accordance with the transformers version in poetry.
"""
class Gemma2SAEForCausalLM(Gemma2ForCausalLM):
    ####### Gemma2 config #######
    config_class = Gemma2SAEConfig
    #############################

    def __init__(self, config):
        super().__init__(config)

        ###### SAE integration ######
        self.model = Gemma2SAEModel(config)
        #############################

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()
    

class Gemma2SAEModel(Gemma2Model):
    def __init__(self, config: Gemma2SAEConfig, 
                #  indiv_conf,
                # sae_layer = 12,
                # return_z = False,
                 ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Gemma2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        ###### SAE integration ######
        self.sae_layer = config.sae_layer if config.sae_layer is not None else 12 # indiv_conf['sae']['sae_layer']
        self.return_z = config.return_z
        
        # self.reconstruction = 0.0

        with torch.no_grad():
            self.sae = None
        #############################

        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            ###### SAE integration ######
            if (layer_idx == self.sae_layer) and self.sae is not None:
                if self.return_z: 
                    hidden_states, sae_latents = self.sae(hidden_states, return_z = self.return_z)
                else:
                    hidden_states = self.sae(hidden_states, return_z = self.return_z)
            #############################

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)


        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

        ###### SAE integration ######
        if self.return_z:
            outputs.sae_latents = sae_latents
        #############################

        return outputs
    

class SAEonGemma2(nn.Module):
    def __init__(self, conf):
        super().__init__()
        assert conf["training"]["include_sae"]
        self.sae = None
        self.language_model = None
        self.sae_layer = conf["sae"]["sae_layer"]
        self.orthogonality_lambda = float(conf["sae"]["finetuning"]["orthogonality_lambda"])
        self.d_sae = conf["sae"]["d_sae"]
        self.d_model = conf["model"]["d_model"]
        self.d_sub_decoder = conf["sae"]["finetuning"]["d_sub_decoder"]
        self.return_z = conf["sae"]["finetuning"]["return_z"]
        self.register_buffer("I", torch.eye(self.d_sub_decoder))


    def _normalize_orthogonal_features(self, D_norm):
    #     # D_sub = F.normalize(D_norm, dim = 0)
        D_sub = D_norm / torch.norm(D_norm, dim = 0, keepdim=True)
        return D_sub
    

    def _sampled_orthogonal_features(self):
        D_full = self.sae.W_dec.weight
        indices = torch.randperm(self.d_sae, device=D_full.device)[:self.d_sub_decoder]
        D_sub = D_full[:,indices]
        assert D_sub.shape == (self.d_model, self.d_sub_decoder)
        return D_sub
        # D_norm = self._normalize_orthogonal_features(D_sub) 
        # assert D_norm.shape == (self.d_model, self.d_sub_decoder)
        # return D_norm


    def _return_tril(self, D_sub):
        tp = D_sub.T @ D_sub
        return torch.tril(tp, diagonal=-1)


    def _orthogonality_loss(self):
        D_sub = self._sampled_orthogonal_features()
        # orthogonality = ((D_norm.T @ D_norm - self.I) ** 2).diag().sum()
        tril = self._return_tril(D_sub)
        orthogonality = tril.square().sum()
        return orthogonality
    

    def _orthogonality_metrics(self):
        D_norm = self._sampled_orthogonal_features()
        D_sub = self._normalize_orthogonal_features(D_norm)
        tril = self._return_tril(D_sub)
        off_diagonal = tril.abs()
        
        return {
            "max_cos": off_diagonal.max().item(),
            "mean_cos": off_diagonal.mean().item(),
        }
    

    def _reconstruction_loss(self, x, x_hat, eps = 1e-8):
        # reconstruction = F.mse_loss(x_hat, x)
        # return reconstruction
        reconstruction = torch.norm(x_hat - x, dim = -1, keepdim = True) / (torch.norm(x, dim = -1, keepdim = True) + eps)
        return reconstruction.mean()
    
    
    def forward(self, input_ids, attention_mask = None, labels = None):
        with torch.no_grad():
            outputs = self.language_model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                output_hidden_states = True,
                use_cache = False,
                return_dict = True,
            )

            x = outputs.hidden_states[self.sae_layer]

        # no need to reshape
        assert self.return_z == True
        x_hat, sae_latents = self.sae(x, return_z = self.return_z)
        z = sae_latents["sae_latents"]

        assert z.shape[-1] == self.d_sae
        b, s, _ = z.shape
        z_flat = z.reshape(b * s, self.d_sae)

        l0 = (z_flat != 0).float().sum(dim=-1).mean()
        
        orthogonality = self._orthogonality_loss()
        reconstruction = self._reconstruction_loss(x, x_hat)

        loss = reconstruction + self.orthogonality_lambda * orthogonality

        return {
            "loss": loss,
            "reconstruction": reconstruction.detach(),
            "orthogonality": orthogonality.detach(),
            "l0": l0.detach(),
            "lambda": self.orthogonality_lambda,
            "extra": {"z_flat": z_flat.detach()}
        }          


def insert_sae(model,
               conf,
               ):

    if not conf['training']['include_sae']: 
        return model
    
    if conf['sae']['type'] == "jump-relu": sae = JumpReLUSAE(conf = conf)
    elif conf['sae']['type'] == "top-k": sae = TopKSAE(conf = conf)
    else: raise NotImplementedError
    
    if conf["model"]["finetune_language_model"]:
        train_W_enc = conf['sae']['train_W_enc']
        train_b_enc = conf['sae']['train_b_enc']
        train_b_dec = conf['sae']['train_b_dec']

        sae.to(model.device, model.dtype)
        print("model", model.device)

        if conf['sae']['use_orthogonal']: sae.from_orthogonal(conf=conf)
        elif conf["sae"]["use_finetuned"]: sae.from_finetuned_2(conf=conf)
        else: sae.from_huggingface(conf=conf)

        model.model.sae = sae
        
        for name, p in model.model.sae.named_parameters():
            p.requires_grad = False

            if train_W_enc and 'W_enc.weight' in name: p.requires_grad = True
            if train_b_enc and 'W_enc.bias' in name: p.requires_grad = True
            if train_b_dec and 'W_dec.bias' in name: p.requires_grad = True

            if 'W_dec.bias' in name: print(name, p.requires_grad)
    
    else:
        sae.from_huggingface(conf=conf)
        model.sae = sae

        for name, p in model.sae.named_parameters():
            p.requires_grad = True

    return model