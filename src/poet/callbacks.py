from transformers import TrainerCallback
import torch
import wandb
import numpy as np
import os
from src.poet.directories import finetuned_sae_states_dir


########### FINETUNE LANGUAGE MODEL ###########
class SaveSAECallback(TrainerCallback):
    def __init__(self, conf):
        self.state_dir = finetuned_sae_states_dir(conf)

        self.any_trainable = conf["sae"]["any_trainable"]


    def _save_sae(self, model, step):
        sae = model.model.model.sae

        state = {
            "W_enc.weight": sae.W_enc.weight.detach().cpu(),
            "W_enc.bias": sae.W_enc.bias.detach().cpu(),
            "W_dec.weight": sae.W_dec.weight.detach().cpu(),
            "W_dec.bias": sae.W_dec.bias.detach().cpu(),
        }

        path = os.path.join(self.state_dir, f"sae_state_{step}.pt")
        torch.save(state, path)


    def on_save(self, args, state, control, **kwargs):
        model = kwargs["model"]
        if self.any_trainable: self._save_sae(model, state.global_step)



class SAEMetricsCallback(TrainerCallback):
    def __init__(self, validation_set, conf):
        self.validation_set = validation_set

        self.max_eval_batches = conf["eval"]["max_eval_batches"]
        self.sae_layer = conf["sae"]["sae_layer"]
        self.d_sae = conf["sae"]["d_sae"]
        self.d_model = conf["model"]["d_model"]


    def _gini(self, usage: torch.Tensor):
        assert len(usage.shape) == 1
        if usage.sum() == 0:
            return torch.tensor(0.0, device = usage.device)
        
        usage = torch.sort(usage).values
        n = usage.numel()

        assert n == self.d_sae

        index = torch.arange(1, n + 1, device=usage.device)
        return (2 * (index * usage).sum() / (n * usage.sum())) - (n + 1) / n
    

    @torch.no_grad()
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        assert model is not None
        model.eval()

        sae = model.model.model.sae
        device = next(model.parameters()).device

        recon_losses = []
        l0s = []
        active_counts = None
        usage = torch.zeros(self.d_sae, device=device)
        ranks = []
        top_sing_1 = []
        top_sing_5 = []

        for i, batch in enumerate(self.validation_set):
            if i >= self.max_eval_batches:
                break
            
            batch = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in batch.items()}

            outputs = model(
                **batch,
                output_hidden_states=True,
                return_dict=True,
            )
            
            x = outputs.hidden_states[self.sae_layer]
            
            # all below is two-dimensional: [batch_size * sequence_length, -1]
            x_flat = x.reshape(-1, self.d_model)
            z_flat, pre = sae.encoder(x_flat, return_pre = True)
            x_hat_flat = sae.decoder(z_flat)

            recon_loss = (x_flat - x_hat_flat).norm(dim = -1) / x_flat.norm(dim = -1)
            assert recon_loss.shape == torch.Size([x_flat.shape[0]])
            recon_losses.append(recon_loss.mean().item())
            
            assert z_flat.shape[-1] == self.d_sae
            l0s.append((z_flat != 0).float().sum(dim=-1).mean().item())

            z_flat_mean = z_flat.mean(dim=0)
            assert len(z_flat_mean.shape) == 1
            assert z_flat_mean.shape == torch.Size([self.d_sae])
            usage += z_flat_mean

            assert len(z_flat.shape) == 2
            feature_active = (z_flat.sum(dim=0) > 0)
            active_counts = (
                feature_active if active_counts is None
                else active_counts | feature_active
            )

            rank = torch.linalg.matrix_rank(pre)
            ranks.append(rank)

            _, sigma, _ = torch.linalg.svd(pre, full_matrices = False)

            top_singular = sigma[:5].detach().cpu().numpy()
            top_sing_1.append(top_singular[0])
            top_sing_5.append(top_singular[-1])

        dead_frac = 1.0 - active_counts.float().mean().item()

        gini = self._gini(usage)
        usage_var = torch.var(usage)

        if state.is_world_process_zero:
            wandb.log(
                {
                    "eval/sae/recon_loss": sum(recon_losses) / len(recon_losses),
                    "eval/sae/l0": sum(l0s) / len(l0s),
                    "eval/sae/dead_frac": dead_frac,
                    "eval/sae/gini": gini.item(),
                    "eval/sae/pre-topk-rank": sum(ranks) / len(ranks),
                    "eval/sae/usage-var": usage_var.item(),
                    "eval/sae/top_singular_value": sum(top_sing_1) / len(top_sing_1),
                    "eval/sae/fifth_singular_value": sum(top_sing_5) / len(top_sing_5),
                },
                step=state.global_step,
            )


class TopKScheduleCallback(TrainerCallback):
    def __init__(self, conf):
        self.top_k_start = conf["curriculum"]["top_k"]["start"]
        self.top_k_end = conf["curriculum"]["top_k"]["end"]
        self.top_k_steps = conf["curriculum"]["top_k"]["steps"]
        self.top_k_reduction_factor = conf["curriculum"]["top_k"]["reduction_factor"]

        self.top_k_curriculum = conf["sae"]["top_k_curriculum"]

        assert self.top_k_start == conf["sae"]["top_k"]

        if self.top_k_start > self.top_k_end:
            assert self.top_k_curriculum
            assert np.log2(self.top_k_start).is_integer()
            assert np.log2(self.top_k_end).is_integer()
            assert np.log2(self.top_k_reduction_factor).is_integer()
        else: 
            assert not self.top_k_curriculum
            assert self.top_k_start == self.top_k_end


    def on_step_begin(self, args, state, control, model=None, **kwargs):
        assert model is not None

        if not self.top_k_curriculum:
            return
        
        if state.global_step % self.top_k_steps != 0:
            return
        
        if model.model.model.sae.top_k == self.top_k_end:
            return

        progression_exponent = self.top_k_reduction_factor * (state.global_step // self.top_k_steps)
        # 4096 > 1024 >  256 >  64
        #    400    800   1200  
        assert progression_exponent.is_integer()
        updated_top_k = max(self.top_k_end, self.top_k_start // (2 ** progression_exponent))
        assert updated_top_k.is_integer()

        model.model.model.sae.set_top_k(updated_top_k=updated_top_k)        


########### FINETUNE SAE ###########
class OrthogonalityCallback(SAEMetricsCallback):
    def __init__(self, validation_set, conf):
        super().__init__(validation_set=validation_set, conf=conf)
        

    @torch.no_grad()
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        assert model is not None
        model.eval()

        device = next(model.parameters()).device

        max_cos = []
        mean_cos = []

        reconstructions = []
        l0s = []
        orthogonalities = []
        lambdas = []
        active_counts = None
        usage = torch.zeros(self.d_sae, device=device)

        for i, batch in enumerate(self.validation_set):
            if i >= self.max_eval_batches:
                break

            batch = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in batch.items()}

            outputs = model(
                input_ids = batch["input_ids"],
                attention_mask = batch["attention_mask"],
            )
            
            reconstructions.append(outputs["reconstruction"])
            orthogonalities.append(outputs["orthogonality"])
            l0s.append(outputs["l0"])
            lambdas.append(outputs["lambda"])

            metrics = model._orthogonality_metrics()
            max_cos.append(metrics["max_cos"])
            mean_cos.append(metrics["mean_cos"])

            z_flat = outputs["extra"]["z_flat"]

            z_flat_mean = z_flat.mean(dim=0)
            assert len(z_flat_mean.shape) == 1
            assert z_flat_mean.shape == torch.Size([self.d_sae])
            usage += z_flat_mean

            assert len(z_flat.shape) == 2
            feature_active = (z_flat.sum(dim=0) > 0)
            active_counts = (
                feature_active if active_counts is None
                else active_counts | feature_active
            )

        dead_frac = 1.0 - active_counts.float().mean().item()

        gini = self._gini(usage)
        usage_var = torch.var(usage)

        if state.is_world_process_zero:
            wandb.log(
                {
                    "eval/max_cos": metrics["max_cos"],
                    "eval/mean_cos": metrics["mean_cos"], 
                    "eval/reconstruction": sum(reconstructions) / len(reconstructions),
                    "eval/orthogonality": sum(orthogonalities) / len(orthogonalities),
                    "eval/l0": sum(l0s) / len(l0s),
                    "eval/lambda": sum(lambdas) / len(lambdas),
                    "eval/dead_frac": dead_frac,
                    "eval/gini": gini.item(),
                    "eval/usage-var": usage_var.item(),
                },
                step=state.global_step,
            )