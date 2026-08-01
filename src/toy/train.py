import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

seed = 0
torch.manual_seed(seed)

d_param = 12
m_param = 8
q_param = 128
k_param = 4
sigma = 1.0

n_test =  5000
deltas = [0.25, 0.5, 1.0, 1.5, 2.0]

palette = {"Low": "#0061AC",
           "Med": "#F1B50E",
           "High": "#DF232C"}

markers = {"Low": "o",
           "Med": "s",
           "High": "^"}

project_path = os.getenv('PROJECT_PATH')
# output path depending on run_tag in src/poet/config.yaml
out = os.path.join(project_path, "toy/plots")
os.makedirs(out, exist_ok=True)

########## Sparse latent samples ##########

def sample_sparse_Z(n, d, k, sigma):
    z_param = torch.zeros(n, d)
    for i in range(n):
        idx = torch.randperm(d)[:k]
        z_param[i, idx] = torch.randn(k) * sigma
    return z_param


########## Decoder construction ##########

def welch_bound(m, d):
    return (d - m) / (m * (d - 1))


def make_decoder_low(m, d):
    torch.manual_seed(1)
    Q_mat, _ = torch.linalg.qr(torch.randn(m, m))
    extras = []
    for i in range(d - m):
        angle = torch.tensor((i + 1) * np.pi / (d - m + 1))
        v = Q_mat[:, i % m].clone()
        u = Q_mat[:, (i + 1) % m].clone()
        col = torch.cos(angle) * v + torch.sin(angle) * u
        extras.append(col)
    extra_mat = torch.stack(extras, dim = 1)
    D_mat = torch.cat([Q_mat, extra_mat], dim = 1)
    return F.normalize(D_mat, dim = 0)


def make_decoder_med(m, d):
    torch.manual_seed(2)
    return F.normalize(torch.randn(m, d), dim=0)


def make_decoder_high(m, d, eps = 0.12):
    torch.manual_seed(3)
    v = F.normalize(torch.randn(m), dim=0)
    D_mat = v.unsqueeze(1) + torch.randn(m, d) * eps
    return F.normalize(D_mat, dim=0)


def mean_self_coherence(D_mat):
    d = D_mat.shape[1]
    Gram = D_mat.T @ D_mat
    off = Gram - torch.eye(d)
    return (off ** 2).sum().item() / (d * (d - 1))


########## Readout matrix ##########

def make_A(d, q):
    torch.manual_seed(7)
    return torch.randn(q, d)


########## Intervention mismatch ##########

def intervention_mismatch_all(Z_latent, D_mat, A_mat, delta):
    d = Z_latent.shape[1]
    Gram = D_mat.T @ D_mat
    per_j = torch.zeros(d)

    for j in range(d):
        e_j = torch.zeros(d); e_j[j] = 1.0
        Z_int = Z_latent + delta * e_j

        p_true = F.softmax(Z_int @ A_mat.T, dim=-1)
        p_model = F.softmax(Z_int @ Gram.T @ A_mat.T, dim=-1)

        per_j[j] = ((p_true - p_model) ** 2).sum(dim=-1).mean()

    return per_j.mean().item(), per_j


def theoretical_bound(mu, d, k, sigma, delta, A_mat):
    L_bound = 0.5
    A_op = torch.linalg.norm(A_mat, ord=2).item()
    return 2 * L_bound**2 * A_op**2 * (d - 1) * mu * (k * sigma**2 + delta**2)


########## Main experiment ##########

def run_experiment(A_mat, Z_test):
    decoders = {
        "Low": make_decoder_low(m_param, d_param),
        "Med": make_decoder_med(m_param, d_param),
        "High": make_decoder_high(m_param, d_param),
    }

    mu_values = {}
    empirical = {n: [] for n in decoders}
    theoretical_ = {n: [] for n in decoders}

    wb = welch_bound(m_param, d_param)
    print(f"\nWelch bound on µ(D):  {wb:.4f}  (d={d_param}, m={m_param})\n")

    for name, D_mat in decoders.items():
        mu = mean_self_coherence(D_mat)
        mu_values[name] = mu
        for delta in deltas:
            avg, _ = intervention_mismatch_all(Z_test, D_mat, A_mat, delta)
            theo = theoretical_bound(mu, d_param, k_param, sigma, delta, A_mat)
            empirical[name].append(avg)
            theoretical_[name].append(theo)

    return decoders, mu_values, empirical, theoretical_


########## Figures ##########

def fig_gram_matrices(decoders, mu_values):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f"Gram matrices  G = DᵀD  |  d={d_param}, m={m_param}  "
        f"(Welch floor µ ≥ {welch_bound(m_param, d_param):.3f})",
        fontsize=12, fontweight="bold")

    for ax, (name, D_mat) in zip(axes, decoders.items()):
        G = (D_mat.T @ D_mat).abs().numpy()
        cmap_gram = mcolors.LinearSegmentedColormap.from_list("", ["#FFFFFF", palette[name]])
        im = ax.imshow(G, vmin=0, vmax=1, cmap=cmap_gram, aspect="auto")
        ax.set_title(f"{name}  µ(D)={mu_values[name]:.4f}",
                     fontsize=11, color=palette[name], fontweight="bold")
        ax.set_xlabel("j"); ax.set_ylabel("k")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out + "/coherence_decoders.pdf", bbox_inches="tight")
    print("Saved to coherence_decoders.pdf")
    plt.close(fig)


def fig_mismatch_vs_delta(mu_values, empirical, theoretical_):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(
        f"Intervention mismatch vs δ  (log y-axis)\n"
        f"d={d_param}, m={m_param}, q={q_param}, K={k_param}",
        fontsize=12, fontweight="bold")

    for name in ("Low", "Med", "High"):
        mu = mu_values[name]
        ax.semilogy(deltas, empirical[name],
                    marker=markers[name], color=palette[name], linewidth=2.2,
                    label=f"{name}  µ={mu:.3f}", zorder=3)
        ax.semilogy(deltas, theoretical_[name],
                    linestyle="--", color=palette[name], linewidth=1.4,
                    alpha=0.55, zorder=2)

    ax.plot([], [], "k-",  linewidth=2,   label="Empirical")
    ax.plot([], [], "k--", linewidth=1.4, label="Theorem bound", alpha=0.6)
    ax.set_xlabel("δ", fontsize=12)
    ax.set_ylabel("E[ ‖Y − Ŷ‖² ]  (log scale)", fontsize=11)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out + "/mismatch_vs_delta.pdf", bbox_inches="tight")
    print("Saved to mismatch_vs_delta.pdf")
    plt.close(fig)


########## Entry point ##########

if __name__ == "__main__":
    A_mat = make_A(d_param, q_param)

    Z_test = sample_sparse_Z(n_test, d_param, k_param, sigma)

    decoders, mu_values, empirical, theoretical_ = \
        run_experiment(A_mat, Z_test)

    fig_gram_matrices(decoders, mu_values)
    fig_mismatch_vs_delta(mu_values, empirical, theoretical_)

    print(f"\nAll figures saved to ./{out}/")
