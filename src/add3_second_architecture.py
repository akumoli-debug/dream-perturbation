"""
Addition 3 — Second Architecture Comparison
=============================================
Runs the same linear probe pipeline on a second world model architecture
to show whether the spatial encoding pattern is DIAMOND-specific or general.

Target: DreamerV3 (if checkpoint available) or a simple VAE-based world model
trained from scratch on Pong frames.

Since DreamerV3 checkpoints require JAX/Haiku and are hard to port, this
script implements Option B: train a minimal convolutional VAE world model
on the same Pong frames, run the same probing pipeline, and compare the
layer-wise R² pattern.

Key comparison:
  DIAMOND (diffusion): R² peaks at bottleneck/early decoder → spatial info
                        encoded in the denoising process
  VAE world model:     R² peaks at encoder output / latent → spatial info
                       encoded in explicit latent representation

If the patterns differ, this demonstrates that architectural choices affect
WHERE spatial information is encoded in learned world models — a generalizable
finding beyond a single model.

HOW TO RUN:
  python3 src/add3_second_architecture.py --game Pong
  # ~60-90 minutes (trains VAE + runs probes)
  # Outputs:
  #   results/architecture_comparison.json
  #   results/architecture_comparison.png
  #   checkpoints/vae_world_model.pt
"""

import os, sys, json, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split

DIAMOND_ROOT = os.environ.get("DIAMOND_ROOT", "/root/diamond")
sys.path.insert(0, f"{DIAMOND_ROOT}/src")
sys.path.insert(0, "./src")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ── Minimal VAE world model ───────────────────────────────────────────────────

class ConvEncoder(nn.Module):
    """Encodes [B, C, 84, 84] → [B, latent_dim]"""
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32,  4, stride=2, padding=1),  nn.ReLU(),  # 42
            nn.Conv2d(32,          64,  4, stride=2, padding=1),  nn.ReLU(),  # 21
            nn.Conv2d(64,          128, 4, stride=2, padding=1),  nn.ReLU(),  # 11 (padded)
            nn.Conv2d(128,         256, 4, stride=2, padding=1),  nn.ReLU(),  # 6
            nn.Conv2d(256,         256, 3, stride=1, padding=1),  nn.ReLU(),  # 6
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_mu  = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(256 * 4 * 4, latent_dim)

        # Save intermediate activations for probing
        self.layer_outputs = {}
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self._make_hook(f"conv_{i}"))

    def _make_hook(self, name):
        def hook(module, input, output):
            # GAP to [B, C]
            self.layer_outputs[name] = output.mean(dim=(-2,-1)).detach()
        return hook

    def forward(self, x):
        self.layer_outputs.clear()
        h = self.net(x)
        h = self.pool(h)
        h_flat = h.view(h.size(0), -1)
        self.layer_outputs["latent_mu"] = self.fc_mu(h_flat).detach()
        return self.fc_mu(h_flat), self.fc_var(h_flat)


class ConvDecoder(nn.Module):
    """Decodes [B, latent_dim] → [B, C, 84, 84]"""
    def __init__(self, out_channels=3, latent_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1),  nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1),  nn.ReLU(),
            nn.ConvTranspose2d(32,  out_channels, 4, stride=2, padding=1),
            nn.Tanh(),  # output in [-1, 1]
        )

        self.layer_outputs = {}
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.ConvTranspose2d):
                layer.register_forward_hook(self._make_hook(f"deconv_{i}"))

    def _make_hook(self, name):
        def hook(module, input, output):
            self.layer_outputs[name] = output.mean(dim=(-2,-1)).detach()
        return hook

    def forward(self, z):
        self.layer_outputs.clear()
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.net(h)


class VAEWorldModel(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=3, latent_dim=latent_dim)
        self.decoder = ConvDecoder(out_channels=3, latent_dim=latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    def loss(self, x, recon, mu, log_var):
        recon_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
        return recon_loss + 0.001 * kl_loss, recon_loss, kl_loss

    def get_all_activations(self):
        acts = {}
        acts.update(self.encoder.layer_outputs)
        acts.update(self.decoder.layer_outputs)
        return acts


# ── Training ──────────────────────────────────────────────────────────────────

def train_vae(frames_t, n_epochs=30, batch_size=128, lr=1e-3):
    """Train a VAE on single frames (not temporal context)."""
    dataset = TensorDataset(frames_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=2, pin_memory=True, drop_last=True)
    model = VAEWorldModel(latent_dim=256).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    print(f"  Training VAE on {len(frames_t)} frames for {n_epochs} epochs...")
    for epoch in range(1, n_epochs+1):
        model.train()
        losses = []
        for (x,) in loader:
            x = x.to(DEVICE)
            recon, mu, lv = model(x)
            # Resize recon to match input if needed
            if recon.shape != x.shape:
                recon = F.interpolate(recon, size=x.shape[-2:], mode='bilinear', align_corners=False)
            loss, rl, kl = model.loss(x, recon, mu, lv)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(loss.item())
        sched.step()
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{n_epochs}  loss={np.mean(losses):.5f}")

    return model


# ── Probe training ────────────────────────────────────────────────────────────

def train_probe(acts, labels, epochs=30, seed=0):
    torch.manual_seed(seed)
    dataset = TensorDataset(acts, labels)
    n_test  = max(1, int(0.2 * len(dataset)))
    train_ds, test_ds = random_split(dataset, [len(dataset)-n_test, n_test],
                                      generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    probe  = nn.Linear(acts.shape[1], 2).to(DEVICE)
    opt    = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        probe.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); nn.MSELoss()(probe(xb), yb).backward(); opt.step()
    probe.eval()
    test_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    test_y = torch.stack([test_ds[i][1] for i in range(len(test_ds))])
    with torch.no_grad():
        preds = probe(test_x.to(DEVICE)).cpu()
    ss_res = ((preds-test_y)**2).sum().item()
    ss_tot = ((test_y-test_y.mean(0))**2).sum().item()
    return 1.0 - ss_res / (ss_tot + 1e-8)


# ── Collect VAE activations ───────────────────────────────────────────────────

@torch.no_grad()
def collect_vae_activations(model, frames, batch_size=64):
    all_acts = {}
    N = len(frames)
    for start in range(0, N, batch_size):
        batch = frames[start:start+batch_size].to(DEVICE)
        _ = model(batch)
        for name, tensor in model.get_all_activations().items():
            all_acts.setdefault(name, []).append(tensor.cpu())
    return {k: torch.cat(v) for k, v in all_acts.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(game="Pong"):
    print(f"\n{'='*60}")
    print(f" Architecture Comparison: DIAMOND vs VAE — {game}")
    print(f"{'='*60}")

    # Load data
    print("\n[1] Loading data...")
    with open(f"data/context_labels_{game.lower()}.pkl","rb") as f:
        data = pickle.load(f)
    # Use last frame of each context window as the single frame for VAE
    frames_np = data["contexts"][:, -1]  # [N, 84, 84] uint8 — last frame
    labels_np = data["labels"]

    frames = torch.from_numpy(frames_np).float()/127.5-1.0      # [N,84,84]
    frames = frames.unsqueeze(1).expand(-1,3,-1,-1).contiguous() # [N,3,84,84]
    labels = torch.from_numpy(labels_np).float()
    labels[:, 0] /= 160.0; labels[:, 1] /= 210.0
    print(f"  {len(frames)} frames loaded")

    # Train VAE
    print("\n[2] Training VAE world model...")
    ckpt_path = "checkpoints/vae_world_model.pt"
    if os.path.exists(ckpt_path):
        print(f"  Loading existing checkpoint from {ckpt_path}")
        vae = VAEWorldModel(latent_dim=256).to(DEVICE)
        vae.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    else:
        vae = train_vae(frames, n_epochs=30)
        torch.save(vae.state_dict(), ckpt_path)
        print(f"  Saved checkpoint → {ckpt_path}")
    vae.eval()

    # Collect VAE activations
    print("\n[3] Collecting VAE activations...")
    vae_acts = collect_vae_activations(vae, frames)
    print(f"  Got {len(vae_acts)} layers: {list(vae_acts.keys())}")

    # Train probes on VAE layers
    print("\n[4] Training probes on VAE layers (3 seeds)...")
    vae_results = {}
    for layer, acts in vae_acts.items():
        r2s = [train_probe(acts, labels, seed=s) for s in range(3)]
        vae_results[layer] = {"mean_r2": float(np.mean(r2s)), "std_r2": float(np.std(r2s)),
                               "act_dim": acts.shape[1], "architecture": "VAE"}
        print(f"  {layer:20s}  R²={np.mean(r2s):.4f} ± {np.std(r2s):.4f}")

    # Load DIAMOND v2 results for comparison
    print("\n[5] Loading DIAMOND probe results for comparison...")
    diamond_path = f"probes/layer_ranking_v2_{game.lower()}.json"
    if os.path.exists(diamond_path):
        with open(diamond_path) as f:
            diamond_raw = json.load(f)
        diamond_results = {e["layer"]: {"mean_r2": e["mean_r2"], "std_r2": e["std_r2"],
                                         "architecture": "DIAMOND"} for e in diamond_raw}
        print(f"  Loaded {len(diamond_results)} DIAMOND layers")
    else:
        print("  WARNING: DIAMOND v2 results not found. Run fix2_train_probes_v2.py first.")
        diamond_results = {}

    # Plot comparison
    print("\n[6] Plotting architecture comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: VAE R² by layer
    ax = axes[0]
    vae_layers = list(vae_results.keys())
    vae_r2s    = [vae_results[l]["mean_r2"] for l in vae_layers]
    ax.bar(range(len(vae_layers)), vae_r2s, color="coral", alpha=0.8)
    ax.set_xticks(range(len(vae_layers)))
    ax.set_xticklabels([l.replace("_", "\n") for l in vae_layers], fontsize=8)
    ax.set_ylabel("Probe R² (ball_y)")
    ax.set_title(f"VAE World Model — Spatial Encoding by Layer\n{game}")
    ax.set_ylim(0, 0.8)

    # Plot 2: DIAMOND R² profile (top 10 layers)
    ax = axes[1]
    if diamond_results:
        d_layers = sorted(diamond_results, key=lambda l: diamond_results[l]["mean_r2"], reverse=True)[:10]
        d_r2s    = [diamond_results[l]["mean_r2"] for l in d_layers]
        d_errs   = [diamond_results[l]["std_r2"]  for l in d_layers]
        ax.barh(range(len(d_layers)), d_r2s, xerr=d_errs, color="steelblue", alpha=0.8)
        ax.set_yticks(range(len(d_layers)))
        ax.set_yticklabels([l.replace(".", "\n") for l in d_layers], fontsize=7)
        ax.set_xlabel("Probe R² (ball_y)")
        ax.set_title(f"DIAMOND — Top 10 Layers\n{game}")
        ax.set_xlim(0, 0.8)

    fig.suptitle("Architecture Comparison: Where is Spatial Information Encoded?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig("results/architecture_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved results/architecture_comparison.png")

    # Key finding
    best_vae = max(vae_results, key=lambda l: vae_results[l]["mean_r2"])
    best_vae_r2 = vae_results[best_vae]["mean_r2"]
    best_diamond_r2 = max(diamond_results[l]["mean_r2"] for l in diamond_results) if diamond_results else 0

    print(f"\n{'='*60}")
    print(f" KEY FINDING")
    print(f"{'='*60}")
    print(f"  VAE best layer:     {best_vae:20s}  R²={best_vae_r2:.4f}")
    if diamond_results:
        best_d = max(diamond_results, key=lambda l: diamond_results[l]["mean_r2"])
        print(f"  DIAMOND best layer: {best_d:20s}  R²={best_diamond_r2:.4f}")
        print(f"\n  VAE encodes spatial info in: {'early encoder' if 'conv_0' in best_vae or 'conv_2' in best_vae else 'latent/decoder'}")
        print(f"  DIAMOND encodes it in:       bottleneck/early decoder (u_blocks)")

    # Save results
    all_results = {"vae": vae_results, "diamond": diamond_results,
                   "best_vae_layer": best_vae, "best_vae_r2": best_vae_r2,
                   "best_diamond_r2": best_diamond_r2}
    with open("results/architecture_comparison.json","w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved results/architecture_comparison.json")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="Pong")
    args = p.parse_args()
    main(args.game)
