
import os, sys, json, pickle, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

DIAMOND_ROOT = os.environ.get("DIAMOND_ROOT", "/root/diamond")
sys.path.insert(0, f"{DIAMOND_ROOT}/src")
sys.path.insert(0, "./src")
from models.diffusion.denoiser import Denoiser, DenoiserConfig
from models.diffusion.inner_model import InnerModel, InnerModelConfig
from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)
print(f"Device: {DEVICE}")

class BallYEmbedding(nn.Module):
    def __init__(self, cond_channels, fourier_dim=64):
        super().__init__()
        self.register_buffer("freq", torch.randn(fourier_dim//2)*4.0)
        self.proj = nn.Sequential(
            nn.Linear(fourier_dim, cond_channels), nn.SiLU(),
            nn.Linear(cond_channels, cond_channels))
        nn.init.normal_(self.proj[0].weight, std=0.01); nn.init.zeros_(self.proj[0].bias)
        nn.init.normal_(self.proj[2].weight, std=0.01); nn.init.zeros_(self.proj[2].bias)
    def forward(self, ball_y):
        f = 2*math.pi*ball_y.unsqueeze(1)*self.freq.unsqueeze(0)
        return self.proj(torch.cat([f.cos(), f.sin()], dim=-1))

class ConditionedInnerModel(nn.Module):
    def __init__(self, original, cond_channels):
        super().__init__()
        self.original = original
        self.ball_y_emb = BallYEmbedding(cond_channels)
    def forward(self, noisy_next_obs, c_noise, obs, act, ball_y=None):
        noise_feat = self.original.noise_emb(c_noise)
        act_feat   = self.original.act_emb(act)
        if ball_y is not None:
            cond = self.original.cond_proj(noise_feat + act_feat + self.ball_y_emb(ball_y))
        else:
            cond = self.original.cond_proj(noise_feat + act_feat)
        x = self.original.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
        x, _, _ = self.original.unet(x, cond)
        return self.original.conv_out(F.silu(self.original.norm_out(x)))

class ConditionedDenoiser(nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.cfg = orig.cfg
        self._orig = orig
        self.inner_model = ConditionedInnerModel(orig.inner_model, orig.cfg.inner_model.cond_channels)
    def compute_conditioners(self, sigma): return self._orig.compute_conditioners(sigma)
    def apply_noise(self, x, sigma, son): return self._orig.apply_noise(x, sigma, son)
    def compute_model_output(self, noisy, obs, act, cs, ball_y=None):
        return self.inner_model(noisy*cs.c_in, cs.c_noise, obs/self.cfg.sigma_data, act, ball_y)
    @torch.no_grad()
    def wrap_model_output(self, noisy, mo, cs): return self._orig.wrap_model_output(noisy, mo, cs)
    @torch.no_grad()
    def denoise(self, noisy, sigma, obs, act, ball_y=None):
        cs = self.compute_conditioners(sigma)
        return self.wrap_model_output(noisy, self.compute_model_output(noisy, obs, act, cs, ball_y), cs)
    @property
    def device(self): return next(self.parameters()).device
    def forward(self, frames, obs, act, sigma, ball_y=None):
        cs = self.compute_conditioners(sigma)
        noisy = self.apply_noise(frames, sigma, self.cfg.sigma_offset_noise)
        mo = self.compute_model_output(noisy, obs, act, cs, ball_y)
        target = (frames - cs.c_skip*noisy)/cs.c_out
        return F.mse_loss(mo, target)

class ConditionedSampler(DiffusionSampler):
    def __init__(self, cd, cfg, ball_y_val=None):
        super().__init__(cd._orig, cfg)
        self.cd = cd
        self.ball_y_val = ball_y_val
    @torch.no_grad()
    def sample(self, prev_obs, prev_act):
        device = prev_obs.device
        b, t, c, h, w = prev_obs.size()
        pof = prev_obs.reshape(b, t*c, h, w)
        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn/(len(self.sigmas)-1), 2**0.5-1)
        x = torch.randn(b, c, h, w, device=device)
        ball_y = torch.full((b,), self.ball_y_val, device=device) if self.ball_y_val is not None else None
        for sigma, next_sigma in zip(self.sigmas[:-1], self.sigmas[1:]):
            gamma = gamma_ if self.cfg.s_tmin<=sigma<=self.cfg.s_tmax else 0
            sh = sigma*(gamma+1)
            if gamma>0: x = x+torch.randn_like(x)*self.cfg.s_noise*(sh**2-sigma**2)**0.5
            den = self.cd.denoise(x, sigma, pof, prev_act, ball_y)
            d = (x-den)/sh; dt = next_sigma-sh
            if self.cfg.order==1 or next_sigma==0:
                x = x+d*dt
            else:
                x2=x+d*dt; d2=(x2-self.cd.denoise(x2,next_sigma*s_in,pof,prev_act,ball_y))/next_sigma
                x=x+(d+d2)/2*dt
        return x, []

def to_img(t):
    return ((t.cpu().float().clamp(-1,1)+1)/2*255).byte().numpy().transpose(1,2,0)

def main():
    print("="*50)
    print("Fine-tune DIAMOND with ball_y conditioning")
    print("="*50)

    with open("data/ram_labels_pong.pkl","rb") as f:
        data = pickle.load(f)
    frames_np = data["frames"]
    ball_y_np = data["labels"][:,1]/210.0
    print(f"[1] {len(frames_np)} frames loaded")

    sd = torch.load("/root/diamond/pretrained/atari_100k/models/Pong.pt",
                    map_location="cpu", weights_only=False)
    dsd = {k.replace("denoiser.","",1):v for k,v in sd.items() if k.startswith("denoiser.")}
    na = dsd["inner_model.act_emb.0.weight"].shape[0]
    cfg = DenoiserConfig(sigma_data=0.5, sigma_offset_noise=0.3,
        inner_model=InnerModelConfig(img_channels=3, num_steps_conditioning=4,
            cond_channels=256, depths=[2,2,2,2], channels=[64,64,64,64],
            attn_depths=[False,False,False,False], num_actions=na))
    orig = Denoiser(cfg); orig.load_state_dict(dsd); orig=orig.to(DEVICE)
    print(f"[2] Loaded denoiser, num_actions={na}")

    cd = ConditionedDenoiser(orig).to(DEVICE)

    frames_t = torch.from_numpy(frames_np).float()/127.5-1.0
    frames_t = frames_t.expand(-1,3,-1,-1).contiguous()
    ball_y_t = torch.from_numpy(ball_y_np).float()
    valid = ball_y_t>0.01
    dataset = TensorDataset(frames_t[valid], ball_y_t[valid])
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    print(f"[3] Dataset: {valid.sum()} valid frames")

    # Freeze all, unfreeze only new modules
    for p in cd._orig.parameters(): p.requires_grad_(False)
    trainable = list(cd.inner_model.ball_y_emb.parameters()) +                 list(cd.inner_model.original.cond_proj.parameters())
    for p in trainable: p.requires_grad_(True)
    print(f"[4] Trainable params: {sum(p.numel() for p in trainable):,}")

    opt = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)

    def sample_sigma(B, device):
        return torch.randn(B,device=device).mul(1.2).sub(1.2).exp().clip(2e-3,5.0)

    history = []
    print("[5] Training 40 epochs...")
    for epoch in range(1,41):
        losses = []
        cd.train()
        for frames, by in loader:
            frames=frames.to(DEVICE); by=by.to(DEVICE)
            B=frames.shape[0]
            obs=frames.unsqueeze(1).expand(-1,4,-1,-1,-1).reshape(B,12,84,84)
            act=torch.zeros(B,4,dtype=torch.long,device=DEVICE)
            sigma=sample_sigma(B,DEVICE)
            loss=cd(frames,obs,act,sigma,by)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,1.0)
            opt.step(); losses.append(loss.item())
        sched.step()
        ml=np.mean(losses); history.append(ml)
        print(f"  Epoch {epoch:2d}/40  loss={ml:.6f}")
        if epoch%10==0:
            torch.save({"epoch":epoch,"ball_y_emb":cd.inner_model.ball_y_emb.state_dict(),
                        "cond_proj":cd.inner_model.original.cond_proj.state_dict(),
                        "loss":history},
                       f"checkpoints/conditioned_epoch_{epoch}.pt")
            print(f"  Saved checkpoint epoch {epoch}")

    print("[6] Evaluating...")
    cd.eval()
    ball_y_t2 = torch.from_numpy(ball_y_np)
    mid = ((ball_y_t2>0.45)&(ball_y_t2<0.55)).nonzero(as_tuple=True)[0][:4]
    tf = frames_t[mid].to(DEVICE)
    obs2 = tf.unsqueeze(1).expand(-1,4,-1,-1,-1).contiguous()
    act2 = torch.zeros(4,4,dtype=torch.long,device=DEVICE)
    sampler_cfg = DiffusionSamplerConfig(num_steps_denoising=16)

    ball_y_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    coms = []
    gens = []
    for val in ball_y_vals:
        s = ConditionedSampler(cd, sampler_cfg, val)
        with torch.no_grad(): g,_=s.sample(obs2,act2)
        gens.append(g)
        pl=slice(15,80)
        f2=g[:,:,pl,:].mean(1)
        ys=torch.arange(f2.shape[1],device=DEVICE).float()
        com=((f2*ys.view(1,-1,1)).sum((1,2))/(f2.abs().sum((1,2)).clamp(1))).mean().item()
        coms.append(com); print(f"  ball_y={val:.1f}  CoM_y={com:.2f}px")

    monotonic = all(coms[i+1]>coms[i] for i in range(len(coms)-1))
    print(f"Monotonic CoM: {monotonic}")
    if monotonic: print("SUCCESS: ball_y conditioning is steering ball position")
    else: print("PARTIAL: some steering, not fully monotonic yet")

    for i in range(4):
        panels=[to_img(tf[i])]+[to_img(g[i]) for g in gens]
        row=np.concatenate(panels,axis=1)
        Image.fromarray(row).resize((row.shape[1]*4,row.shape[0]*4),Image.NEAREST).save(
            f"results/conditioned_{i}.png")
    print("Saved results/conditioned_0..3.png")
    print("Panels: [input|y=0.1|y=0.3|y=0.5|y=0.7|y=0.9]")
    with open("results/finetune_results.json","w") as f:
        json.dump({"loss":history,"coms":coms,"monotonic":monotonic},f,indent=2)
    print("Done. Check results/ in the morning.")

if __name__=="__main__":
    main()
