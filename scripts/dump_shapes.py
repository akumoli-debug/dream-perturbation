# Why: pick probe sites from ground-truth shapes, not config inference.
import sys, torch
sys.path.insert(0, "/workspace/dream-perturbation")
from src.diamond_loader import load_denoiser

denoiser, cfg = load_denoiser()
denoiser.eval()
device = next(denoiser.parameters()).device

B = 2
n = cfg.denoiser.inner_model.num_steps_conditioning
C = cfg.denoiser.inner_model.img_channels

noisy = torch.randn(B, C, 64, 64, device=device)
c_noise = torch.randn(B, device=device)
obs = torch.randn(B, n * C, 64, 64, device=device)
act = torch.randint(0, 6, (B, n), device=device)

shapes, order = [], []

def hook(name):
    def fn(m, i, o):
        if isinstance(o, torch.Tensor):
            shapes.append((name, type(m).__name__, tuple(o.shape)))
        elif isinstance(o, (tuple, list)):
            for j, oj in enumerate(o):
                if isinstance(oj, torch.Tensor):
                    shapes.append((name + "[" + str(j) + "]", type(m).__name__, tuple(oj.shape)))
                elif isinstance(oj, (tuple, list)):
                    for k, ok in enumerate(oj):
                        if isinstance(ok, torch.Tensor):
                            shapes.append((name + "[" + str(j) + "][" + str(k) + "]", type(m).__name__, tuple(ok.shape)))
        order.append(name)
    return fn

handles = []
for name, mod in denoiser.inner_model.named_modules():
    if name == "":
        continue
    handles.append(mod.register_forward_hook(hook(name)))

with torch.no_grad():
    _ = denoiser.inner_model(noisy, c_noise, obs, act)

for h in handles:
    h.remove()

header = "MODULE".ljust(70) + "  " + "TYPE".ljust(20) + "  SHAPE"
print(header)
print("-" * 110)
for name, tp, shp in shapes:
    print(name.ljust(70) + "  " + tp.ljust(20) + "  " + str(shp))

print()
print("=== " + str(len(order)) + " hook fires ===")
print("first 15 call order:", order[:15])
