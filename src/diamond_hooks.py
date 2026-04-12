"""Hook manager for DIAMOND inner_model.

Usage:
    with ActivationCapture(denoiser, SITES) as cap:
        _ = denoiser.denoise(noisy, sigma, obs, act)
        # cap.acts[site_name] -> Tensor (B, C, H, W)
"""
import torch
from contextlib import ContextDecorator

# (site_name, submodule_path)
SITES = [
    ("s0_conv_in",  "conv_in"),
    ("s1_d_stage1", "unet.d_blocks.1"),
    ("s2_d_stage3", "unet.d_blocks.3"),
    ("s3_mid",      "unet.mid_blocks"),
    ("s4_u_stage1", "unet.u_blocks.1"),
    ("s5_u_stage3", "unet.u_blocks.3"),
]


def _get_submodule(root, path):
    m = root
    for part in path.split("."):
        if part.isdigit():
            m = m[int(part)]
        else:
            m = getattr(m, part)
    return m


class ActivationCapture:
    """Capture named submodule outputs on inner_model forward pass.

    ResBlocks returns a tuple; we take the final tensor output (element 0).
    """

    def __init__(self, denoiser, sites=SITES, detach=True, to_cpu=False):
        self.inner = denoiser.inner_model
        self.sites = sites
        self.detach = detach
        self.to_cpu = to_cpu
        self.acts = {}
        self._handles = []

    def _make_hook(self, name):
        def fn(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            if self.detach:
                t = t.detach()
            if self.to_cpu:
                t = t.cpu()
            self.acts[name] = t
        return fn

    def __enter__(self):
        self.acts = {}
        for name, path in self.sites:
            mod = _get_submodule(self.inner, path)
            self._handles.append(mod.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


def probe_one_step(denoiser, noisy, sigma, obs, act):
    """One inner_model forward with capture. Mirrors Denoiser.denoise exactly."""
    cs = denoiser.compute_conditioners(sigma)
    rescaled_obs = obs / denoiser.cfg.sigma_data
    rescaled_noise = noisy * cs.c_in
    with ActivationCapture(denoiser) as cap:
        _ = denoiser.inner_model(rescaled_noise, cs.c_noise.squeeze(), rescaled_obs, act)
    return cap.acts
