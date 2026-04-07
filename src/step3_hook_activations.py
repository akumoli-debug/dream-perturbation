from __future__ import annotations
from typing import Dict, List
import torch
from torch import Tensor
import torch.nn as nn

class ActivationStore:
    def __init__(self):
        self.activations: Dict[str, Tensor] = {}
        self._hooks = []

    def register_on_unet(self, unet: nn.Module) -> None:
        self._hooks = []
        def _make_hook(name: str):
            def hook(module, input, output):
                self.activations[name] = output.mean(dim=(-2, -1)).detach()
            return hook

        # d_blocks and u_blocks are ModuleList of ResBlocks
        for block_group_name in ("d_blocks", "u_blocks"):
            block_group = getattr(unet, block_group_name)
            for i, res_blocks in enumerate(block_group):
                for j, res_block in enumerate(res_blocks.resblocks):
                    name = f"{block_group_name}.{i}.resblocks.{j}"
                    h = res_block.register_forward_hook(_make_hook(name))
                    self._hooks.append(h)

        # mid_blocks is a single ResBlocks object, not a list
        mid = unet.mid_blocks
        for j, res_block in enumerate(mid.resblocks):
            name = f"mid_blocks.resblocks.{j}"
            h = res_block.register_forward_hook(_make_hook(name))
            self._hooks.append(h)

    def clear(self) -> None:
        self.activations.clear()

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

def build_store_and_register(inner_model: nn.Module) -> ActivationStore:
    store = ActivationStore()
    store.register_on_unet(inner_model.unet)
    return store
