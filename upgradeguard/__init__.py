"""UpgradeGuard experiment pipeline."""

from __future__ import annotations

import torch


def _install_torch_compat_shims() -> None:
    if hasattr(torch.nn.Module, "set_submodule"):
        return

    def set_submodule(self, target: str, module: torch.nn.Module) -> None:
        if not target:
            raise ValueError("target must be a non-empty module path")
        atoms = target.split(".")
        parent = self if len(atoms) == 1 else self.get_submodule(".".join(atoms[:-1]))
        child_name = atoms[-1]
        if not isinstance(parent, torch.nn.Module):
            raise AttributeError(f"Parent path for '{target}' is not a torch.nn.Module")
        parent._modules[child_name] = module

    torch.nn.Module.set_submodule = set_submodule


_install_torch_compat_shims()
