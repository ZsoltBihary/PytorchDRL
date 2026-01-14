# ============================================================
# torchinfo_tree_summary.py
# ============================================================

from __future__ import annotations

from typing import Tuple, Union, List
from torch import Tensor
import torch.nn as nn
from torchinfo import summary


# ============================================================
# Public type
# ============================================================
# TODO: get these from one true source ...
TensorTree = Union[Tensor, Tuple["TensorTree", ...]]


# ============================================================
# Tree utilities
# ============================================================

def _flatten_tree(tree: TensorTree) -> List[Tensor]:
    """Flatten a TensorTree into a list of tensors (preorder)."""
    if isinstance(tree, Tensor):
        return [tree]
    flat: List[Tensor] = []
    for subtree in tree:
        flat.extend(_flatten_tree(subtree))
    return flat


def _rebuild_tree(template: TensorTree, flat: List[Tensor], idx: int = 0):
    """Rebuild TensorTree from flat tensors using template structure."""
    if isinstance(template, Tensor):
        return flat[idx], idx + 1

    rebuilt = []
    for subtree in template:
        node, idx = _rebuild_tree(subtree, flat, idx)
        rebuilt.append(node)

    # Preserve NamedTuple types automatically
    return type(template)(*rebuilt), idx


# ============================================================
# Wrapper module (torchinfo adapter)
# ============================================================

class _TensorTreeWrapper(nn.Module):
    def __init__(self, model: nn.Module, template: TensorTree):
        super().__init__()
        self.model = model
        self.template = template
        self.num_inputs = len(_flatten_tree(template))

    def forward(self, *flat_inputs: Tensor):
        if len(flat_inputs) != self.num_inputs:
            raise ValueError(
                f"Expected {self.num_inputs} inputs, got {len(flat_inputs)}"
            )

        tree, _ = _rebuild_tree(self.template, list(flat_inputs))
        return self.model(tree)


# ============================================================
# 🔥 Public API (this is what you import)
# ============================================================

def torchinfo_tree_summary(
    model: nn.Module,
    obs: TensorTree,
    **summary_kwargs,
):
    """
    Print torchinfo.summary for models that take a single TensorTree input.

    Parameters
    ----------
    model : nn.Module
        Model whose forward takes a single TensorTree argument.
    obs : TensorTree
        A representative observation instance (used for structure + tensors).
    summary_kwargs :
        Forwarded to torchinfo.summary (e.g. col_names, depth, device).

    Example
    -------
    torchinfo_tree_summary(model, obs, col_names=["input_size", "num_params"])
    """
    flat_inputs = tuple(_flatten_tree(obs))
    wrapper = _TensorTreeWrapper(model, obs)

    return summary(
        wrapper,
        input_data=flat_inputs,
        **summary_kwargs,
    )
