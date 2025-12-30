# drl/common/tensor_tree.py
from __future__ import annotations
from typing import Callable
import torch
from torch import Tensor


# ============================================================
# Core type
# ============================================================
TensorTree = Tensor | tuple["TensorTree", ...]


# ============================================================
# Internal helpers
# ============================================================
def _is_namedtuple(x: object) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_fields")


# ============================================================
# Core traversal primitives
# ============================================================
def apply(fn: Callable[[Tensor], Tensor], tree: TensorTree) -> TensorTree:
    """Functional apply to all tensor leaves."""
    if isinstance(tree, Tensor):
        return fn(tree)

    if isinstance(tree, tuple):
        if _is_namedtuple(tree):
            return type(tree)(*(apply(fn, t) for t in tree))
        return tuple(apply(fn, t) for t in tree)

    raise TypeError(f"Unsupported TensorTree type: {type(tree)}")


def apply_(fn_: Callable[[Tensor], None], tree: TensorTree) -> None:
    """In-place apply to all tensor leaves."""
    if isinstance(tree, Tensor):
        fn_(tree)
        return

    if isinstance(tree, tuple):
        for t in tree:
            apply_(fn_, t)
        return

    raise TypeError(f"Unsupported TensorTree type: {type(tree)}")


# ============================================================
# Structural utilities
# ============================================================
def clone(tree: TensorTree) -> TensorTree:
    return apply(lambda t: t.clone(), tree)


def detach(tree: TensorTree) -> TensorTree:
    return apply(lambda t: t.detach(), tree)


def copy_(dst: TensorTree, src: TensorTree) -> None:
    if isinstance(dst, Tensor):
        dst.copy_(src)
        return

    if isinstance(dst, tuple):
        if type(dst) is not type(src) or len(dst) != len(src):
            raise TypeError("TensorTree structure mismatch in copy_")
        for d, s in zip(dst, src):
            copy_(d, s)
        return

    raise TypeError(f"Unsupported TensorTree type: {type(dst)}")


def to(
    tree: TensorTree,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> TensorTree:
    if device is None and dtype is None:
        return tree

    def move(t: Tensor) -> Tensor:
        return t.to(
            device=device if device is not None else t.device,
            dtype=dtype if dtype is not None else t.dtype,
        )

    return apply(move, tree)


# ============================================================
# Batched allocation
# ============================================================
def batch_zeros(
    template: TensorTree,
    batch_size: tuple[int, ...] = (),
    *,
    device=None,
    dtype=None,
) -> TensorTree:
    def make(t: Tensor) -> Tensor:
        return t.new_zeros(
            size=batch_size + t.shape,
            device=device,
            dtype=dtype if dtype is not None else t.dtype,
        )

    return apply(make, template)


def zeros_like(
    tree: TensorTree,
    *,
    device=None,
    dtype=None,
) -> TensorTree:
    return apply(
        lambda t: t.new_zeros(
            t.shape,
            device=device,
            dtype=dtype if dtype is not None else t.dtype,
        ),
        tree,
    )


# ============================================================
# Explicit indexing (no operator overloading)
# ============================================================
def index(tree: TensorTree, idx) -> TensorTree:
    return apply(lambda t: t[idx], tree)


def assign_index_(dst: TensorTree, idx, src: TensorTree) -> None:
    if isinstance(dst, Tensor):
        dst[idx] = src
        return

    if isinstance(dst, tuple):
        if type(dst) is not type(src) or len(dst) != len(src):
            raise TypeError("TensorTree structure mismatch in assign_index_")
        for d, s in zip(dst, src):
            assign_index_(d, idx, s)
        return

    raise TypeError(f"Unsupported TensorTree type: {type(dst)}")


# ============================================================
# Inspection / debugging
# ============================================================
def summary_tree(
    tree: TensorTree,
    *,
    indent: int = 0,
    name: str | None = None,
) -> None:
    pad = "  " * indent
    prefix = f"{pad}{name}: " if name is not None else pad

    if isinstance(tree, Tensor):
        print(
            f"{prefix}Tensor(shape={tuple(tree.shape)}, "
            f"dtype={tree.dtype}, device={tree.device})"
        )
        return

    if isinstance(tree, tuple):
        fields = getattr(tree, "_fields", None)
        if fields is not None:
            print(f"{prefix}{type(tree).__name__}(")
            for field, value in zip(fields, tree):
                summary_tree(value, indent=indent + 1, name=field)
            print(f"{pad})")
        else:
            print(f"{prefix}tuple(")
            for i, item in enumerate(tree):
                summary_tree(item, indent=indent + 1, name=str(i))
            print(f"{pad})")
        return

    print(f"{prefix}{type(tree).__name__}")


def print_tree(
    tree: TensorTree,
    *,
    indent: int = 0,
    name: str | None = None,
) -> None:
    pad = "  " * indent
    prefix = f"{pad}{name}: " if name is not None else pad

    if isinstance(tree, Tensor):
        tensor_str = repr(tree).replace("\n", "\n" + pad + "  ")
        print(f"{prefix}{tensor_str}")
        return

    if isinstance(tree, tuple):
        fields = getattr(tree, "_fields", None)
        if fields is not None:
            print(f"{prefix}{type(tree).__name__}(")
            for field, value in zip(fields, tree):
                print_tree(value, indent=indent + 1, name=field)
            print(f"{pad})")
        else:
            print(f"{prefix}tuple(")
            for i, item in enumerate(tree):
                print_tree(item, indent=indent + 1, name=str(i))
            print(f"{pad})")
        return

    print(f"{prefix}{type(tree).__name__}")


# ============================================================
# Systematic sanity checks
# ============================================================
if __name__ == "__main__":
    from collections import namedtuple

    torch.manual_seed(0)

    Market = namedtuple("Market", ["bid", "ask"])
    Portfolio = namedtuple("Portfolio", ["shares", "cash"])
    Observation = namedtuple("Observation", ["market", "portfolio"])

    L = 3
    B = 2

    obs_template: TensorTree = Observation(
        market=Market(
            bid=torch.randn(L),
            ask=torch.randn(L),
        ),
        portfolio=Portfolio(
            shares=torch.tensor(3, dtype=torch.int64),
            cash=torch.tensor(1000.0),
        ),
    )

    print("\n=== TEMPLATE SUMMARY ===")
    summary_tree(obs_template)

    print("\n=== TEMPLATE PRINT ===")
    print_tree(obs_template)

    # --------------------------------------------------
    # batch_zeros
    # --------------------------------------------------
    print("\n=== batch_zeros ===")
    obs = batch_zeros(obs_template, batch_size=(B,))
    summary_tree(obs, name="obs")
    print_tree(obs, name="obs")

    # --------------------------------------------------
    # index
    # --------------------------------------------------
    print("\n=== index(obs, 0) ===")
    obs0 = index(obs, 0)
    summary_tree(obs0, name="obs[0]")
    print_tree(obs0, name="obs[0]")

    # --------------------------------------------------
    # assign_index_
    # --------------------------------------------------
    print("\n=== assign_index_ ===")
    assign_index_(obs, 0, obs_template)
    print_tree(obs, name="obs after assignment")

    # --------------------------------------------------
    # clone
    # --------------------------------------------------
    print("\n=== clone ===")
    cloned = clone(obs)
    print_tree(cloned, name="cloned")

    apply_(lambda t: t.add_(1), cloned)
    print("\ncloned after mutation:")
    print_tree(cloned, name="cloned")

    print("\noriginal obs (unchanged):")
    print_tree(obs, name="obs")

    # --------------------------------------------------
    # detach
    # --------------------------------------------------
    print("\n=== detach ===")
    detached = detach(obs)
    print_tree(detached, name="detached")

    # --------------------------------------------------
    # apply
    # --------------------------------------------------
    print("\n=== apply (multiply by 10) ===")
    scaled = apply(lambda t: t * 10, obs)
    print_tree(scaled, name="scaled")

    # --------------------------------------------------
    # apply_
    # --------------------------------------------------
    print("\n=== apply_ (add 5) ===")
    apply_(lambda t: t.add_(5), obs)
    print_tree(obs, name="obs after apply_")

    # --------------------------------------------------
    # copy_
    # --------------------------------------------------
    print("\n=== copy_ ===")
    target = batch_zeros(obs_template, batch_size=(B,))
    copy_(target, obs)
    print_tree(target, name="target after copy_")

    # --------------------------------------------------
    # to(dtype)
    # --------------------------------------------------
    print("\n=== to(dtype=torch.long) ===")
    moved = to(obs, dtype=torch.long)
    summary_tree(moved, name="moved")
    print_tree(moved, name="moved")

    print("\n=== ALL SANITY CHECKS COMPLETED ===")
