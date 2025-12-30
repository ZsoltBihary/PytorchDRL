# # drl/common/tensor_tree.py
# from __future__ import annotations
# from typing import Callable
# import torch
# from torch import Tensor
#
# # Type alias for the underlying data of a TensorTree
# TreeData = Tensor | tuple["TreeData", ...]
#
#
# class TensorTree:
#     """
#     A lightweight wrapper around a tuple-based tree of torch.Tensors.
#
#     - self.data: the underlying TreeData
#     - Functional methods return new TensorTree
#     - In-place methods mutate self.data
#     """
#     data: TreeData
#
#     def __init__(self, data: TreeData):
#         self.data = data
#
#     # ---------------------------
#     # Static helpers
#     # ---------------------------
#     @staticmethod
#     def _is_namedtuple(x: object) -> bool:
#         return isinstance(x, tuple) and hasattr(x, "_fields")
#
#     @staticmethod
#     def _apply(fn: Callable[[Tensor], Tensor], tree_data: TreeData) -> TreeData:
#         """Functional apply to all tensor leaves."""
#         if isinstance(tree_data, Tensor):
#             return fn(tree_data)
#         if isinstance(tree_data, tuple):
#             if TensorTree._is_namedtuple(tree_data):
#                 return type(tree_data)(*(TensorTree._apply(fn, t) for t in tree_data))
#             else:
#                 return tuple(TensorTree._apply(fn, t) for t in tree_data)
#         raise TypeError(f"Unsupported type in TensorTree: {type(tree_data)}")
#
#     @staticmethod
#     def _apply_(fn_: Callable[[Tensor], None], tree_data: TreeData) -> None:
#         """In-place apply to all tensor leaves."""
#         if isinstance(tree_data, Tensor):
#             fn_(tree_data)
#             return
#         if isinstance(tree_data, tuple):
#             for t in tree_data:
#                 TensorTree._apply_(fn_, t)
#             return
#         raise TypeError(f"Unsupported type in TensorTree: {type(tree_data)}")
#
#     # ---------------------------
#     # Public functional API
#     # ---------------------------
#     def apply(self, fn: Callable[[Tensor], Tensor]) -> TensorTree:
#         return TensorTree(self._apply(fn, self.data))
#
#     def apply_(self, fn_: Callable[[Tensor], None]) -> TensorTree:
#         self._apply_(fn_, self.data)
#         return self
#
#     def clone(self) -> TensorTree:
#         return self.apply(lambda t: t.clone())
#
#     def detach(self) -> TensorTree:
#         return self.apply(lambda t: t.detach())
#
#     def copy_(self, other: TensorTree) -> TensorTree:
#         if not isinstance(other, TensorTree):
#             raise TypeError("copy_ expects a TensorTree")
#
#         def assign(dst: TreeData, src: TreeData) -> None:
#             if isinstance(dst, Tensor):
#                 dst.copy_(src)
#                 return
#             if isinstance(dst, tuple):
#                 if type(dst) is not type(src) or len(dst) != len(src):
#                     raise TypeError("TensorTree structure mismatch in copy_")
#                 for d, s in zip(dst, src):
#                     assign(d, s)
#                 return
#             raise TypeError(f"Unsupported type in TensorTree: {type(dst)}")
#
#         assign(self.data, other.data)
#         return self
#
#     def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> TensorTree:
#         if device is None and dtype is None:
#             return self
#
#         def move(t: Tensor) -> Tensor:
#             return t.to(
#                 device=device if device is not None else t.device,
#                 dtype=dtype if dtype is not None else t.dtype
#             )
#         return self.apply(move)
#
#     # ----------------------------------
#     # Indexing (batched semantics)
#     # ----------------------------------
#     def __getitem__(self, idx) -> TensorTree:
#         return TensorTree(self._apply(lambda t: t[idx], self.data))
#
#     def __setitem__(self, idx, value: TensorTree) -> None:
#         if not isinstance(value, TensorTree):
#             raise TypeError("__setitem__ expects a TensorTree")
#
#         def assign(dst: TreeData, src: TreeData) -> None:
#             if isinstance(dst, Tensor):
#                 dst[idx] = src
#                 return
#
#             if isinstance(dst, tuple):
#                 if type(dst) is not type(src) or len(dst) != len(src):
#                     raise TypeError("TensorTree structure mismatch in __setitem__")
#                 for d, s in zip(dst, src):
#                     assign(d, s)
#                 return
#
#             raise TypeError(f"Unsupported TreeData type: {type(dst)}")
#
#         assign(self.data, value.data)
#
#     # ---------------------------
#     # Convenience constructors
#     # ---------------------------
#     @classmethod
#     def batch_zeros(cls, template: TreeData, batch_size: tuple[int, ...] = (), device=None, dtype=None) -> TensorTree:
#         def make(t: Tensor) -> Tensor:
#             return t.new_zeros(size=batch_size + t.size(),
#                                device=device, dtype=dtype if dtype is not None else t.dtype)
#         return cls(cls._apply(make, template))
#
#     @classmethod
#     def zeros_like(cls, other: TensorTree, device=None, dtype=None) -> TensorTree:
#         return cls(cls._apply(lambda t: t.new_zeros(t.shape, device=device, dtype=dtype if dtype else t.dtype),
#                               other.data))
#
#     # ----------------------------------
#     # Summary / inspection
#     # ----------------------------------
#     @staticmethod
#     def summary_tree(tree_data: TreeData, indent: int = 0, name: str | None = None) -> None:
#         pad = "  " * indent
#         prefix = f"{pad}{name}: " if name is not None else pad
#
#         if isinstance(tree_data, Tensor):
#             print(
#                 f"{prefix}Tensor(shape={tuple(tree_data.shape)}, "
#                 f"dtype={tree_data.dtype}, device={tree_data.device})"
#             )
#             return
#
#         if isinstance(tree_data, tuple):
#             fields = getattr(tree_data, "_fields", None)
#             if fields is not None:
#                 print(f"{prefix}{type(tree_data).__name__}(")
#                 for field, value in zip(fields, tree_data):
#                     TensorTree.summary_tree(value, indent + 1, field)
#                 print(f"{pad})")
#             else:
#                 print(f"{prefix}tuple(")
#                 for i, item in enumerate(tree_data):
#                     TensorTree.summary_tree(item, indent + 1, str(i))
#                 print(f"{pad})")
#             return
#
#         print(f"{prefix}{type(tree_data).__name__}")
#
#     def summary(self, name: str | None = None) -> None:
#         TensorTree.summary_tree(self.data, name=name)
#
#     @staticmethod
#     def print_tree(
#         tree_data: TreeData,
#         indent: int = 0,
#         name: str | None = None,
#     ) -> None:
#         pad = "  " * indent
#         prefix = f"{pad}{name}: " if name is not None else pad
#
#         if isinstance(tree_data, Tensor):
#             # repr(tensor) already includes dtype, device, shape, values
#             tensor_str = repr(tree_data)
#             tensor_str = tensor_str.replace("\n", "\n" + pad + "  ")
#             print(f"{prefix}{tensor_str}")
#             return
#
#         if isinstance(tree_data, tuple):
#             fields = getattr(tree_data, "_fields", None)
#             if fields is not None:  # namedtuple
#                 print(f"{prefix}{type(tree_data).__name__}(")
#                 for field, value in zip(fields, tree_data):
#                     TensorTree.print_tree(value, indent + 1, field)
#                 print(f"{pad})")
#             else:  # plain tuple
#                 print(f"{prefix}tuple(")
#                 for i, item in enumerate(tree_data):
#                     TensorTree.print_tree(item, indent + 1, str(i))
#                 print(f"{pad})")
#             return
#
#         print(f"{prefix}{type(tree_data).__name__}")
#
#     def print(self, name: str | None = None) -> None:
#         TensorTree.print_tree(self.data, name=name)
#
#
# # ---------------------------
# # Example usage (financial)
# # Systematic sanity checks
# # ---------------------------
# if __name__ == "__main__":
#     from collections import namedtuple
#
#     torch.manual_seed(0)
#
#     Market = namedtuple("Market", ["bid", "ask"])
#     Portfolio = namedtuple("Portfolio", ["shares", "cash"])
#     TradingObservation = namedtuple("Observation", ["market", "portfolio"])
#
#     L = 3
#     B = 2
#
#     obs_template: TreeData = TradingObservation(
#         market=Market(
#             bid=torch.randn(L),
#             ask=torch.randn(L),
#         ),
#         portfolio=Portfolio(
#             shares=torch.tensor(3, dtype=torch.int64),
#             cash=torch.tensor(1000.0),
#         ),
#     )
#
#     print("\n=== TEMPLATE SUMMARY ===")
#     TensorTree.summary_tree(obs_template)
#
#     print("\n=== TEMPLATE PRINT ===")
#     TensorTree.print_tree(obs_template)
#
#     # --------------------------------------------------
#     # batch_zeros
#     # --------------------------------------------------
#     print("\n=== batch_zeros ===")
#     obs = TensorTree.batch_zeros(obs_template, batch_size=(B,))
#     obs.summary(name="obs")
#     obs.print(name="obs")
#
#     # --------------------------------------------------
#     # __getitem__
#     # --------------------------------------------------
#     print("\n=== __getitem__ (obs[0]) ===")
#     obs0 = obs[0]
#     obs0.summary(name="obs[0]")
#     obs0.print(name="obs[0]")
#
#     # --------------------------------------------------
#     # __setitem__
#     # --------------------------------------------------
#     print("\n=== __setitem__ ===")
#     replacement = TensorTree(obs_template)
#     obs[0] = replacement
#     obs.print(name="obs after obs[0] = replacement")
#
#     # --------------------------------------------------
#     # clone
#     # --------------------------------------------------
#     print("\n=== clone ===")
#     cloned = obs.clone()
#     cloned.print(name="cloned")
#
#     # mutate clone to prove separation
#     cloned.apply_(lambda t: t.add_(1))
#     print("\ncloned after mutation:")
#     cloned.print(name="cloned")
#
#     print("\noriginal obs (should be unchanged):")
#     obs.print(name="obs")
#
#     # --------------------------------------------------
#     # detach
#     # --------------------------------------------------
#     print("\n=== detach ===")
#     detached = obs.detach()
#     detached.print(name="detached")
#
#     # --------------------------------------------------
#     # apply (functional)
#     # --------------------------------------------------
#     print("\n=== apply (multiply by 10) ===")
#     scaled = obs.apply(lambda t: t * 10)
#     scaled.print(name="scaled")
#
#     # --------------------------------------------------
#     # apply_ (in-place)
#     # --------------------------------------------------
#     print("\n=== apply_ (add 5 in-place) ===")
#     obs.apply_(lambda t: t.add_(5))
#     obs.print(name="obs after apply_")
#
#     # --------------------------------------------------
#     # copy_
#     # --------------------------------------------------
#     print("\n=== copy_ ===")
#     target = TensorTree.batch_zeros(obs_template, batch_size=(B,))
#     target.copy_(obs)
#     target.print(name="target after copy_")
#
#     # --------------------------------------------------
#     # to(device, dtype)
#     # --------------------------------------------------
#     print("\n=== to(dtype=torch.long) ===")
#     moved = obs.to(dtype=torch.long)
#     moved.summary(name="moved")
#     moved.print(name="moved")
#
#     print("\n=== ALL SANITY CHECKS COMPLETED ===")
