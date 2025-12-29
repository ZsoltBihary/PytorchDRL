from torch import Tensor


class TensorTree:
    """
    A lightweight wrapper around a tuple-based tree of torch.Tensors.

    Conventions:
    - Methods without '_' are functional (return new TensorTree)
    - Methods with '_' mutate in place
    """

    def __init__(self, data):
        self.data = data

    # ------------------------------------------------------------------
    # Core traversal utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _is_namedtuple(x):
        return isinstance(x, tuple) and hasattr(x, "_fields")

    @staticmethod
    def _apply(fn, obj):
        """Functional apply to all tensor leaves."""
        if isinstance(obj, Tensor):
            return fn(obj)

        if isinstance(obj, tuple):
            if TensorTree._is_namedtuple(obj):
                return type(obj)(*(TensorTree._apply(fn, t) for t in obj))
            else:
                return tuple(TensorTree._apply(fn, t) for t in obj)

        raise TypeError(f"Unsupported type in TensorTree: {type(obj)}")

    @staticmethod
    def _apply_(fn_, obj):
        """In-place apply to all tensor leaves."""
        if isinstance(obj, Tensor):
            fn_(obj)
            return

        if isinstance(obj, tuple):
            for t in obj:
                TensorTree._apply_(fn_, t)
            return

        raise TypeError(f"Unsupported type in TensorTree: {type(obj)}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply(self, fn):
        """Return a new TensorTree with fn applied to all tensor leaves."""
        return TensorTree(self._apply(fn, self.data))

    def apply_(self, fn_):
        """Apply fn_ in-place to all tensor leaves."""
        self._apply_(fn_, self.data)
        return self

    # ------------------------------------------------------------------
    # Standard tensor-like operations
    # ------------------------------------------------------------------
    def clone(self):
        """Return a deep clone (new storage)."""
        return self.apply(lambda t: t.clone())

    def detach(self):
        """Return a detached view (shared storage, no grad)."""
        return self.apply(lambda t: t.detach())

    def copy_(self, other: "TensorTree"):
        """In-place copy of values from another TensorTree."""
        if not isinstance(other, TensorTree):
            raise TypeError("copy_ expects a TensorTree")

        def assign(dst, src):
            if isinstance(dst, Tensor):
                dst.copy_(src)
                return

            if isinstance(dst, tuple):
                if type(dst) is not type(src) or len(dst) != len(src):
                    raise TypeError("TensorTree structure mismatch in copy_")
                for d, s in zip(dst, src):
                    assign(d, s)
                return

            raise TypeError(f"Unsupported type in TensorTree: {type(dst)}")

        assign(self.data, other.data)
        return self

    def to(self, device=None, dtype=None):
        """
        Functional device / dtype transfer.

        None means: do not change that attribute.
        """
        if device is None and dtype is None:
            return self  # cheap no-op

        def move(t: Tensor):
            return t.to(
                device=device if device is not None else t.device,
                dtype=dtype if dtype is not None else t.dtype,
            )

        return self.apply(move)

    # ------------------------------------------------------------------
    # Indexing (batched semantics)
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        return TensorTree(self.apply(lambda t: t[idx]).data)

    def __setitem__(self, idx, value: "TensorTree"):
        if not isinstance(value, TensorTree):
            raise TypeError("__setitem__ expects a TensorTree")

        def assign(dst, src):
            if isinstance(dst, Tensor):
                dst[idx] = src
                return

            if isinstance(dst, tuple):
                for d, s in zip(dst, src):
                    assign(d, s)
                return

            raise TypeError(f"Unsupported type in TensorTree: {type(dst)}")

        assign(self.data, value.data)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def zeros_like(cls, template, batch_size=(), device=None, dtype=None):
        def make(t):
            return t.new_zeros(
                batch_size + t.shape,
                device=device,
                dtype=dtype if dtype is not None else t.dtype,
            )

        return cls(cls._apply(make, template))

    # ------------------------------------------------------------------
    # Debugging / inspection
    # ------------------------------------------------------------------
    def summary(self, obj=None, indent=0, name=None):
        """
        Recursively print a summary of the tensor tree.
        Works with torch.Tensor, tuple, and namedtuple leaves.
        """
        if obj is None:
            obj = self.data

        pad = "  " * indent
        prefix = f"{pad}{name}: " if name is not None else pad

        if isinstance(obj, Tensor):
            print(
                f"{prefix}Tensor("
                f"shape={tuple(obj.shape)}, "
                f"dtype={obj.dtype}, "
                f"device={obj.device})"
            )
            return

        if isinstance(obj, tuple):
            fields = getattr(obj, "_fields", None)
            if fields is not None:  # namedtuple
                print(f"{prefix}{type(obj).__name__}(")
                for field, value in zip(fields, obj):
                    self.summary(value, indent + 1, field)
                print(f"{pad})")
            else:  # plain tuple
                print(f"{prefix}tuple(")
                for i, item in enumerate(obj):
                    self.summary(item, indent + 1, str(i))
                print(f"{pad})")
            return

        print(f"{prefix}{type(obj).__name__}")


# ---------------------------
# Example usage (financial)
# ---------------------------
if __name__ == "__main__":
    from collections import namedtuple
    import torch

    Market = namedtuple("Market", ["bid", "ask"])
    Portfolio = namedtuple("Portfolio", ["shares", "cash"])
    ObservationSchema = namedtuple("Observation", ["market", "portfolio"])

    L = 10
    obs_template = ObservationSchema(
        market=Market(
            bid=torch.randn(L),
            ask=torch.randn(L)
        ),
        portfolio=Portfolio(
            shares=torch.tensor(3, dtype=torch.int64),
            cash=torch.tensor(1000.0)
        )
    )

    # wrap in TensorTree
    obs = TensorTree.zeros_like(obs_template, batch_size=(4,), device="cuda")
    obs.summary(name="Observation")
