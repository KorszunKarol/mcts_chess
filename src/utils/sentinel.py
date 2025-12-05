from __future__ import annotations

import contextlib
import inspect
import logging
from functools import wraps
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import jax
    import jax.numpy as jnp
    from jax import core as jax_core
    import jax.debug as jax_debug
except Exception:  # pragma: no cover
    jax = None
    jnp = None
    jax_core = None
    jax_debug = None


class Verbosity:
    """Verbosity levels for the Sentinel."""

    OFF = 0
    SHAPES = 1
    VALUES = 2
    FULL_TRACE = 3

    _NAME_TO_LEVEL = {
        "OFF": OFF,
        "SHAPES": SHAPES,
        "VALUES": VALUES,
        "FULL_TRACE": FULL_TRACE,
    }

    @classmethod
    def normalize(cls, value: Union[int, str]) -> int:
        if isinstance(value, str):
            upper = value.strip().upper()
            if upper not in cls._NAME_TO_LEVEL:
                raise ValueError(f"Unknown verbosity '{value}'. Valid: {list(cls._NAME_TO_LEVEL)}")
            return cls._NAME_TO_LEVEL[upper]
        if isinstance(value, int):
            if value not in cls._NAME_TO_LEVEL.values():
                raise ValueError(f"Verbosity int must be in {list(cls._NAME_TO_LEVEL.values())}")
            return value
        raise TypeError("Verbosity must be an int or str")

    @classmethod
    def name(cls, value: int) -> str:
        for name, level in cls._NAME_TO_LEVEL.items():
            if level == value:
                return name
        return "UNKNOWN"


class TorchInspector:
    """Shape/value summaries for torch tensors."""

    @staticmethod
    def describe(tensor: "torch.Tensor", verbosity: int) -> str:
        shape = tuple(tensor.shape)
        dtype = str(tensor.dtype)
        device = str(tensor.device)
        parts = [f"shape={shape}", f"dtype={dtype}", f"device={device}"]

        if tensor.requires_grad:
            parts.append("requires_grad=True")

        # Detect invalid values early
        invalid_flag: Optional[bool] = None
        try:
            invalid_flag = (~torch.isfinite(tensor)).any().item()
        except Exception:
            invalid_flag = None
        if invalid_flag:
            parts.append("contains_nan_or_inf=True")

        # Only compute stats when we are explicitly inspecting values
        if verbosity >= Verbosity.VALUES and tensor.numel() > 0:
            try:
                data = tensor.detach()
                parts.append(f"mean={data.mean().item():.4g}")
                parts.append(f"std={data.std().item():.4g}")
            except Exception:
                parts.append("stats_error=True")

        return ", ".join(parts)


class JaxInspector:
    """Shape/value summaries for JAX arrays and tracers."""

    @staticmethod
    def _is_tracer(x: Any) -> bool:
        if jax_core is None:
            return False
        return isinstance(x, jax_core.Tracer)

    @staticmethod
    def describe(array: Any, verbosity: int, inspect_runtime_values: bool) -> str:
        is_tracer = JaxInspector._is_tracer(array) or hasattr(array, "aval")
        shape: Tuple[int, ...] = ()
        dtype = None

        if is_tracer and hasattr(array, "aval"):
            aval = array.aval
            shape = tuple(getattr(aval, "shape", ()))
            dtype = getattr(aval, "dtype", None)
        elif hasattr(array, "shape"):
            shape = tuple(array.shape)  # type: ignore[arg-type]
            dtype = getattr(array, "dtype", None)

        parts = [f"shape={shape}"]
        if dtype is not None:
            parts.append(f"dtype={dtype}")
        if is_tracer:
            parts.append("tracer=True")

        # Only run expensive checks on real arrays when requested
        if not is_tracer and verbosity >= Verbosity.VALUES and jnp is not None:
            try:
                invalid = (~jnp.isfinite(array)).any()
                if bool(invalid):
                    parts.append("contains_nan_or_inf=True")
            except Exception:
                parts.append("nan_check_error=True")

        # Inject runtime values via JAX-side printing if explicitly enabled
        if (
            inspect_runtime_values
            and jax_debug is not None
            and verbosity >= Verbosity.VALUES
        ):
            try:
                jax_debug.print("[Sentinel] {msg}", msg=str(parts))
            except Exception:
                # Avoid breaking traces for logging failures
                parts.append("debug_print_failed=True")

        return ", ".join(parts)


class Sentinel:
    """
    Centralized, context-aware debugging assistant for hybrid JAX/PyTorch code.
    """

    def __init__(self) -> None:
        self.verbosity_level = Verbosity.OFF
        self.inspect_jax_tracers: bool = True
        self.inspect_runtime_values: bool = False
        self.context_stack: list[str] = []
        self.logger = logging.getLogger("sentinel")

    # --- Configuration ---
    def set_verbosity(self, level: Union[int, str]) -> None:
        self.verbosity_level = Verbosity.normalize(level)
        self.log(f"Sentinel verbosity set to {Verbosity.name(self.verbosity_level)}")

    def configure(
        self,
        *,
        verbosity: Union[int, str, None] = None,
        inspect_jax_tracers: Optional[bool] = None,
        inspect_runtime_values: Optional[bool] = None,
    ) -> None:
        if verbosity is not None:
            self.verbosity_level = Verbosity.normalize(verbosity)
        if inspect_jax_tracers is not None:
            self.inspect_jax_tracers = inspect_jax_tracers
        if inspect_runtime_values is not None:
            self.inspect_runtime_values = inspect_runtime_values
        self.log(
            f"Sentinel configured: verbosity={Verbosity.name(self.verbosity_level)}, "
            f"inspect_jax_tracers={self.inspect_jax_tracers}, "
            f"inspect_runtime_values={self.inspect_runtime_values}"
        )

    @property
    def enabled(self) -> bool:
        return self.verbosity_level > Verbosity.OFF

    # --- Context management ---
    def enter_scope(self, name: str) -> None:
        self.context_stack.append(name)

    def exit_scope(self) -> None:
        if self.context_stack:
            self.context_stack.pop()

    @contextlib.contextmanager
    def scope(self, name: str):
        self.enter_scope(name)
        try:
            yield
        finally:
            self.exit_scope()

    # --- Logging helpers ---
    def log(self, message: str) -> None:
        indent = "  " * len(self.context_stack)
        prefix = ""
        if self.context_stack:
            prefix = f"[{self.context_stack[-1]}] "
        self.logger.info("%s%s%s", indent, prefix, message)

    def log_tensor(self, name: str, tensor: Any) -> None:
        if not self.enabled:
            return
        description = self._describe_value(tensor)
        self.log(f"{name}: {description}")

    def _describe_value(self, value: Any) -> str:
        # Torch tensors
        if torch is not None and isinstance(value, torch.Tensor):
            return TorchInspector.describe(value, self.verbosity_level)

        # JAX arrays / tracers
        if jax is not None and (
            (hasattr(jax, "Array") and isinstance(value, jax.Array))
            or (jnp is not None and isinstance(value, jnp.ndarray))
            or (jax_core is not None and isinstance(value, jax_core.Tracer))
            or hasattr(value, "aval")
        ):
            return JaxInspector.describe(
                value, self.verbosity_level, self.inspect_runtime_values
            )

        # NumPy arrays
        if np is not None and isinstance(value, np.ndarray):
            shape = value.shape
            dtype = value.dtype
            summary = f"shape={shape}, dtype={dtype}"
            if self.verbosity_level >= Verbosity.VALUES and value.size > 0:
                summary += f", mean={value.mean():.4g}, std={value.std():.4g}"
            return summary

        # Fallback primitives
        return f"type={type(value).__name__}, value={value}"

    # --- Decorators ---
    def trace(self, fn: Callable) -> Callable:
        """
        Decorator that logs entry/exit with argument shapes.
        In production (OFF) mode it is a near no-op.
        """

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return fn(*args, **kwargs)

            fn_name = fn.__qualname__
            self.enter_scope(fn_name)
            try:
                if self.verbosity_level >= Verbosity.SHAPES:
                    arg_signature = self._format_args(fn, args, kwargs)
                    self.log(f"Entering {fn_name}{arg_signature}")
                result = fn(*args, **kwargs)
                if self.verbosity_level >= Verbosity.SHAPES:
                    self.log(f"Exiting {fn_name} -> {self._format_result(result)}")
                return result
            finally:
                self.exit_scope()

        return wrapper

    def shape_guard(
        self,
        *,
        inputs: Optional[Dict[str, str]] = None,
        outputs: Optional[Union[str, Dict[str, str]]] = None,
    ) -> Callable:
        """
        Decorator to validate shapes against a simple schema.
        Schema strings are comma-separated dimensions (e.g., "B, C, H, W").
        Repeated dimension names must match (e.g., "B, B" enforces equality).
        """
        inputs = inputs or {}

        def decorator(fn: Callable) -> Callable:
            sig = inspect.signature(fn)

            @wraps(fn)
            def wrapper(*args, **kwargs):
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()

                # Validate input shapes
                for name, spec in inputs.items():
                    if name in bound.arguments:
                        self._check_shape(name, bound.arguments[name], spec, "input")

                result = fn(*args, **kwargs)

                # Validate outputs
                if outputs:
                    self._validate_outputs(result, outputs)

                return result

            return wrapper

        return decorator

    # --- Shape helpers ---
    def _get_shape(self, value: Any) -> Optional[Tuple[int, ...]]:
        if value is None:
            return None
        if torch is not None and isinstance(value, torch.Tensor):
            return tuple(value.shape)
        if jax is not None and (
            (hasattr(jax, "Array") and isinstance(value, jax.Array))
            or (jax_core is not None and isinstance(value, jax_core.Tracer))
            or hasattr(value, "aval")
            or (jnp is not None and isinstance(value, jnp.ndarray))
        ):
            if hasattr(value, "aval") and hasattr(value.aval, "shape"):
                return tuple(value.aval.shape)  # type: ignore[attr-defined]
            if hasattr(value, "shape"):
                return tuple(value.shape)  # type: ignore[arg-type]
        if np is not None and isinstance(value, np.ndarray):
            return tuple(value.shape)
        if hasattr(value, "shape"):
            try:
                return tuple(value.shape)  # type: ignore[arg-type]
            except Exception:
                return None
        return None

    def _check_shape(self, name: str, value: Any, spec: str, label: str) -> None:
        actual_shape = self._get_shape(value)
        if actual_shape is None:
            return

        expected_tokens = [token.strip() for token in spec.split(",") if token.strip()]
        if len(actual_shape) != len(expected_tokens):
            raise ValueError(
                f"Shape mismatch for {label} '{name}': expected {spec} "
                f"({len(expected_tokens)} dims) but got {actual_shape}"
            )

        dim_map: Dict[str, int] = {}
        for idx, (actual, token) in enumerate(zip(actual_shape, expected_tokens)):
            if token in {"*", "...", "_"}:
                continue
            if token.isdigit():
                expected = int(token)
            else:
                expected = dim_map.setdefault(token, actual)
            if actual != expected:
                raise ValueError(
                    f"Dimension {idx} for {label} '{name}' expected '{token}'={expected} "
                    f"but got {actual} (spec: {spec})"
                )

    def _validate_outputs(self, result: Any, outputs: Union[str, Dict[str, str]]) -> None:
        if isinstance(outputs, str):
            self._check_shape("return", result, outputs, "output")
            return

        # Assume mapping of attribute/key -> shape spec
        if isinstance(result, dict):
            for key, spec in outputs.items():
                if key in result:
                    self._check_shape(f"output[{key}]", result[key], spec, "output")
        else:
            for key, spec in outputs.items():
                if hasattr(result, key):
                    self._check_shape(f"output.{key}", getattr(result, key), spec, "output")

    # --- Formatting helpers ---
    def _format_args(self, fn: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
        try:
            bound = inspect.signature(fn).bind_partial(*args, **kwargs)
            bound.apply_defaults()
            parts = []
            for name, value in bound.arguments.items():
                shape = self._get_shape(value)
                if shape:
                    parts.append(f"{name}={shape}")
                else:
                    parts.append(f"{name}={type(value).__name__}")
            return f"({', '.join(parts)})"
        except Exception:
            return "(uninspectable args)"

    def _format_result(self, result: Any) -> str:
        shape = self._get_shape(result)
        if shape:
            return f"shape={shape}"

        if isinstance(result, dict):
            keys = ", ".join(result.keys())
            return f"dict_keys=[{keys}]"

        attrs: Dict[str, Any] = {}
        for attr in ("policy", "policy_logits", "q_value", "value", "action"):
            if hasattr(result, attr):
                attrs[attr] = getattr(result, attr)
        if attrs:
            summary = ", ".join(
                f"{k}={self._get_shape(v) or type(v).__name__}" for k, v in attrs.items()
            )
            return summary

        return type(result).__name__


# Global singleton instance
sentinel = Sentinel()


