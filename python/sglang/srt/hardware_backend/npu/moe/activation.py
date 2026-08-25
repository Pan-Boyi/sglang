import functools
import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_all_gather,
)
from sglang.srt.layers.activation import GeluAndMul
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


# =============================================================================
# Abstract base for all activation variants
# =============================================================================
class BaseActivation(ABC):
    @abstractmethod
    def _apply_activation(
        self, *args, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]: ...


# =============================================================================
# SwiGLU clamp (DeepSeek "swiglu_limit")
# =============================================================================
@functools.lru_cache(maxsize=2)
def _log_swiglu_limit_once(limit: float) -> None:
    """Positive evidence in the log. A limit that never reaches the runner config silently
    disables the clamp, which is invisible from the outside -- print it once either way."""
    if limit > 0:
        logger.info("[SWIGLU] NPU expert clamp ACTIVE, limit=%.4g", limit)
    elif limit < 0:
        logger.warning(
            "[SWIGLU] NPU expert clamp DISABLED by KT_DISABLE_SWIGLU_CLAMP=1 "
            "-- experiment only, this deviates from the model definition"
        )
    else:
        logger.info("[SWIGLU] NPU expert clamp OFF (swiglu_limit unset or 0)")


# Experiment-only kill switch. It must disable the clamp on ALL THREE expert paths at once
# (NPU resident, streaming prefill, CPU offload) -- turning off only some of them would create
# a *new* inconsistency, which is exactly the bug this clamp fixes. The NPU and streaming paths
# both funnel through apply_swiglu_limit_ below; the CPU path is gated at the same-named check
# in layers/moe/kt_ep_wrapper.py where swiglu_limit is handed to kt-kernel.
# NEVER set this when serving: it deviates from the model definition.
def swiglu_clamp_disabled() -> bool:
    return os.environ.get("KT_DISABLE_SWIGLU_CLAMP", "") == "1"


@functools.lru_cache(maxsize=8)
def _swiglu_clamp_bounds(n_gate_up: int, limit: float, dtype: torch.dtype, device: str):
    """Per-column (min, max) vectors implementing the asymmetric clamp in one pass.

    Clamping the two halves separately means two *strided* passes over the tensor -- the
    halves of a [rows, 2I] buffer are non-contiguous views -- and measured only 186 GB/s.
    Folding the asymmetry into broadcast bound vectors turns it into a single contiguous
    elementwise op: 589 GB/s, 3.2x faster, bit-identical output.
    """
    half = n_gate_up // 2
    lo = torch.cat(
        [
            torch.full((half,), float("-inf")),  # gate: no lower bound
            torch.full((half,), -limit),         # up: two-sided
        ]
    ).to(dtype=dtype, device=device)
    hi = torch.full((n_gate_up,), limit, dtype=dtype, device=device)
    return lo, hi


def apply_swiglu_limit_(hidden_states: torch.Tensor, limit: Optional[float]) -> None:
    """Clamp the gate/up halves in place, matching DeepSeek's reference Expert.forward.

    The upstream definition (inference/model.py) is asymmetric and applies to routed *and*
    shared experts alike::

        up   = clamp(up, min=-limit, max=limit)   # two-sided
        gate = clamp(gate, max=limit)             # upper bound only -- SiLU already
                                                  # saturates on the negative side
        x = silu(gate) * up

    Doing it here, before the fused swiglu op, keeps the fusion intact: the kernel then
    computes silu(clamped_gate) * clamped_up, which is exactly the reference.

    No fused alternative exists in this operator release, which was checked rather than
    assumed: ``custom::npu_dequant_swiglu_clamp_quant`` accepts a ``clamp_limit`` but ignores
    it (output stays bit-identical to the unclamped op for every ``swiglu_mode`` 0-7), and
    ``custom::npu_swiglu_clip_quant`` computes the gpt-oss activation instead -- its
    ``group_alpha`` is the GLU alpha, not a clamp limit. ``npu_swiglu`` takes only a dim.

    ``activate_left=True`` is the convention at every call site, i.e. the left half is the
    silu input (gate) and the right half is up.

    Note: the reference clamps in fp32 while the gmm1 output reaching us is already bf16, so
    the clamp happens one rounding later. With a limit that is exactly representable (10.0)
    the two agree; this is the earliest point in the NPU pipeline where the value exists.
    """
    if swiglu_clamp_disabled():
        _log_swiglu_limit_once(-1.0)
        return
    if not limit or limit <= 0:
        _log_swiglu_limit_once(0.0)
        return
    _log_swiglu_limit_once(float(limit))
    lo, hi = _swiglu_clamp_bounds(
        hidden_states.shape[-1], float(limit), hidden_states.dtype,
        str(hidden_states.device),
    )
    torch.clamp_(hidden_states, min=lo, max=hi)


# =============================================================================
# Concrete activation implementations
# =============================================================================
class NPUSwiglu(BaseActivation):
    def __init__(self, swiglu_limit: Optional[float] = None):
        self._swiglu_limit = swiglu_limit

    def _apply_activation(self, hidden_states: torch.Tensor):
        apply_swiglu_limit_(hidden_states, self._swiglu_limit)
        return torch.ops.npu.npu_swiglu(hidden_states), None


class NPUSwigluQuant(BaseActivation):
    def __init__(self, swiglu_limit: Optional[float] = None):
        self._swiglu_limit = swiglu_limit

    def _apply_activation(self, hidden_states: torch.Tensor):
        apply_swiglu_limit_(hidden_states, self._swiglu_limit)
        hidden_states, swiglu_out_scale = torch.ops.npu.npu_dequant_swiglu_quant(
            hidden_states,
            quant_mode=1,
            activate_left=True,
        )
        return hidden_states, swiglu_out_scale


class NPUSwigluQuantWithScales(BaseActivation):
    def _apply_activation(
        self,
        hidden_states: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor,
        group_index: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        quant_scale: Optional[torch.Tensor] = None,
        quant_offset: Optional[torch.Tensor] = None,
    ):
        hidden_states, swiglu_out_scale = torch.ops.npu.npu_dequant_swiglu_quant(
            x=hidden_states,
            weight_scale=weight_scale,
            activation_scale=activation_scale,
            bias=bias,
            quant_scale=quant_scale,
            quant_offset=quant_offset,
            group_index=group_index,
            activate_left=True,
            quant_mode=1,
        )
        return hidden_states, swiglu_out_scale


class NPUSwigluDeepEPKernel(BaseActivation):
    """DeepEP grouped SwiGLU for the Ascend MoE runner; picks ``swiglu_quant`` vs the MiniMax
    SwiGLU-OAI variant (``swiglu_oai_quant``: ``gate*sigmoid(gate*alpha)*(up+1)`` w/ clamping)
    based on whether ``alpha``/``limit`` are given. The runner must forward
    ``gemm1_alpha``/``gemm1_clamp_limit`` here or experts fall back to wrong SwiGLU."""

    def __init__(
        self,
        need_quant: bool = True,
        alpha: Optional[float] = None,
        limit: Optional[float] = None,
    ):
        self.need_quant = need_quant
        self.alpha = alpha
        self.limit = limit
        self._use_oai = alpha is not None and limit is not None
        if self._use_oai:
            from sgl_kernel_npu.activation.swiglu_oai_quant import (
                swiglu_oai_quant,
            )

            self._kernel = swiglu_oai_quant
        else:
            from sgl_kernel_npu.activation.swiglu_quant import swiglu_quant

            self._kernel = swiglu_quant

    def _apply_activation(
        self,
        hidden_states: torch.Tensor,
        group_list: torch.Tensor,
        group_list_type: int,
    ):
        if self._use_oai:
            hidden_states, per_token_scale = self._kernel(
                hidden_states,
                self.alpha,
                self.limit,
                need_quant=self.need_quant,
                group_list=group_list,
                group_list_type=group_list_type,
            )
        else:
            hidden_states, per_token_scale = self._kernel(
                hidden_states, group_list, group_list_type, need_quant=self.need_quant
            )
        if self.need_quant:
            return hidden_states, per_token_scale
        return hidden_states, None


class NPUSitu(BaseActivation):
    """SiTU activation and optional INT8 requantization for grouped rows."""

    def __init__(
        self,
        *,
        need_quant: bool,
        beta: float = 4.0,
        linear_beta: Optional[float] = 25.0,
    ):
        from sgl_kernel_npu.activation.situ import situ

        self.situ = situ
        self.need_quant = need_quant
        self.beta = float(beta)
        self.linear_beta = None if linear_beta is None else float(linear_beta)

    def _apply_activation(
        self,
        hidden_states: torch.Tensor,
        group_list: torch.Tensor,
        group_list_type: int,
    ):
        return self.situ(
            hidden_states,
            group_list,
            group_list_type,
            need_quant=self.need_quant,
            beta=self.beta,
            linear_beta=self.linear_beta,
        )


class NPUGeluAndMul(BaseActivation):
    def __init__(self):
        self._gelu = GeluAndMul()

    def _apply_activation(self, hidden_states: torch.Tensor):
        return self._gelu(hidden_states), None


class NPUSwigluOAI(BaseActivation):
    def __init__(self, moe_runner_config=None):
        from sgl_kernel_npu.activation.swiglu_oai import swiglu_oai_triton

        self._kernel = swiglu_oai_triton
        self._moe_runner_config = moe_runner_config

    def _apply_activation(self, hidden_states: torch.Tensor):
        # hidden_states is the output of the grouped matmul with shape
        # [num_tokens, 2 * inter].  The old swiglu_oai kernel derived the
        # gate_up dimension from layer.w13_weight.shape[2], which now fails
        # because w13_weight is stored un-transposed.  Instead we pass
        # the gate_up dimension explicitly from the tensor itself.
        alpha = 1.0
        clamp = None
        if self._moe_runner_config is not None:
            alpha = getattr(self._moe_runner_config, "gemm1_alpha", 1.0)
            clamp = getattr(self._moe_runner_config, "gemm1_clamp_limit", None)

        output = self._kernel(
            hidden_states,
            hidden_states.shape[-1],  # gate_up dim = 2 * inter
            alpha,
            clamp,
        )
        return output, None


class NPUSwigluStepAndMul(BaseActivation):
    def __init__(self, clamp_limit: Optional[float] = None):
        self._clamp_limit = clamp_limit

    def _apply_activation(self, hidden_states: torch.Tensor):
        if self._clamp_limit is not None:
            return self._swiglustep_and_mul(hidden_states, self._clamp_limit), None
        return torch.ops.npu.npu_swiglu(hidden_states), None

    @staticmethod
    def _swiglustep_and_mul(x: torch.Tensor, limit: float = 7.0) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        gate = F.silu(gate).clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        return gate * up


# =============================================================================
# Generic TP all‑gather wrapper – used by the runner when needed
# =============================================================================
class AllGatherActivationWrapper(BaseActivation):
    """
    Wraps any activation and adds an all‑gather along `dim` if TP > 1.

    This allows the runner to stay TP‑agnostic: the wrapper is applied
    transparently at construction time.
    """

    def __init__(self, inner: BaseActivation, dim: int = -1):
        self.inner = inner
        self.dim = dim

    def _apply_activation(self, *args, **kwargs):
        out, scale = self.inner._apply_activation(*args, **kwargs)
        if get_parallel().tp_size > 1:
            out = tensor_model_parallel_all_gather(out, dim=self.dim)
        return out, scale


# =============================================================================
# Factory (unchanged, returns *base* activations)
# =============================================================================
def get_swiglu_variant(method: str, **kwargs: Any) -> BaseActivation:
    variants: dict[str, type[BaseActivation]] = {
        "standard": NPUSwiglu,
        "dequant_swiglu_quant": NPUSwigluQuant,
        "dequant_swiglu_quant_with_scales": NPUSwigluQuantWithScales,
        "swiglu_quant_deepep_kernel": NPUSwigluDeepEPKernel,
        "gelu_and_mul": NPUGeluAndMul,
    }
    if method == "swiglu_oai":
        # The OAI variant now uses the triton kernel that derives the gate_up
        # dimension from the tensor itself.  No extra parameters are needed.
        return NPUSwigluOAI()
    if method == "swiglustep_and_mul":
        clamp_limit = kwargs.pop("clamp_limit", None)
        return NPUSwigluStepAndMul(clamp_limit=clamp_limit)
    if method not in variants:
        raise ValueError(f"Unknown SwiGLU variant: {method}")
    return variants[method]()
