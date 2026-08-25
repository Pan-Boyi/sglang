# SPDX-License-Identifier: Apache-2.0
"""Streaming prefill for the KT MoE offload: stream ALL routed experts per layer DDR->HBM.

Env-gated bypass of the hybrid (resident-on-NPU + CPU) MoE path.  During a long
prefill (``KT_PREFILL_STREAM=1`` and chunk token count ``M >=
KT_PREFILL_STREAM_THRESHOLD``) each layer streams its full expert set from a
pinned DDR pool into a single reused HBM slot and runs the whole MoE over all
experts on the NPU -- no CPU experts, no submit/sync round trip.

Design notes:
- Serial single slot (double buffering measured <=2% for twice the HBM).
- The pool is built at model-load time when possible, lazily on the first
  qualifying forward otherwise; the ND->NZ cast is chunked to bound peak HBM.
- Pure bypass: the streaming path never touches the CPU wrapper, so it is
  orthogonal to the hybrid path.
- Any failure returns ``None`` and the caller falls through to the hybrid path.
  With ``KT_PREFILL_STREAM`` unset this module is inert.

Entry points:
- :func:`maybe_reserve_slot` from ``KTEPWrapperMethod.process_weights_after_loading``
  (model-load time),
- :func:`maybe_streaming_forward` from ``FusedMoE.forward_impl``, *before* the
  dispatcher runs.  It has to sit there rather than inside ``quant_method.apply``
  because ``AscendTPDispatcher.dispatch`` has already permuted and int8-quantised
  the hidden states by the time ``apply`` is reached, and because the KT
  pre-dispatch hook (CPU submit + expert-id masking) must not run for a streamed
  layer.  A dispatcher hook cannot express that: hooks transform the dispatch
  inputs, they cannot short-circuit the dispatch.
"""

import functools
import json
import logging
import os
from typing import Optional

import torch

from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod

logger = logging.getLogger(__name__)

_KT_PREFILL_STREAM = os.environ.get("KT_PREFILL_STREAM", "") == "1"
_T = int(os.environ.get("KT_PREFILL_STREAM_THRESHOLD", "512"))
# KT_HOT_TAIL_TOKENS=N (opt-in; default 0 = off = whole-prompt selection): pick the decode resident
# hot pool from only the LAST N prompt tokens' routing instead of the whole prefill.  Decode
# continues from the prompt tail, so the tail's expert distribution predicts decode routing better
# -> higher decode hit rate.  Long contexts gain, short prompts (<~2k tokens) regress slightly (a
# short prompt is already all "recent"), hence default off.  Pure perf knob: it only changes which
# experts are NPU-resident vs CPU, not the computed output.
_HOT_TAIL = int(os.environ.get("KT_HOT_TAIL_TOKENS", "0") or "0")
_CKPT = os.environ.get("KT_PREFILL_STREAM_CKPT", "")
_NZ_CHUNK = int(
    os.environ.get("KT_PREFILL_STREAM_NZ_CHUNK", "64")
)  # experts/chunk for ND->NZ
_ACL_FORMAT_FRACTAL_NZ = 29

# DEPOOL (KT_MXFP4_DEPOOL=1): instead of a resident W8A8 NZ pool (~277GB for DeepSeek-V4), store the
# MXFP4 codes+scale (~137GB, 4-bit) and convert MXFP4 -> W8A8-NZ on the fly per layer with the fused
# AscendC kernel (KT_MXFP4_OP_DIR), hidden under the H2D.  Fully gated: when off, the W8A8 path
# below is unchanged.  The MXFP4 weights are the original safetensors (.weight = codes, .scale =
# e8m0), NOT the W8A8 checkpoint.
_KT_MXFP4_DEPOOL = os.environ.get("KT_MXFP4_DEPOOL", "") == "1"
# KT_MXFP4_POOL_NO_PIN=1: store the MXFP4 pool in pageable (unpinned) host memory.  Pinning the
# ~140GB pool inflates the decode CPU-MoE wall (pin tax); unpinning removes it at the cost of a
# slower streaming prefill H2D (no async DMA).  Default pinned (fast prefill).
_PIN_MXFP4 = os.environ.get("KT_MXFP4_POOL_NO_PIN", "") != "1"
_MXFP4_CKPT = os.environ.get("KT_MXFP4_CKPT", "")

# GGUF DEDUP (KT_MXFP4_GGUF_DEDUP=1, requires KT_MXFP4_DEPOOL=1): read the layer's MXFP4 codes
# straight from the per-layer GGUF (KT_GGUF_TEMPLATE, block_mxfp4 = e8m0 + half-block-packed codes)
# that the CPU MoE already holds, instead of ALSO keeping a separate ~137GB pinned codes pool.
#
# Sharing is real but partial, so this stays default-off.  kt-kernel does map the GGUF
# (``kt-kernel/python/utils/loader.py`` hands ``moe.hpp`` a view over an ``np.memmap``), so the
# reader below and the CPU MoE hit the same page cache and the pinned pool is genuinely recovered.
# What is NOT recovered is kt-kernel's own copy: ``LLAMA_MOE_TP::load_weights`` memcpys each NUMA
# subpool's ``intermediate_size / KT_THREADPOOL_COUNT`` slice into node-local buffers, so ~137GB of
# anonymous DDR stays resident whatever this flag does.  With ``--kt-threadpool-count 8`` that copy
# is unavoidable -- the zero-copy alias only applies to a single un-split pool -- so enabling this
# trades a pinned, non-evictable pool for an evictable page cache that must survive alongside those
# copies.  Enable it only when the box has the headroom to keep the GGUF cached; otherwise every
# long prefill re-reads it from disk.
_KT_GGUF_DEDUP = os.environ.get("KT_MXFP4_GGUF_DEDUP", "") == "1"
_GGUF_TMPL = os.environ.get("KT_GGUF_TEMPLATE", "")
_GGUF_READERS: dict = {}  # layer_idx -> GGUFReader (memmap)
_GGUF_BLOCKS: dict = (
    {}
)  # layer_idx -> (gate, up, down) np memmap views [E,N,nb*17] block_mxfp4

_MXFP4_POOL: dict = {}  # layer_idx -> (c13, s13, c2, s2) pinned host MXFP4 (codes+e8m0)
_MXFP4_POOL_BUILT = False  # set once the pool is fully populated
_MXSTAGE: dict = (
    {}
)  # shape -> reused pinned [K,...] staging buf for the dyn-resident switch
_MXIDX = None  # cached weight_map of the MXFP4 checkpoint index

# ``npu_moe_init_routing_v2(expert_tokens_num_type=1)`` returns the per-expert token COUNT, which is
# what ``npu_grouped_matmul(group_list_type=1)`` expects; v1 routing
# (``npu_moe_compute_expert_tokens``) returns the CUMULATIVE form, which is ``group_list_type=0``.
# The two must be kept in step: a mismatch is NOT rejected by the operator, it silently computes the
# wrong result.  This module uses v2 routing, hence 1 -- the same pairing AscendTPDispatcher uses
# (see token_dispatcher/ascend_tp.py, which sets group_list_type=1 for every v2 variant).
_GROUP_LIST_TYPE = 1


def _require_ckpt_dir(path: str, env_name: str, what: str) -> str:
    """Return the checkpoint dir, or raise with guidance when the env var is unset."""
    if not path:
        raise ValueError(
            f"{env_name} is not set, but it is required to load the {what} checkpoint. "
            f"Set {env_name}=/path/to/checkpoint."
        )
    return path


def _ckpt_dir() -> str:
    return _require_ckpt_dir(_CKPT, "KT_PREFILL_STREAM_CKPT", "W8A8 (NPU-side)")


def _mxfp4_ckpt_dir() -> str:
    return _require_ckpt_dir(_MXFP4_CKPT, "KT_MXFP4_CKPT", "native MXFP4 (CPU-side)")


def _add_sys_path(d: str) -> None:
    import sys

    if d not in sys.path:
        sys.path.insert(0, d)


def _gguf_reader_cls():
    """Return ``gguf.GGUFReader``.

    Prefer the installed ``gguf`` distribution; fall back to a checkout pointed at by
    ``KT_GGUF_PY_DIR`` (``<llama.cpp>/gguf-py``).  The repository-relative discovery the
    original patch used does not apply here: sglang and ktransformers are independent
    clones, not one nested inside the other.
    """
    try:
        from gguf import GGUFReader

        return GGUFReader
    except ImportError:
        pass
    d = os.environ.get("KT_GGUF_PY_DIR")
    if not d:
        raise ImportError(
            "the `gguf` package is not importable and KT_GGUF_PY_DIR is not set. "
            "Install gguf (pip install gguf) or set KT_GGUF_PY_DIR=<llama.cpp>/gguf-py."
        )
    _add_sys_path(d)
    from gguf import GGUFReader

    return GGUFReader


def _gguf_layer_blocks(layer: int):
    """Return (gate, up, down) block_mxfp4 memmap views [E,N,nb*17] for one layer.

    Lazily opens and caches one GGUFReader per layer; ``t.data`` is a file-backed memmap.
    """
    blk = _GGUF_BLOCKS.get(layer)
    if blk is None:
        r = _GGUF_READERS.get(layer)
        if r is None:
            r = _gguf_reader_cls()(_GGUF_TMPL.format(layer_idx=layer))
            _GGUF_READERS[layer] = r
        byname = {t.name: t for t in r.tensors}
        blk = tuple(
            byname[f"blk.{layer}.{n}.weight"].data
            for n in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
        )
        _GGUF_BLOCKS[layer] = blk
    return blk


# ----- prefetch (double-buffered) -----
# The per-layer CPU memcpy mmap->pinned is device-independent, so a worker thread fills layer L+1's
# pinned staging while the main thread is blocked in layer L's convert syncs.  Needs PING-PONG
# buffers (2 per key, alternating by layer parity): a single buffer would race the in-flight H2D.
# No device event is needed: the main thread is serial and each convert syncs, so layer L-1's H2D
# (from the buffer the worker reuses for L+1) has finished before L starts.
_KT_PREFETCH = os.environ.get("KT_MXFP4_PREFETCH", "1") == "1"
_MX_PP: dict = {}  # key -> [buf0, buf1] pinned ping-pong staging
_PF = {
    "ex": None,
    "futs": {},
    "next": None,
}  # worker, layer->future, expected next layer


def _pp_buf(key, parity, E, OUT, nb17):
    bufs = _MX_PP.get(key)
    if bufs is None:
        bufs = [None, None]
        _MX_PP[key] = bufs
    b = bufs[parity]
    if b is None or tuple(b.shape) != (E, OUT, nb17):
        b = torch.empty((E, OUT, nb17), dtype=torch.uint8, pin_memory=True)
        bufs[parity] = b
    return b


_COPY_POOL = None
_COPY_NTHREADS = int(os.environ.get("KT_MXFP4_COPY_THREADS", "32"))


def _par_copy(dst, src_np):
    """Copy ``src_np`` (GGUF memmap) -> ``dst`` (pinned), parallelised over the expert dim.

    ``torch.copy_`` runs single-threaded in the server (the OMP pool is saturated by the
    kt-cpuinfer threads), which is the whole long-prefill bottleneck on a slow single-core
    memory path.  Explicit threads each release the GIL inside ``copy_``.
    ``KT_MXFP4_COPY_THREADS=0`` restores the single-threaded copy.
    """
    global _COPY_POOL
    src = torch.from_numpy(src_np)
    E = src.shape[0]
    n = _COPY_NTHREADS
    if n <= 1 or E < n:
        dst.copy_(src)
        return
    if _COPY_POOL is None:
        import concurrent.futures

        _COPY_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=n)
    futs = [
        _COPY_POOL.submit(
            lambda lo, hi: dst[lo:hi].copy_(src[lo:hi]), E * k // n, E * (k + 1) // n
        )
        for k in range(n)
    ]
    for f in futs:
        f.result()


def _fill_stage(layer):
    """Copy this layer's GGUF blocks into its parity's pinned ping-pong buffers.

    ``w13 = cat(gate, up)`` along OUT, ``w2 = down``.  H2D and the de-interleave stay on the
    main thread / in the kernel.
    """
    gate, up, down = _gguf_layer_blocks(layer)
    par = layer % 2
    E = gate.shape[0]
    b13 = _pp_buf("w13", par, E, gate.shape[1] + up.shape[1], gate.shape[2])
    _par_copy(b13[:, : gate.shape[1]], gate)
    _par_copy(b13[:, gate.shape[1] :], up)
    b2 = _pp_buf("w2", par, E, down.shape[1], down.shape[2])
    _par_copy(b2, down)


def _prefetch_ensure(layer, num_layers):
    """Ensure ``layer``'s buffers are filled, then kick off ``layer + 1``.

    Waits for this layer's prefetch, or fills synchronously on a new prefill / out-of-sequence
    layer.  Returns ``layer % 2``.
    """
    import concurrent.futures

    if _PF["ex"] is None:
        _PF["ex"] = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    if _PF["next"] == layer and layer in _PF["futs"]:
        _PF["futs"].pop(layer).result()  # warm: the worker already filled it
    else:
        _PF["futs"].clear()  # new prefill / resync -> fill this layer now
        _fill_stage(layer)
    nxt = layer + 1
    if nxt < num_layers:
        _PF["futs"][nxt] = _PF["ex"].submit(_fill_stage, nxt)
    _PF["next"] = nxt
    return layer % 2


def _stage_pin_h2d(src, idx_cpu, dev):
    """Gather ``src[idx_cpu]`` (K hot experts) into a reused pinned buffer, then DMA to ``dev``.

    Plain advanced indexing returns an UNpinned tensor, and the following H2D then loses DMA.
    ``index_select`` into a pinned out buffer keeps the copy on the DMA path.  Works whether or
    not the pool itself is pinned.
    """
    K = int(idx_cpu.numel())
    shp = (K,) + tuple(src.shape[1:])
    buf = _MXSTAGE.get(shp)
    if buf is None:
        buf = torch.empty(shp, dtype=src.dtype, pin_memory=True)
        _MXSTAGE[shp] = buf
    torch.index_select(src, 0, idx_cpu, out=buf)
    return buf.to(dev, non_blocking=True)


def _mxfp4_op_dir_on_path() -> None:
    """Put the AscendC MXFP4 operator wrapper on ``sys.path``.

    Only ``KT_MXFP4_OP_DIR`` is honoured.  The original patch also guessed the directory from
    a fixed ``<ktransformers>/third_party/sglang/...`` nesting, which does not exist here --
    the two repositories are independent clones -- so a wrong guess would silently import
    nothing useful.
    """
    d = os.environ.get("KT_MXFP4_OP_DIR")
    if not d:
        raise ValueError(
            "KT_MXFP4_OP_DIR is not set, but the MXFP4 depool path needs the fused AscendC "
            "operator wrapper. Set "
            "KT_MXFP4_OP_DIR=<ktransformers>/kt-kernel/tools/ascendc_mxfp4."
        )
    _add_sys_path(d)


def _mxfp4_convert_fn():
    """Lazily import the fused-kernel wrapper (``mxfp4_fused_op.py``)."""
    _mxfp4_op_dir_on_path()
    from mxfp4_fused_op import mxfp4_layer_to_nz_slots

    return mxfp4_layer_to_nz_slots


def _mxfp4_convert_blk_fn():
    """Wrapper that converts straight from RAW GGUF blocks (in-kernel de-interleave)."""
    _mxfp4_op_dir_on_path()
    from mxfp4_fused_op import mxfp4_layer_to_nz_slots_blk

    return mxfp4_layer_to_nz_slots_blk


_KT_BLK_KERNEL = os.environ.get("KT_MXFP4_BLK_KERNEL", "1") == "1"

# Module-level singletons (shared across all layers / wrapper instances).
_POOL: dict = {}  # layer_idx -> (w13_host_nz, w2_host_nz, s13_bf16_npu, s2_bf16_npu)
_SLOT: dict = {}  # 'w13'/'w2' -> reused NZ HBM slot
_POOL_BUILT = False
_SLOT_RESERVED = False

# Dynamic decode-resident expert pool.  During a streaming prefill we count per-layer expert
# activations (device-side bincount, cheap); the per-layer top-K (K = the number of resident slots)
# then replaces the static-prefix resident set.  Weights are gathered from the pool into the
# resident params and all routing structures are updated IN PLACE (same storage, so decode graph
# replay and the C++ side observe them):
#   1. KTEPWrapperMethod.gpu_experts_mask / logical_to_gpu_index (device tensors)
#   2. kt_kernel wrapper.gpu_experts_mask (pinned CPU bool, shared with C++ by pointer)
_KT_DYN_RESIDENT = os.environ.get("KT_DYNAMIC_RESIDENT", "") == "1"
_REQ_HIST: dict = {}  # layer_idx -> int64 device tensor [E] (current prefill pass)
_REGISTRY: dict = {}  # layer_idx -> (layer_module, ktep_wrapper)

# Incremental build: capture MATERIALISES each expert's int8 weight straight into that layer's FINAL
# pinned ND buffer as the load loop reads it.  Once a layer's tensors are all in, the layer is
# NZ-cast IN PLACE (chunked: pinned ND -> HBM -> transpose + format_cast -> bytes back into the SAME
# pinned region; ND[E,A,B] and NZ[E,B,A] have identical byte counts).  The whole build is spread
# inside the model-load loop, so the extra peak DDR is zero -- the pinned pool IS the product.
_CFG: dict = (
    {}
)  # E, H, I, num_layers (from the wrapper, or from the checkpoint config.json)
_LBUF: dict = {}  # layer -> {flat13, flat2 (pinned int8), s13, s2 (cpu fp32), count}


def _remember_dims(E: int, H: int, I: int, num_layers: int) -> None:
    """Cache the MoE dimensions taken from the layer wrapper.

    Preferred over reading them back out of ``config.json``: the GGUF-dedup path needs no
    safetensors checkpoint at all, so it must not be forced to name one.
    """
    if E and H and I and num_layers:
        _CFG.update(E=E, H=H, I=I, num_layers=num_layers)


def _get_cfg():
    if not _CFG:
        cfg = json.load(open(os.path.join(_ckpt_dir(), "config.json")))
        _CFG["E"] = int(cfg["n_routed_experts"])
        _CFG["H"] = int(cfg["hidden_size"])
        _CFG["I"] = int(cfg["moe_intermediate_size"])
        _CFG["num_layers"] = int(cfg["num_hidden_layers"])
    return _CFG["E"], _CFG["H"], _CFG["I"], _CFG["num_layers"]


def _layer_buf(L: int, E: int, H: int, I: int) -> dict:
    b = _LBUF.get(L)
    if b is None:
        b = {
            "flat13": torch.empty(E * 2 * I * H, dtype=torch.int8, pin_memory=True),
            "flat2": torch.empty(E * H * I, dtype=torch.int8, pin_memory=True),
            "s13": torch.empty(E, 2 * I, 1, dtype=torch.float32),
            "s2": torch.empty(E, H, 1, dtype=torch.float32),
            "count": 0,
        }
        _LBUF[L] = b
    return b


# ----- parallel O_DIRECT pool reader -----
# The build bottleneck is reading the expert int8 out of the W8A8 checkpoint.  The loader's buffered
# single-thread reads are the floor; parallel O_DIRECT (bypassing the page cache) plus a per-expert
# rearrange is several times faster.  Raw O_DIRECT is faster still, but the Python per-expert
# rearrange (experts are expert-major yet scattered on disk) is then the hard cap.
_NVME_ALIGN = 4096
_IDX = None


def _index():
    global _IDX
    if _IDX is None:
        _IDX = json.load(
            open(os.path.join(_ckpt_dir(), "model.safetensors.index.json"))
        )["weight_map"]
    return _IDX


def _shard_header(path):
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def _read_layer_odirect(L, E, H, I, scratch) -> None:
    """Fill ``_LBUF[L]`` (pinned flat13/flat2 + scales) via O_DIRECT reads + per-expert rearrange.

    ``scratch``: a page-aligned mmap at least as large as one shard's expert region (reused).
    """
    from safetensors import safe_open

    idx = _index()
    b = _layer_buf(L, E, H, I)
    f13 = b["flat13"].view(E, 2 * I, H)
    f2 = b["flat2"].view(E, H, I)
    byfile = {}
    for e in range(E):
        for w in ("w1", "w2", "w3"):
            byfile.setdefault(idx[f"layers.{L}.ffn.experts.{e}.{w}.weight"], []).append(
                (e, w)
            )
    for fn, items in byfile.items():
        path = os.path.join(_ckpt_dir(), fn)
        hdr, base = _shard_header(path)
        offs = {
            (e, w): hdr[f"layers.{L}.ffn.experts.{e}.{w}.weight"]["data_offsets"]
            for e, w in items
        }
        lo = min(o[0] for o in offs.values())
        hi = max(o[1] for o in offs.values())
        region = _odirect_region(path, base, lo, hi, scratch)
        for e, w in items:
            o0, o1 = offs[(e, w)]
            blk = torch.frombuffer(region[o0 - lo : o1 - lo], dtype=torch.int8)
            if w == "w1":
                f13[e, 0:I].copy_(blk.view(I, H))
            elif w == "w3":
                f13[e, I : 2 * I].copy_(blk.view(I, H))
            else:
                f2[e].copy_(blk.view(H, I))
    # scales (tiny) via normal get_tensor
    sfiles = {}
    for e in range(E):
        for w in ("w1", "w2", "w3"):
            sfiles.setdefault(
                idx[f"layers.{L}.ffn.experts.{e}.{w}.weight_scale"], []
            ).append((e, w))
    for fn, items in sfiles.items():
        with safe_open(os.path.join(_ckpt_dir(), fn), framework="pt") as f:
            for e, w in items:
                t = f.get_tensor(f"layers.{L}.ffn.experts.{e}.{w}.weight_scale")
                if w == "w1":
                    b["s13"][e, 0:I] = t.reshape(I, 1)
                elif w == "w3":
                    b["s13"][e, I : 2 * I] = t.reshape(I, 1)
                else:
                    b["s2"][e] = t.reshape(H, 1)


_BG = {"ex": None, "done_q": None, "t_start": 0.0, "started": False}


def _start_bg_reads(E, H, I, num_layers, nworkers=8) -> None:
    """Start the O_DIRECT read workers in the BACKGROUND (host-only, no HBM).

    They overlap the rest of model load: the reads contend with the load on NVMe but use its
    NPU/CPU phases freely.  The NZ cast happens later in :func:`_finish_bg_build`, which needs
    HBM scratch.
    """
    import mmap
    import queue
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    _BG["done_q"] = queue.Queue()
    _BG["t_start"] = time.perf_counter()
    _BG["ex"] = ThreadPoolExecutor(max_workers=nworkers)
    _tls = threading.local()

    def rd(L):
        if not hasattr(_tls, "scratch"):
            _tls.scratch = mmap.mmap(
                -1, 8 * 1024**3
            )  # page-aligned, per-thread, reused
        _read_layer_odirect(L, E, H, I, _tls.scratch)
        _BG["done_q"].put(L)

    for L in range(num_layers):
        _BG["ex"].submit(rd, L)
    logger.info(
        "[KT_STREAM] background reads started (%d layers, %d workers), overlapping load",
        num_layers,
        nworkers,
    )


def _finish_bg_build(num_layers, dev) -> None:
    """Drain the background reads, NZ-casting each layer as its read completes."""
    import time

    _free_slot()  # its HBM is the NZ-cast scratch
    t0 = time.perf_counter()
    for _ in range(num_layers):
        L = _BG["done_q"].get()
        _finalize_layer(L, dev)
    _BG["ex"].shutdown()
    logger.info(
        "[KT_STREAM] pool build done: total %.0fs (NZ drain %.0fs, %d layers)",
        time.perf_counter() - _BG["t_start"],
        time.perf_counter() - t0,
        num_layers,
    )


def _build_pool_parread(E, H, I, num_layers, dev, nworkers=8) -> None:
    """Non-overlapped path (lazy fallback): start the reads then immediately drain + NZ."""
    _start_bg_reads(E, H, I, num_layers, nworkers)
    _finish_bg_build(num_layers, dev)


def _inplace_nz(flat: torch.Tensor, E: int, A: int, B: int, dev) -> torch.Tensor:
    """Chunked in-place ND[E,A,B] -> FRACTAL_NZ[E,B,A] over the same pinned bytes."""
    import torch_npu

    nd = flat.view(E, A, B)
    nz_host = flat.view(E, B, A)
    for c in range(0, E, _NZ_CHUNK):
        sub = nd[c : c + _NZ_CHUNK].to(dev).transpose(1, 2).contiguous()
        nz = torch_npu.npu_format_cast(sub, _ACL_FORMAT_FRACTAL_NZ)
        nz_host[c : c + _NZ_CHUNK].copy_(nz)
        del sub, nz
    torch.npu.empty_cache()
    return nz_host


def _finalize_layer(L: int, dev) -> None:
    """NZ-cast a completed layer in place and publish it to the pool."""
    global _POOL_BUILT
    import time

    E, H, I, num_layers = _get_cfg()
    b = _LBUF[L]
    t0 = time.perf_counter()
    h13 = _inplace_nz(b["flat13"], E, 2 * I, H, dev)  # -> [E, H, 2I] NZ view
    h2 = _inplace_nz(b["flat2"], E, H, I, dev)  # -> [E, I, H] NZ view
    s13b = b["s13"].squeeze(-1).to(torch.bfloat16).to(dev)
    s2b = b["s2"].squeeze(-1).to(torch.bfloat16).to(dev)
    _POOL[L] = (h13, h2, s13b, s2b)
    b["s13"] = b["s2"] = None
    if len(_POOL) == num_layers:
        _POOL_BUILT = True
    logger.info(
        "[KT_STREAM] layer %d NZ-finalized in-loop (%.1fs, %d/%d done)",
        L,
        time.perf_counter() - t0,
        len(_POOL),
        num_layers,
    )


def _is_prefill() -> bool:
    try:
        return not torch.npu.is_current_stream_capturing()
    except Exception:
        return True


# ---------------------------------------------------------------------------
#  DEPOOL: load MXFP4 codes+scale (instead of W8A8) and build a small pinned pool
# ---------------------------------------------------------------------------
def _as_u8(t):
    return (t if t.dtype == torch.uint8 else t.view(torch.uint8)).contiguous()


def _load_layer_mxfp4(layer: int, E: int, H: int, I: int):
    """Read one layer's E experts of native MXFP4 (codes + e8m0 scale) and build w13 = cat(w1,w3).

    Returns pinned host tensors: c13 [E,2I,H/2] u8, s13 [E,2I,H/32] u8, c2 [E,H,I/2] u8,
    s2 [E,H,I/32] u8.
    """
    from safetensors import safe_open

    idx = _mxfp4_index()
    cache: dict = {}

    def _open(k):
        sh = idx[k]
        if sh not in cache:
            cache[sh] = safe_open(os.path.join(_mxfp4_ckpt_dir(), sh), framework="pt")
        return cache[sh]

    def stack(proj):
        cs, ss = [], []
        for e in range(E):
            wk = f"layers.{layer}.ffn.experts.{e}.{proj}.weight"
            sk = f"layers.{layer}.ffn.experts.{e}.{proj}.scale"
            h = _open(wk)
            cs.append(_as_u8(h.get_tensor(wk)))
            ss.append(_as_u8(h.get_tensor(sk)))
        return torch.stack(cs), torch.stack(ss)

    _pin = (lambda t: t.pin_memory()) if _PIN_MXFP4 else (lambda t: t)
    c1, s1 = stack("w1")
    c3, s3 = stack("w3")
    c13 = _pin(torch.cat([c1, c3], dim=1))
    s13 = _pin(torch.cat([s1, s3], dim=1))
    c2, s2 = stack("w2")
    return c13, s13, _pin(c2), _pin(s2)


def _build_mxfp4_pool(E: int, H: int, I: int, num_layers: int) -> None:
    """Serial fallback: fill ``_MXFP4_POOL`` with pinned MXFP4 codes+scale per layer.

    The fast path is the load-time parallel O_DIRECT build
    (:func:`_start_bg_reads_mxfp4`); this ``safe_open`` reader only runs if that failed or was
    never started.
    """
    global _MXFP4_POOL_BUILT
    import time

    if _MXFP4_POOL_BUILT:
        return
    _MXFP4_POOL.clear()  # drop any partial buffers from a failed parallel build
    t0 = time.perf_counter()
    logger.info(
        "[KT_STREAM][depool] building MXFP4 pool (serial): %d layers from %s",
        num_layers,
        _mxfp4_ckpt_dir(),
    )
    for L in range(num_layers):
        _MXFP4_POOL[L] = _load_layer_mxfp4(L, E, H, I)
    _MXFP4_POOL_BUILT = True
    logger.info(
        "[KT_STREAM][depool] MXFP4 pool built in %.0fs", time.perf_counter() - t0
    )


# ----- load-time parallel O_DIRECT MXFP4 pool reader (mirrors the W8A8 reader) -----
# The depool pool is just pinned host MXFP4 codes+scale (no NZ cast -- the bytes ARE the product),
# so building it is purely a read problem.  Reading all layers in parallel with O_DIRECT, started at
# model-load time, overlaps the rest of the load.  MXFP4 is 4-bit, so the reads are cheap.
_BG_MX = {"ex": None, "done_q": None, "t_start": 0.0, "started": False}


def _mxfp4_index():
    global _MXIDX
    if _MXIDX is None:
        _MXIDX = json.load(
            open(os.path.join(_mxfp4_ckpt_dir(), "model.safetensors.index.json"))
        )["weight_map"]
    return _MXIDX


def _odirect_region(path, base, lo, hi, scratch):
    """O_DIRECT-read ``[base+lo, base+hi)`` into the page-aligned ``scratch`` mmap.

    Returns a memoryview of exactly the ``[lo, hi)`` payload.
    """
    a_lo = ((base + lo) // _NVME_ALIGN) * _NVME_ALIGN
    skip = (base + lo) - a_lo
    need = (base + hi) - a_lo
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        dv = memoryview(scratch)
        got = 0
        while got < need:
            n = os.preadv(fd, [dv[got:]], a_lo + got)
            if n <= 0:
                break
            got += n
    finally:
        os.close(fd)
    return memoryview(scratch)[skip : skip + (hi - lo)]


def _mxfp4_layer_buf(L, E, H, I):
    """Get (or allocate) layer L's pinned destination tensors.

    The pinned buffer IS the pool product (no NZ round trip), so it is filled in place:
    c13/s13 = cat(w1,w3) along OUT, c2/s2 = w2.
    """
    b = _MXFP4_POOL.get(L)
    if b is None:
        pin = _PIN_MXFP4
        b = (
            torch.empty(
                E, 2 * I, H // 2, dtype=torch.uint8, pin_memory=pin
            ),  # c13 codes
            torch.empty(
                E, 2 * I, H // 32, dtype=torch.uint8, pin_memory=pin
            ),  # s13 e8m0
            torch.empty(E, H, I // 2, dtype=torch.uint8, pin_memory=pin),  # c2 codes
            torch.empty(E, H, I // 32, dtype=torch.uint8, pin_memory=pin),  # s2 e8m0
        )
        _MXFP4_POOL[L] = b
    return b


def _read_layer_mxfp4_odirect(L, E, H, I, scratch) -> None:
    """Fill ``_MXFP4_POOL[L]`` via O_DIRECT reads + per-expert rearrange.

    Codes (``.weight``) and scales (``.scale``) sit in two separate contiguous on-disk blocks,
    so each is one tight region read per shard file.  Byte-equivalent to
    :func:`_load_layer_mxfp4`.
    """
    idx = _mxfp4_index()
    c13, s13, c2, s2 = _mxfp4_layer_buf(L, E, H, I)
    for suf, (dst13, dst2, w13, w2_n) in (
        ("weight", (c13, c2, H // 2, I // 2)),
        ("scale", (s13, s2, H // 32, I // 32)),
    ):
        byfile = {}
        for e in range(E):
            for proj in ("w1", "w2", "w3"):
                k = f"layers.{L}.ffn.experts.{e}.{proj}.{suf}"
                byfile.setdefault(idx[k], []).append((e, proj))
        for fn, items in byfile.items():
            path = os.path.join(_mxfp4_ckpt_dir(), fn)
            hdr, base = _shard_header(path)
            offs = {
                (e, proj): hdr[f"layers.{L}.ffn.experts.{e}.{proj}.{suf}"][
                    "data_offsets"
                ]
                for e, proj in items
            }
            lo = min(o[0] for o in offs.values())
            hi = max(o[1] for o in offs.values())
            region = _odirect_region(path, base, lo, hi, scratch)
            for e, proj in items:
                o0, o1 = offs[(e, proj)]
                blk = torch.frombuffer(region[o0 - lo : o1 - lo], dtype=torch.uint8)
                if proj == "w1":
                    dst13[e, 0:I].copy_(blk.view(I, w13))
                elif proj == "w3":
                    dst13[e, I : 2 * I].copy_(blk.view(I, w13))
                else:
                    dst2[e].copy_(blk.view(H, w2_n))


def _start_bg_reads_mxfp4(E, H, I, num_layers, nworkers=8) -> None:
    """Submit the O_DIRECT MXFP4 read workers in the BACKGROUND (host-only, no HBM).

    Each worker fills a layer's pinned pool buffers in place; the build is done once all are
    drained (there is no NZ pass for depool).
    """
    import mmap
    import queue
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    _BG_MX["done_q"] = queue.Queue()
    _BG_MX["t_start"] = time.perf_counter()
    _BG_MX["ex"] = ThreadPoolExecutor(max_workers=nworkers)
    _tls = threading.local()

    def rd(L):
        try:
            if not hasattr(_tls, "scratch"):
                _tls.scratch = mmap.mmap(
                    -1, 4 * 1024**3
                )  # >= one layer's weight region
            _read_layer_mxfp4_odirect(L, E, H, I, _tls.scratch)
            _BG_MX["done_q"].put(L)
        except Exception as e:
            _BG_MX["done_q"].put(("ERR", L, repr(e)[:200]))

    for L in range(num_layers):
        _BG_MX["ex"].submit(rd, L)
    logger.info(
        "[KT_STREAM][depool] MXFP4 background reads started (%d layers, %d workers), "
        "overlapping load",
        num_layers,
        nworkers,
    )


def _finish_bg_build_mxfp4(num_layers) -> None:
    """Drain the background MXFP4 reads (no NZ pass).

    Raises if any layer read failed, so the caller can fall back to the serial builder.
    """
    global _MXFP4_POOL_BUILT
    import time

    errs = []
    for _ in range(num_layers):
        item = _BG_MX["done_q"].get()
        if isinstance(item, tuple):
            errs.append(item)
    _BG_MX["ex"].shutdown()
    if errs:
        raise RuntimeError(
            f"MXFP4 parallel read failed on {len(errs)} layer(s): {errs[:3]}"
        )
    _MXFP4_POOL_BUILT = True
    logger.info(
        "[KT_STREAM][depool] MXFP4 pool built in %.0fs (parallel O_DIRECT, %d layers)",
        time.perf_counter() - _BG_MX["t_start"],
        num_layers,
    )


def reserve_slot(E: int, H: int, I: int, dev) -> None:
    """Allocate the reused NZ HBM streaming slot EARLY, during model load.

    KV-pool sizing measures free HBM after loading, so a slot reserved here is automatically
    excluded from the KV pool and cannot contend with it mid-forward.  Idempotent.
    """
    global _SLOT_RESERVED
    if _SLOT_RESERVED:
        return
    import torch_npu

    s13 = torch_npu.npu_format_cast(
        torch.empty(E, H, 2 * I, dtype=torch.int8, device=dev), _ACL_FORMAT_FRACTAL_NZ
    )
    s2 = torch_npu.npu_format_cast(
        torch.empty(E, I, H, dtype=torch.int8, device=dev), _ACL_FORMAT_FRACTAL_NZ
    )
    _SLOT["w13"], _SLOT["w2"] = s13, s2
    _SLOT_RESERVED = True
    logger.info(
        "[KT_STREAM] reserved streaming slot %s+%s (%.2fGB) at model-load time",
        tuple(s13.shape),
        tuple(s2.shape),
        (s13.numel() + s2.numel()) / 1e9,
    )


def reserve_slot_depool(E: int, H: int, I: int, dev) -> None:
    """Reserve the depool convert-output slot as PLAIN ND ``torch.empty`` (NOT ``format_cast``).

    The depool convert fills it via ``out_nz[c:ce].copy_(nz_chunk)``: an ND-tagged destination
    takes a raw byte copy of the NZ bytes (which is what the W8A8 operator then consumes),
    whereas a slice copy into an NZ-FORMATTED destination triggers a full-tensor de-format
    round trip -- a fresh multi-GB allocation, i.e. OOM on the serving headroom.  Shapes match
    ``(E,) + nz.shape[1:]``: w13 [E,H,2I], w2 [E,I,H].  Idempotent.
    """
    global _SLOT_RESERVED
    if _SLOT_RESERVED:
        return
    _SLOT["w13"] = torch.empty(E, H, 2 * I, dtype=torch.int8, device=dev)
    _SLOT["w2"] = torch.empty(E, I, H, dtype=torch.int8, device=dev)
    _SLOT_RESERVED = True
    logger.info(
        "[KT_STREAM][depool] reserved ND streaming slot %s+%s (%.2fGB) at model-load time",
        tuple(_SLOT["w13"].shape),
        tuple(_SLOT["w2"].shape),
        (_SLOT["w13"].numel() + _SLOT["w2"].numel()) / 1e9,
    )


def _ensure_slot(w13_shape, w2_shape, dev):
    import torch_npu

    if "w13" not in _SLOT:
        s13 = torch.empty(w13_shape, dtype=torch.int8, device=dev)
        s2 = torch.empty(w2_shape, dtype=torch.int8, device=dev)
        _SLOT["w13"] = torch_npu.npu_format_cast(s13, _ACL_FORMAT_FRACTAL_NZ)
        _SLOT["w2"] = torch_npu.npu_format_cast(s2, _ACL_FORMAT_FRACTAL_NZ)
    return _SLOT["w13"], _SLOT["w2"]


def _wrapper_dims(wrapper: KTEPWrapperMethod):
    E = int(wrapper.global_num_experts or 0)
    H = int(wrapper.hidden_size or 0)
    I = int(wrapper.intermediate_size_per_partition or 0)
    num_layers = int(wrapper.kt_config.num_layers or 0)
    return E, H, I, num_layers


def _free_slot() -> None:
    """Release the reserved slot so its HBM can serve as build scratch.

    The NZ cast has to round-trip through HBM, and the build runs before any streaming, so the
    slot is idle then; :func:`_ensure_slot` re-allocates it afterwards.  The net HBM the
    feature needs stays at one slot.
    """
    global _SLOT_RESERVED
    _SLOT.clear()
    _SLOT_RESERVED = False
    torch.npu.empty_cache()


def _remap_resident_params_to_cache(layer: torch.nn.Module, wrapper) -> None:
    """Move the resident params/masks off the model's loaded-weight memory region.

    Loaded weights live in a flush/coherence-optimised read-only region; writing it at runtime
    triggers a device coherence flush that stalls the per-layer host syncs.  Cloning makes them
    ordinary caching-allocator tensors, where the per-prefill rewrite is free.  Done at
    model-load time, i.e. BEFORE graph capture, so the graph captures the new storage.
    """
    for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
        p = getattr(layer, name, None)
        if p is not None:
            p.data = p.data.clone()
    for name in ("gpu_experts_mask", "logical_to_gpu_index"):
        t = getattr(wrapper, name, None)
        if t is not None and t.device.type == "npu":
            setattr(wrapper, name, t.clone())


def maybe_reserve_slot(wrapper, dev, layer=None) -> None:
    """Called from ``process_weights_after_loading`` when streaming is enabled.

    Reserves the streaming slot BEFORE KV-pool sizing so the KV pool accounts for it (the pool
    itself is built here too when the checkpoint layout allows, else lazily on the first long
    prefill).  Also registers ``(layer, wrapper)`` for the dynamic decode-resident pool.
    """
    global _MXFP4_POOL_BUILT

    if not _KT_PREFILL_STREAM or wrapper.tp_rank != 0:
        return
    try:
        E, H, I, num_layers = _wrapper_dims(wrapper)
        _remember_dims(E, H, I, num_layers)
        if layer is not None:
            _REGISTRY[wrapper.kt_config.layer_idx] = (layer, wrapper)
            if _KT_DYN_RESIDENT:
                _remap_resident_params_to_cache(layer, wrapper)
        if _KT_MXFP4_DEPOOL and _KT_GGUF_DEDUP:
            # GGUF dedup: there is no codes pool to build (each layer is read from the GGUF on
            # the fly), so mark it built and skip the lazy serial builder.  The convert-output
            # slot is still reserved so the KV pool is sized around it.
            _MXFP4_POOL_BUILT = True
            if not _GGUF_TMPL:
                logger.warning(
                    "[KT_STREAM][dedup] KT_MXFP4_GGUF_DEDUP=1 but KT_GGUF_TEMPLATE is empty"
                )
            if E and H and I:
                reserve_slot_depool(E, H, I, dev)
            return
        if _KT_MXFP4_DEPOOL:
            # Depool builds no W8A8 pool.  Reserve the convert-output slot (same reason as the
            # dedup branch) and build the small MXFP4 pool with parallel O_DIRECT reads started
            # on the first process_weights call, drained on the last layer.
            if E and H and I:
                reserve_slot_depool(E, H, I, dev)
            if E and H and I and num_layers and not _MXFP4_POOL_BUILT:
                if not _BG_MX["started"]:
                    _BG_MX["started"] = True
                    _start_bg_reads_mxfp4(E, H, I, num_layers)
                if wrapper.kt_config.layer_idx == num_layers - 1:
                    _finish_bg_build_mxfp4(num_layers)
            return
        if E and H and I:
            reserve_slot(E, H, I, dev)
        # Overlap the pool build's O_DIRECT reads with the rest of model load: start the
        # background reads on the FIRST process_weights call (host-only, while the remaining
        # weights load), then drain + NZ-cast on the LAST call.
        if E and H and I and num_layers and not _POOL_BUILT:
            if not _BG["started"]:
                _BG["started"] = True
                _start_bg_reads(E, H, I, num_layers)
            if wrapper.kt_config.layer_idx == num_layers - 1:
                _finish_bg_build(num_layers, dev)
                reserve_slot(E, H, I, dev)
    except Exception as e:
        logger.warning(
            "[KT_STREAM] reserve/build at load failed (%s); lazy fallback",
            repr(e)[:160],
        )


_INIT_ROUTING = None
_FINALIZE_ROUTING = None


def _routing_ops():
    """Build (and cache) the init/finalize routing helpers used by the streaming MoE."""
    global _INIT_ROUTING, _FINALIZE_ROUTING
    if _INIT_ROUTING is None:
        from sglang.srt.hardware_backend.npu.moe.finalize_routing import (
            NPUFinalizeRouting,
        )
        from sglang.srt.hardware_backend.npu.moe.init_routing import (
            NPUMoEInitRouting_v2,
        )

        # quant_mode=1 -> the routing op emits int8 activations plus the per-token scale that
        # gmm1 dequantises with, matching the W8A8 expert weights this module streams.
        _INIT_ROUTING = NPUMoEInitRouting_v2(quant_mode=1)
        # drop_pad_mode=2 -> no capacity dropping, same as AscendTPDispatcher.
        _FINALIZE_ROUTING = NPUFinalizeRouting(drop_pad_mode=2)
    return _INIT_ROUTING, _FINALIZE_ROUTING


@functools.lru_cache(maxsize=2)
def _log_stream_swiglu_once(limit: float) -> None:
    print(f"[KT_STREAM][swiglu] streaming-prefill clamp "
          f"{'ACTIVE limit=%.4g' % limit if limit > 0 else 'OFF'}", flush=True)


@functools.lru_cache(maxsize=1)
def _swiglu_limit() -> float:
    """The checkpoint's swiglu_limit, read the same way the module reads E/H/I."""
    try:
        cfg = json.load(open(os.path.join(_ckpt_dir(), "config.json")))
        return float(cfg.get("swiglu_limit") or 0.0)
    except Exception:
        return 0.0


def _apply_swiglu_limit_streaming(x: torch.Tensor) -> None:
    """In-place asymmetric clamp on the gate/up halves. Shares the runner's implementation so
    the streamed experts and the resident ones cannot drift apart."""
    from sglang.srt.hardware_backend.npu.moe.activation import apply_swiglu_limit_

    limit = _swiglu_limit()
    _log_stream_swiglu_once(limit)
    apply_swiglu_limit_(x, limit)


def _streaming_fused_experts(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    num_experts: int,
) -> torch.Tensor:
    """Run one W8A8 MoE layer over the streamed expert set, end to end.

    This is the whole ``dispatch -> gmm1 -> swiglu -> gmm2 -> combine`` chain that the regular
    path spreads over ``AscendTPDispatcher`` + ``AscendRunnerCore`` +
    ``NPUW8A8Int8MoEMethod``, inlined here because the streamed weights are plain tensors, not
    layer parameters, and because the expert count is the GLOBAL one rather than the resident
    subset the layer's runner is configured for.

    Argument layout (identical to what ``process_weights_after_loading`` leaves on the layer):
      w13  FRACTAL_NZ int8 [E, H, 2I]   w13_scale bf16 [E, 2I]
      w2   FRACTAL_NZ int8 [E, I, H]    w2_scale  bf16 [E, H]

    group_list_type PAIRING -- read before changing the routing version.  ``group_list`` is
    produced by init routing and consumed by both grouped matmuls, and the two ends must agree
    on its encoding:
      * v2 routing (``npu_moe_init_routing_v2``, ``expert_tokens_num_type=1``) yields per-expert
        COUNTS   -> ``group_list_type=1``   <- what this function uses
      * v1 routing (``npu_moe_compute_expert_tokens``) yields the CUMULATIVE prefix sums
        -> ``group_list_type=0``
    Pairing v2 counts with ``group_list_type=0`` (or vice versa) is accepted by the operator and
    silently produces wrong numbers -- there is no shape or dtype error to catch it.
    """
    init_routing, finalize_routing = _routing_ops()
    topk_weights = topk_weights.to(hidden_states.dtype)
    topk_ids = topk_ids.to(torch.int32)

    # 1. dispatch: permute tokens into expert order and quantise them to int8.
    permuted, expanded_row_idx, expert_tokens, pertoken_scale = (
        init_routing._init_routing(hidden_states, topk_ids, num_experts, top_k)
    )
    # Derived from the PERMUTED tensor, not from hidden_states, so that this matches
    # AscendRunnerCore.run byte for byte: routing has already quantised to int8 there too, so
    # the expression always selects bfloat16.
    output_dtype = torch.float16 if permuted.dtype == torch.float16 else torch.bfloat16

    # 2. gmm1 (gate & up).  The weights are already stored transposed + NZ, so they are passed
    #    through as-is, exactly like GroupedMatmul(transposed=True).
    permuted = torch.ops.npu.npu_grouped_matmul(
        x=[permuted],
        weight=[w13],
        scale=[w13_scale],
        per_token_scale=[pertoken_scale],
        split_item=2,
        group_list_type=_GROUP_LIST_TYPE,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=output_dtype,
    )[0]

    # 3. activation: swiglu plus the re-quantisation gmm2 needs (NPUSwigluQuant).
    #    The model's own SwiGLU clamp has to be applied here too, or the streamed experts
    #    would differ numerically from both the CPU MoE and DeepSeek's reference.
    _apply_swiglu_limit_streaming(permuted)
    permuted, swiglu_scale = torch.ops.npu.npu_dequant_swiglu_quant(
        permuted, quant_mode=1, activate_left=True
    )

    # 4. gmm2 (down).
    permuted = torch.ops.npu.npu_grouped_matmul(
        x=[permuted],
        weight=[w2],
        scale=[w2_scale],
        per_token_scale=[swiglu_scale],
        split_item=2,
        group_list_type=_GROUP_LIST_TYPE,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=output_dtype,
    )[0]

    # 5. combine: weighted un-permute back to token order.
    return finalize_routing._finalize_routing(
        permuted,
        topk_weights=topk_weights,
        expanded_row_idx=expanded_row_idx,
        topk_ids=topk_ids,
    )


def _stream_layer_weights(layer_idx: int, dev):
    """Materialise this layer's full expert weights in the reused HBM slot.

    Returns ``(w13, w13_scale, w2, w2_scale)``.
    """
    if not _KT_MXFP4_DEPOOL:
        h13, h2, s13b, s2b = _POOL[layer_idx]
        slot13, slot2 = _ensure_slot(h13.shape, h2.shape, dev)
        slot13.copy_(
            h13
        )  # H2D this layer's experts (default stream, serial single slot)
        slot2.copy_(h2)
        return slot13, s13b, slot2, s2b

    E, H, I, num_layers = _get_cfg()
    # Reserved streaming slot (maybe_reserve_slot): the convert writes the full expert set
    # straight into it, reused across layers, so no per-layer multi-GB output allocation
    # competes with the KV pool.  None (reserve skipped or failed) -> convert allocates fresh.
    slot13, slot2 = _SLOT.get("w13"), _SLOT.get("w2")
    if not _KT_GGUF_DEDUP:
        # H2D this layer's MXFP4 (4-bit, about half the W8A8 bytes), then convert to W8A8-NZ.
        c13, s13, c2, s2 = _MXFP4_POOL[layer_idx]
        w13, s13b, w2, s2b = _mxfp4_convert_fn()(
            c13.to(dev, non_blocking=True),
            s13.to(dev, non_blocking=True),
            c2.to(dev, non_blocking=True),
            s2.to(dev, non_blocking=True),
            H,
            I,
            out_w13=slot13,
            out_w2=slot2,
        )
        return w13, s13b, w2, s2b

    # GGUF dedup: read this layer's MXFP4 straight from the CPU MoE's GGUF (block_mxfp4).  The
    # prefetch worker copies the next layer's raw blocks into a pinned ping-pong buffer while
    # this layer converts.  The raw blocks go to the device and the AscendC kernel
    # de-interleaves (scale|codes) in UB via Gather (KT_MXFP4_BLK_KERNEL, default); the software
    # 16-of-17 strided de-interleave it replaces was the prefill bottleneck.
    if _KT_PREFETCH:
        par = _prefetch_ensure(layer_idx, num_layers)
    else:
        _fill_stage(layer_idx)
        par = layer_idx % 2
    blk13 = _MX_PP["w13"][par].to(dev, non_blocking=True)
    blk2 = _MX_PP["w2"][par].to(dev, non_blocking=True)
    if _KT_BLK_KERNEL:
        return _mxfp4_convert_blk_fn()(blk13, blk2, H, I, out_w13=slot13, out_w2=slot2)

    def _di(d):
        E_, OUT_, n17 = d.shape
        nbq = n17 // 17
        b = d.view(E_, OUT_, nbq, 17)
        return (
            b[..., 1:17].reshape(E_, OUT_, nbq * 16).contiguous(),
            b[..., 0].contiguous(),
        )

    c13d, s13d = _di(blk13)
    c2d, s2d = _di(blk2)
    return _mxfp4_convert_fn()(
        c13d, s13d, c2d, s2d, H, I, packing="halfblock", out_w13=slot13, out_w2=slot2
    )


def _streaming_forward(layer_idx, x, topk_output, top_k, num_experts) -> torch.Tensor:
    w13, s13b, w2, s2b = _stream_layer_weights(layer_idx, x.device)
    out = _streaming_fused_experts(
        hidden_states=x,
        w13=w13,
        w13_scale=s13b,
        w2=w2,
        w2_scale=s2b,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        top_k=top_k,
        num_experts=num_experts,
    )

    # Dynamic resident: the W8A8 path gathers from the resident W8A8 pool at the end of the
    # prefill; the depool path gathers the hot-K experts out of the weights it just converted.
    if _KT_DYN_RESIDENT:
        E, _, _, num_layers = _get_cfg()
        if _KT_MXFP4_DEPOOL:
            try:
                _apply_resident_layer_depool(layer_idx, topk_output, w13, s13b, w2, s2b)
            except Exception as e:
                logger.warning(
                    "[KT_STREAM] inline resident L%d failed (%s); static set kept",
                    layer_idx,
                    repr(e)[:140],
                )
        else:
            if layer_idx == 0:
                _REQ_HIST.clear()
            _REQ_HIST[layer_idx] = torch.bincount(
                _hist_ids(topk_output.topk_ids).to(torch.int64), minlength=E
            )[:E]
            if layer_idx == num_layers - 1 and len(_REQ_HIST) == num_layers:
                try:
                    _apply_dynamic_residency()
                except Exception as e:
                    logger.warning(
                        "[KT_STREAM] dynamic residency failed (%s); static set kept",
                        repr(e)[:160],
                    )
    return out


def _hist_ids(topk_ids):
    """Flattened routed expert ids used to build the resident-set histogram.

    When ``KT_HOT_TAIL_TOKENS > 0``, restrict to the last N prompt tokens (recency); else use
    the whole prefill.  ``topk_ids`` is [M_tokens, top_k], so slicing rows keeps the last N
    tokens' routing.
    """
    if _HOT_TAIL > 0 and topk_ids.dim() == 2 and topk_ids.shape[0] > _HOT_TAIL:
        topk_ids = topk_ids[-_HOT_TAIL:]
    return topk_ids.reshape(-1)


def _pick_resident_top(counts, K):
    """Pick the K resident experts for a layer: top-K by activation, ascending int64.

    Shared by the inline (depool) and post-pass (W8A8) paths so both agree on the selection.
    """
    return counts.topk(K).indices.sort().values


def _set_resident_masks(wrap, top_cpu, K, E):
    """Rewrite the routing structures so the resident set is exactly ``top_cpu``.

    In place, therefore safe for both decode graph replay and the C++ side.
    """
    new_mask = torch.zeros(E, dtype=torch.bool)
    new_mask[top_cpu] = True
    l2g = torch.full((E,), -1, dtype=torch.int64)
    l2g[top_cpu] = torch.arange(K, dtype=torch.int64)
    wrap.gpu_experts_mask.copy_(new_mask.to(wrap.gpu_experts_mask.device))
    wrap.logical_to_gpu_index.copy_(
        l2g.to(
            device=wrap.logical_to_gpu_index.device,
            dtype=wrap.logical_to_gpu_index.dtype,
        )
    )
    if wrap.wrapper is not None:  # pinned CPU mask, the C++ side reads it live
        wrap.wrapper.gpu_experts_mask.copy_(new_mask)


_RES_PEND = {}  # L -> (wrap, top_device, counts_device): deferred mask updates


def _apply_resident_layer_depool(L, topk_output, w13, s13b, w2, s2b):
    """Depool: populate the decode resident slots from this layer's converted expert set.

    ``index_select`` gathers the hot-K experts of the already-converted weights straight into
    the resident params (zero-alloc, and a first-dim gather is format-safe on NZ).  The mask
    updates are deferred to the last layer so no per-layer host sync is needed.  This folds the
    decode hot-expert update into the streaming prefill at roughly zero cost.

    Prerequisite (see :func:`_remap_resident_params_to_cache`): the resident params must have
    been remapped to caching-allocator memory at model load, otherwise writing them stalls the
    per-layer host syncs.
    """
    layer, wrap = _REGISTRY[L]
    K = int(wrap.num_gpu_experts)
    if K <= 0 or wrap.gpu_experts_mask is None or wrap.logical_to_gpu_index is None:
        return
    E, H, I, num_layers = _get_cfg()
    if L == 0:
        _RES_PEND.clear()
    counts = torch.bincount(
        _hist_ids(topk_output.topk_ids).to(torch.int64), minlength=E
    )[:E]
    top = _pick_resident_top(counts, K)
    # index_select(..., out=<param>.data) 在目标带 ACL 私有格式(FRACTAL_NZ)时是**静默 no-op**:
    # torch_npu 无法往 NZ 的 out 里写, 于是把 .data 返回的临时 Tensor 重绑定到一个新分配的
    # ND tensor 并丢弃, nn.Parameter 自己的 storage 一个字节都没动 —— 无异常无告警。
    # 紧邻的 scale 是 ND, 所以 scale 写进去了; mask/l2g 也写进去了。结果是槽位 i 用专家 i 的
    # 权重配专家 top[i] 的 scale, 而 CPU 又因 mask 认为 top[i] 已常驻而跳过它。
    # Tensor.copy_ 是格式感知的, 原地写 Parameter 自己的 storage(decode NPU graph 捕获的正是它)。
    layer.w13_weight.data.copy_(torch.index_select(w13, 0, top))
    layer.w2_weight.data.copy_(torch.index_select(w2, 0, top))
    layer.w13_weight_scale.data.copy_(torch.index_select(s13b, 0, top))
    layer.w2_weight_scale.data.copy_(torch.index_select(s2b, 0, top))
    _RES_PEND[L] = (wrap, top, counts)
    if L == num_layers - 1:
        share_sum = 0.0
        for _LL, (wr, tp, cnt) in sorted(_RES_PEND.items()):
            _set_resident_masks(wr, tp.cpu(), K, E)
            share_sum += float(cnt[tp].sum().item()) / max(float(cnt.sum().item()), 1.0)
        n = len(_RES_PEND)
        _RES_PEND.clear()
        logger.info(
            "[KT_STREAM] inline resident: top-%d x %d layers folded into prefill, share=%.3f",
            K,
            n,
            share_sum / max(n, 1),
        )


def _apply_dynamic_residency() -> None:
    """Replace the static-prefix resident expert set with this prefill's per-layer top-K.

    Updates the resident weights, scales and routing structures in place (decode-graph and
    C++-side safe).  Called at the end of a streaming prefill pass.
    """
    import time

    E, H, I, num_layers = _get_cfg()
    for L in range(num_layers):
        _pool_ok = (L in _MXFP4_POOL) if _KT_MXFP4_DEPOOL else (L in _POOL)
        if L not in _REQ_HIST or L not in _REGISTRY or not _pool_ok:
            logger.warning("[KT_STREAM] dyn-resident: layer %d incomplete; abort", L)
            return
    t0 = time.perf_counter()
    share_sum = 0.0
    K = 0
    for L in range(num_layers):
        layer, wrap = _REGISTRY[L]
        K = int(wrap.num_gpu_experts)
        if K <= 0 or wrap.gpu_experts_mask is None or wrap.logical_to_gpu_index is None:
            logger.warning("[KT_STREAM] dyn-resident: no resident slots/masks; abort")
            return
        counts = _REQ_HIST[L]
        top = _pick_resident_top(counts, K)  # device, ascending logical ids
        top_cpu = top.cpu()
        if _KT_MXFP4_DEPOOL:
            # Convert ONLY the hot-K experts' MXFP4 into the resident slots.  MXFP4 codes are
            # plain packed bytes (not NZ), so a first-dim [top] slice is format-safe, and the
            # fused kernel emits resident-shaped NZ + bf16 scale straight into place -- no
            # whole-pool H2D and no NZ round-trip gather.
            c13, s13, c2, s2 = _MXFP4_POOL[L]
            dev = layer.w13_weight.device
            c13d = _stage_pin_h2d(c13, top_cpu, dev)  # pinned staging -> DMA H2D
            s13d = _stage_pin_h2d(s13, top_cpu, dev)
            c2d = _stage_pin_h2d(c2, top_cpu, dev)
            s2d = _stage_pin_h2d(s2, top_cpu, dev)
            w13_top, s13b_top, w2_top, s2b_top = _mxfp4_convert_fn()(
                c13d, s13d, c2d, s2d, H, I
            )
            layer.w13_weight.data.copy_(w13_top)
            layer.w2_weight.data.copy_(w2_top)
            layer.w13_weight_scale.data.copy_(s13b_top)
            layer.w2_weight_scale.data.copy_(s2b_top)
        else:
            # Gather the resident experts ON THE DEVICE: host pool slices are format-unaware
            # (NZ bytes sliced as ND give garbage), device slices are format-aware.  So H2D the
            # whole pool into the NZ slot and slice there.
            h13, h2, s13b, s2b = _POOL[L]
            dev = layer.w13_weight.device
            slot13, slot2 = _ensure_slot(h13.shape, h2.shape, dev)  # [E,...] NPU NZ
            slot13.copy_(h13)  # whole-tensor H2D (correct NZ)
            slot2.copy_(h2)
            # Gather via an ND round trip: a per-slot NZ device copy is bandwidth-pathological,
            # whereas format_cast NZ->ND runs at full HBM bandwidth, the ND fancy-index is
            # cheap, and ND->NZ restores the format.  Equivalent, much faster.
            import torch_npu as _tn

            _topd = top.to(dev)
            _nd13 = _tn.npu_format_cast(slot13, 2)  # whole pool NZ->ND
            _g13 = _tn.npu_format_cast(
                _nd13[_topd].contiguous(), _ACL_FORMAT_FRACTAL_NZ
            )
            del _nd13
            layer.w13_weight.data.copy_(_g13)
            del _g13
            _nd2 = _tn.npu_format_cast(slot2, 2)
            _g2 = _tn.npu_format_cast(_nd2[_topd].contiguous(), _ACL_FORMAT_FRACTAL_NZ)
            del _nd2
            layer.w2_weight.data.copy_(_g2)
            del _g2
            layer.w13_weight_scale.data.copy_(s13b[top])
            layer.w2_weight_scale.data.copy_(s2b[top])
        _set_resident_masks(wrap, top_cpu, K, E)
        share_sum += float(counts[top].sum().item()) / max(
            float(counts.sum().item()), 1.0
        )
    torch.npu.synchronize()
    logger.info(
        "[KT_STREAM] dynamic resident applied: top-%d x %d layers in %.1fs, "
        "prefill top-K activation share=%.3f",
        K,
        num_layers,
        time.perf_counter() - t0,
        share_sum / num_layers,
    )


# Run the first N real prefills through the HYBRID path (not streamed) to prime the CPU MoE.
# Streamed prefills never invoke kt_kernel, so a stream-everything server (low threshold) keeps the
# CPU MoE cold and decode stays slow until enough hybrid traffic warms it.  The OS page cache is not
# the issue (the GGUF is already cached); the warming is process-local (kt_kernel threadpool and
# buffers, first-touch PTEs).  A few hybrid prefills fix it.
_KT_STREAM_WARMUP = int(os.environ.get("KT_STREAM_WARMUP", "0") or "0")
_STREAM_WARMUP_STATE: dict = {}


def _warmup_consumes(layer_idx: int, num_tokens: int) -> bool:
    """Return True while the startup warmup budget forces this prefill down the hybrid path.

    MUST be checked BEFORE the token threshold: at a high threshold every sub-threshold prefill
    (including sglang's own startup warmup) would return at the gate without counting, so the
    budget would instead land on the first long user prefill and wrongly force it hybrid.
    """
    if _KT_STREAM_WARMUP <= 0 or num_tokens <= 1:
        return False
    st = _STREAM_WARMUP_STATE
    if layer_idx == 0:
        st["seen"] = st.get("seen", 0) + 1
    if st.get("seen", 0) > _KT_STREAM_WARMUP:
        return False
    if layer_idx == 0:
        logger.info(
            "[KT_STREAM] warmup prefill %d/%d -> hybrid (prime the CPU MoE)",
            st["seen"],
            _KT_STREAM_WARMUP,
        )
    return True


def maybe_streaming_forward(
    quant_method,
    hidden_states: torch.Tensor,
    topk_output,
    tp_reduce_needed: bool = False,
) -> Optional[torch.Tensor]:
    """Entry from ``FusedMoE.forward_impl``, before the dispatcher runs.

    Returns the layer's final hidden states when streaming handled this layer, else ``None``
    (the caller then falls through to the normal dispatch/compute/combine path).  Never raises.

    Returning the finished tensor rather than a ``CombineInput`` is deliberate: the streaming
    path replaces dispatch, expert compute AND combine in one go, so there is nothing left for
    the caller's combine step to do.  ``tp_reduce_needed`` therefore disables streaming, since
    the skipped tail also contains the TP all-reduce.
    """
    if not _KT_PREFILL_STREAM or tp_reduce_needed:
        return None
    if not isinstance(quant_method, KTEPWrapperMethod) or quant_method.tp_rank != 0:
        return None
    if not _is_prefill():
        return None
    if _warmup_consumes(quant_method.kt_config.layer_idx, hidden_states.shape[0]):
        return None
    if hidden_states.shape[0] < _T:
        return None
    try:
        layer_idx = quant_method.kt_config.layer_idx
        E, H, I, num_layers = _wrapper_dims(quant_method)
        if not (E and H and I and num_layers):
            logger.warning(
                "[KT_STREAM] missing dims (E=%s H=%s I=%s L=%s) -> hybrid",
                E,
                H,
                I,
                num_layers,
            )
            return None
        _remember_dims(E, H, I, num_layers)
        if _KT_MXFP4_DEPOOL:
            if _KT_GGUF_DEDUP:
                pass  # no pool: each layer is read from the CPU MoE's GGUF on the fly
            elif not _MXFP4_POOL_BUILT:
                # Normally built at model-load time (maybe_reserve_slot, parallel O_DIRECT);
                # this serial builder only runs if that path failed or never started.
                _build_mxfp4_pool(E, H, I, num_layers)
        elif not _POOL_BUILT:
            # Lazy fallback: parallel O_DIRECT build.  _build_pool_parread frees the slot
            # internally for NZ scratch; _stream_layer_weights re-allocates it afterwards.
            _build_pool_parread(E, H, I, num_layers, hidden_states.device)
        top_k = topk_output.topk_ids.shape[1]
        return _streaming_forward(layer_idx, hidden_states, topk_output, top_k, E)
    except (
        Exception
    ) as e:  # any failure -> fall back to hybrid, never crash the forward
        logger.warning(
            "[KT_STREAM] streaming failed (%s) -> hybrid fallback", repr(e)[:160]
        )
        return None
