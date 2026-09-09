"""
Copyright (c) 2025 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file additionally carries the SGLang/boogu optimizations:
  * native GQA/MQA support (no K/V expansion, both in delta_s and in kernel)
  * non-standard head dims (e.g. 120) quantized straight into the padded
    64/128 kernel layout without a host-side F.pad pass
  * device-adaptive persistent-grid sizing via ``num_sms``
"""
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from typing import Tuple
from torch.nn.functional import scaled_dot_product_attention as sdpa
import fp4attn_cuda
import fp4quant_cuda

# --- SGLang capability ABI -------------------------------------------------
# The SGLang SageAttention3 backend probes these attributes to decide whether
# the native-GQA fast path and the dynamic persistent-grid scheduler exist.
SGLANG_NATIVE_GQA = True
SGLANG_DYNAMIC_SM_SCHEDULER = True


def supports_current_device(device=None) -> bool:
    """Whether the FP4 Blackwell kernel supports the given (or current) GPU."""
    if not torch.cuda.is_available():
        return False
    try:
        if device is None:
            major, minor = torch.cuda.get_device_capability()
        else:
            idx = device.index if isinstance(device, torch.device) else int(device)
            major, minor = torch.cuda.get_device_capability(idx)
    except Exception:
        return False
    return (major, minor) in {(10, 0), (12, 0), (12, 1)}


@triton.jit
def group_mean_kernel(
    q_ptr,
    q_out_ptr,
    qm_out_ptr,
    B, H, L, D: tl.constexpr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_qmb, stride_qmh, stride_qml, stride_qmd,
    GROUP_SIZE: tl.constexpr,
    D_POW2: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_group = tl.program_id(2)
    
    group_start = pid_group * GROUP_SIZE
    offsets = group_start + tl.arange(0, GROUP_SIZE)
    # Masked feature range supports non-power-of-2 head dims (e.g. 120).
    d_range = tl.arange(0, D_POW2)
    d_mask = d_range < D
    
    q_offsets = pid_b * stride_qb + pid_h * stride_qh + offsets[:, None] * stride_ql + d_range[None, :] * stride_qd
    q_group = tl.load(q_ptr + q_offsets, mask=d_mask[None, :], other=0.0)
    
    qm_group = tl.sum(q_group, axis=0) / GROUP_SIZE
    
    q_group = q_group - qm_group
    tl.store(q_out_ptr + q_offsets, q_group, mask=d_mask[None, :])

    qm_offset = pid_b * stride_qmb + pid_h * stride_qmh + pid_group * stride_qml + d_range * stride_qmd
    tl.store(qm_out_ptr + qm_offset, qm_group, mask=d_mask)


def triton_group_mean(q: torch.Tensor):
    B, H, L, D = q.shape
    GROUP_SIZE = 128
    num_groups = L // GROUP_SIZE
    d_pow2 = max(16, triton.next_power_of_2(D))
    
    q_out = torch.empty_like(q)  # [B, H, L, D]
    qm = torch.empty(B, H, num_groups, D, device=q.device, dtype=q.dtype)
    
    grid = (B, H, num_groups)
    
    group_mean_kernel[grid](
        q, q_out, qm,
        B, H, L, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        qm.stride(0), qm.stride(1), qm.stride(2), qm.stride(3),
        GROUP_SIZE=GROUP_SIZE,
        D_POW2=d_pow2
    )
    return q_out, qm


def preprocess_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, per_block_mean: bool = True, pad_seq: bool = None):

    def pad_128(x):
        L = x.size(2)
        pad_len = (128 - L % 128) % 128
        if pad_len == 0:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()

    # Sequence padding is only needed by the per-block-mean path (the triton
    # group-mean kernel requires L % 128 == 0). The attention kernel itself
    # handles ragged sequence tails via the unpadded_k interface, and the
    # quantization kernels already predicate their token loop, so the
    # per_block_mean=False path skips the three full-tensor F.pad passes.
    if pad_seq is None:
        pad_seq = per_block_mean

    # Do not mutate the caller's K tensor.  The original in-place subtraction
    # also prevents safely sharing K across grouped-query heads.
    k = k - k.mean(dim=-2, keepdim=True)
    if pad_seq:
        q, k, v = map(lambda x: pad_128(x), [q, k, v])
    if per_block_mean:
        q, qm = triton_group_mean(q)
    else:
        qm = q.mean(dim=-2, keepdim=True)
        q = q - qm
    num_q_heads = q.size(1)
    num_kv_heads = k.size(1)
    if num_q_heads % num_kv_heads:
        raise ValueError("the number of Q heads must be divisible by KV heads")
    if num_q_heads == num_kv_heads:
        delta_s = torch.matmul(qm, k.transpose(-2, -1))
    else:
        # Compute the mean correction with native GQA indexing instead of
        # materializing K num_q_heads / num_kv_heads times.
        q_per_kv = num_q_heads // num_kv_heads
        grouped_qm = qm.reshape(
            qm.size(0), num_kv_heads, q_per_kv, qm.size(-2), qm.size(-1)
        )
        grouped_kt = k.transpose(-2, -1).unsqueeze(2)
        delta_s = torch.matmul(grouped_qm, grouped_kt).reshape(
            qm.size(0), num_q_heads, qm.size(-2), k.size(-2)
        )
    delta_s = delta_s.to(torch.float32).contiguous()
    if not pad_seq:
        # The kernel-side blockscaled DS layout is tiled from the padded K
        # length; keep delta_s column-aligned with it (tiny tensor, so the
        # column pad is negligible compared to the removed full-tensor pads).
        lk = k.size(2)
        lk_pad = (lk + 127) // 128 * 128
        if lk_pad != lk:
            delta_s = F.pad(delta_s, (0, lk_pad - lk))
    return q, k, v, delta_s


def _pad_head_dim(head_dim: int) -> int:
    """Padded kernel head-dim layout for a possibly non-standard head dim."""
    if head_dim <= 64:
        return 64
    if head_dim <= 128:
        return 128
    raise ValueError(f"Unsupported head dimension: {head_dim}")


def scale_and_quant_fp4(x: torch.Tensor, d_pad: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    if d_pad is None:
        d_pad = D
    # Row count is padded to the 128 kernel block: the attention kernel's
    # blockscaled SF layout tiles by 64 rows and requires an integral number
    # of tiles, and the kernel's block scheduler walks whole 128-row tiles.
    # Tail rows are written as zeros by the kernel's guarded load and are
    # sliced away from the attention output afterwards.
    n_pad = (N + 127) // 128 * 128
    packed_fp4 = torch.empty((B, H, n_pad, d_pad // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, n_pad, d_pad // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale


def scale_and_quant_fp4_permute(x: torch.Tensor, d_pad: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    if d_pad is None:
        d_pad = D
    # Output token dim is padded to the 128 kernel block; tail rows are
    # written as zeros by the kernel's guarded load.
    n_pad = (N + 127) // 128 * 128
    packed_fp4 = torch.empty((B, H, n_pad, d_pad // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, n_pad, d_pad // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant_permute(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale


def scale_and_quant_fp4_transpose(x: torch.Tensor, d_pad: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    if d_pad is None:
        d_pad = D
    n_pad = (N + 127) // 128 * 128
    packed_fp4 = torch.empty((B, H, d_pad, n_pad // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, d_pad, n_pad // 16), device=x.device, dtype=torch.float8_e4m3fn)
    fp4quant_cuda.scaled_fp4_quant_trans(x, packed_fp4, fp8_scale, 1)
    return packed_fp4, fp8_scale


def blockscaled_fp4_attn(qlist: Tuple, 
                         klist: Tuple,
                         vlist: Tuple,
                         delta_s: torch.Tensor,
                         KL: int,
                         is_causal: bool = False, 
                         per_block_mean: bool = True,
                         is_bf16: bool = True,
                         softmax_scale: float = None,
                         num_sms: int = 0,
                        ):
    if softmax_scale is None:
        softmax_scale = (qlist[0].shape[-1] * 2) ** (-0.5)
    return fp4attn_cuda.fwd(qlist[0], klist[0], vlist[0], qlist[1], klist[1], vlist[1], delta_s, KL, None, softmax_scale, is_causal, per_block_mean, is_bf16, num_sms)


def sageattn3_blackwell(q, k, v, attn_mask = None, is_causal = False, per_block_mean = True, **kwargs):
    if q.size(-1) > 128:
        # The compiled kernel instantiation set covers head dims <= 128 only.
        print(f"Unsupported Headdim {q.size(-1)}")
        return sdpa(q, k, v, is_causal = is_causal, enable_gqa=q.size(1) != k.size(1))
    QL = q.size(2)
    KL = k.size(2)
    head_dim = q.size(-1)
    if k.size(-1) != head_dim or v.size(-1) != head_dim:
        raise ValueError("Q, K, and V must have the same head dimension")
    num_q_heads = q.size(1)
    num_kv_heads = k.size(1)
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            "GQA/MQA requires query heads to be a multiple of KV heads, "
            f"got q_heads={num_q_heads} and kv_heads={num_kv_heads}"
        )
    # Non-standard head dims (e.g. 120) are NOT padded on the host anymore:
    # the quantization kernels read the unpadded input with predication and
    # write straight into the padded (64/128/256) kernel layout.
    d_pad = _pad_head_dim(head_dim)
    is_bf16 = q.dtype == torch.bfloat16
    q, k, v, delta_s = preprocess_qkv(q, k, v, per_block_mean)
    qlist_from_cuda = scale_and_quant_fp4(q, d_pad)
    klist_from_cuda = scale_and_quant_fp4_permute(k, d_pad)
    vlist_from_cuda = scale_and_quant_fp4_transpose(v, d_pad)
    o_fp4 = blockscaled_fp4_attn(
        qlist_from_cuda,
        klist_from_cuda,
        vlist_from_cuda,
        delta_s,
        KL,
        is_causal,
        per_block_mean,
        is_bf16,
        kwargs.get("sm_scale", head_dim ** (-0.5)),
        # num_sms=0 lets the C++ side size the persistent grid to the runtime
        # device instead of the upstream hard-coded 170 (GB200).
        kwargs.get("num_sms", 0),
    )[0][:, :, :QL, :head_dim].contiguous()
    return o_fp4
