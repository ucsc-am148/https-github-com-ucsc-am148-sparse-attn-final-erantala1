"""STUDENT FILE: implement the three block-sparse rung functions.

Implement these three functions from the spec in ALGORITHMS.md -- no reference
code is shipped:

  dsd_matmul             (A1) block-sparse (BCSR) A @ dense B -> dense C
  sparse_flash_forward   (A2) block-sparse flash attention forward
  sparse_flash_backward  (A3) block-sparse flash attention backward

Your functions must match the signatures below: the SHAPES and DTYPES of the
inputs and outputs (each docstring states them; ALGORITHMS.md sec 0.1 collects
them). EVERYTHING ELSE IS YOURS -- how many @triton.jit kernels you write, the
grid, the (B, H) flatten, strides, output allocation, and the launch/tuning. The
grader asserts the returned shapes and dtypes, then checks correctness against an
fp64 reference.

ALGORITHMS.md is the complete spec: the BCSR layout and its two transpose views,
what each output equals, and the five backward equations.

When `python sanity_check.py` passes all three rungs, you're done.
"""
import torch
import triton
import triton.language as tl

# autograder is not running code

@triton.jit
def dsd_matmul_kernel(values_ptr, row_offsets_ptr, column_indices_ptr, B_ptr, C_ptr, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, M: tl.constexpr, N: tl.constexpr):

    # launch is (BLOCK_M, cdiv(N, BLOCK_N))
    pid_M = tl.program_id(0)
    pid_N = tl.program_id(1)

    # get offsets
    # block of C/output tile is (BLOCK_M, BLOCK_N)
    offset_M = pid_M * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_N = pid_N * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_N = (offset_N < N)[None, :]
    block_rows = tl.arange(0, BLOCK_M)[:, None]

    # active blocks
    lo = tl.load(row_offsets_ptr + pid_M)
    hi = tl.load(row_offsets_ptr + pid_M + 1)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
    # only loop over live row blocks
    for idx in range(lo, hi):
        k_blk = tl.load(column_indices_ptr + idx)
        values = values_ptr + idx * BLOCK_M * BLOCK_M
        b_row = k_blk * BLOCK_M

        # inner loop over tile in chunks of BLOCK_K
        for kk in range(0, BLOCK_M, BLOCK_K):
            
            offset_k = kk + tl.arange(0, BLOCK_K)
            offset_a = block_rows * BLOCK_M + offset_k[None, :]
            offset_b = (b_row + offset_k)[:, None] * N + offset_N[None, :]
            a = tl.load(values + offset_a)
            b = tl.load(B_ptr + offset_b, mask=mask_N, other=0.0)
            accumulator += tl.dot(a, b, allow_tf32=False)

    c_ptrs = C_ptr + offset_M[:, None] * N + offset_N[None, :]
    tl.store(c_ptrs, accumulator, mask=mask_N)


def dsd_matmul(values, row_offsets, column_indices, B, M, K, N, block):
    """A1 -- block-sparse C = A @ B. See ALGORITHMS.md sec 1-2.

    Inputs:
      values         (nnz, block, block)  fp32   A's live blocks, row-major
      row_offsets    (M//block + 1,)      int32  per block-row prefix sum of nnz
      column_indices (nnz,)               int32  K-block of each live block
      B              (K, N)               fp32   dense right operand
      M, K, N, block                      ints   dims and block size
    Returns:
      C              (M, N)               fp32

    fp32 throughout, allow_tf32=False.

    TODO: implement.
    """
    C = torch.empty((M, N), device=B.device, dtype=torch.float32)
    BLOCK_M, BLOCK_N, BLOCK_K = block, 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    dsd_matmul_kernel[grid](values, row_offsets, column_indices, B, C, BLOCK_M, BLOCK_N, BLOCK_K, M, N)
    return C


@triton.jit
def sparse_flash_forward_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, q_row_offsets_ptr, q_col_indices_ptr, sm_scale,
    T: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,):
    # one program handles one query block for one flattened batch/head
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # query block offset
    offset_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offset_d = tl.arange(0, BLOCK_D)
    offset_k = tl.arange(0, BLOCK_K)
    q_mask = offset_q < T
    d_mask = offset_d < HEAD_DIM
    base_qkv = pid_bh * T * HEAD_DIM
    base_l = pid_bh * T

    # load Q block (BLOCK_Q, BLOCK_D)
    q = tl.load(Q_ptr + base_qkv + offset_q[:, None] * HEAD_DIM + offset_d[None, :], mask=q_mask[:, None] & d_mask[None, :], other=0.0,)
    # m is running row max, l is running denominator
    m = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    # live key blocks are q_col_indices[lo : hi]
    lo = tl.load(q_row_offsets_ptr + pid_q)
    hi = tl.load(q_row_offsets_ptr + pid_q + 1)

    # loop only over live key blocks for this query block
    for idx in range(lo, hi):
        k_blk = tl.load(q_col_indices_ptr + idx)
        k_start = k_blk * BLOCK_K
        k_rows = k_start + offset_k
        k_mask = k_rows < T
        kv_offset = base_qkv + k_rows[:, None] * HEAD_DIM + offset_d[None, :]

        k = tl.load(K_ptr + kv_offset, mask=k_mask[:, None] & d_mask[None, :], other=0.0,)
        v = tl.load(V_ptr + kv_offset, mask=k_mask[:, None] & d_mask[None, :], other=0.0,)

        # score_e = sm_scale * QK^T
        # score_2 = score_e * log2(e)
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        scores = scores * (sm_scale * 1.4426950408889634)

        # mask invalid query/key rows before softmax
        scores = tl.where(q_mask[:, None] & k_mask[None, :], scores, -float("inf"))

        # online softmax update
        m_new = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m - m_new)
        p = tl.math.exp2(scores - m_new[:, None])
        l_new = alpha * l + tl.sum(p, axis=1)

        # rescale old accumulator and add this block's contribution
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v, out_dtype=tl.float32,)
        m = m_new
        l = l_new

    # normalize output
    out = acc / l[:, None]

    # L is stored in log2 units:
    # L_i = log2(sum_j exp(score_ij))
    L_vals = m + tl.math.log2(l)

    tl.store(O_ptr + base_qkv + offset_q[:, None] * HEAD_DIM + offset_d[None, :], out, mask=q_mask[:, None] & d_mask[None, :],)
    tl.store(L_ptr + base_l + offset_q, L_vals, mask=q_mask,)

def sparse_flash_forward(Q, K, V, q_row_offsets, q_col_indices,
                         sm_scale, BLOCK_Q, BLOCK_K):
    """A2 -- block-sparse flash attention forward. See ALGORITHMS.md sec 1, 3.

    Inputs:
      Q, K, V        (B, H, T, d)         fp16
      q_row_offsets  (T//block + 1,)      int32  query-block view: for query
      q_col_indices  (nnz,)               int32  block i, its live key blocks j
      sm_scale       float                       1/sqrt(d)
      BLOCK_Q, BLOCK_K  ints                     == block (the mask granularity)
    Returns:
      O              (B, H, T, d)         fp16
      L              (B, H, T)            fp32   log2 of the softmax denominator (sec 3)

    See ALGORITHMS.md sec 3 for O and L.

    TODO: implement.
    """
    B, H, T, d = Q.shape
    n_q = triton.cdiv(T, BLOCK_Q)
    O = torch.empty_like(Q)
    L = torch.empty((B, H, T), device=Q.device, dtype=torch.float32)

    # flatten (B, H, T, d) to (B*H, T, d)
    Qf = Q.reshape(B * H, T, d)
    Kf = K.reshape(B * H, T, d)
    Vf = V.reshape(B * H, T, d)
    Of = O.reshape(B * H, T, d)
    Lf = L.reshape(B * H, T)

    BLOCK_D = triton.next_power_of_2(d)
    grid = (n_q, B * H)

    sparse_flash_forward_kernel[grid](Qf, Kf, Vf, Of, Lf, q_row_offsets, q_col_indices, sm_scale,
        T, d, BLOCK_Q, BLOCK_K, BLOCK_D,)

    return O, L
    

@triton.jit
def _sparse_attn_bwd_d_kernel(O_ptr, dO_ptr, D_ptr, T: tl.constexpr, HEAD_DIM: tl.constexpr, 
                              BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    offset_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offset_d = tl.arange(0, BLOCK_D)
    q_mask = offset_q < T
    d_mask = offset_d < HEAD_DIM

    base = pid_bh * T * HEAD_DIM
    base_d = pid_bh * T

    offset = base + offset_q[:, None] * HEAD_DIM + offset_d[None, :]
    o = tl.load(O_ptr + offset, mask=q_mask[:, None] & d_mask[None, :], other=0.0,).to(tl.float32)

    do = tl.load(dO_ptr + offset, mask=q_mask[:, None] & d_mask[None, :], other=0.0,).to(tl.float32)

    # D_i = sum_d dO_i[d] * O_i[d]
    d_vals = tl.sum(o * do, axis=1)

    tl.store(D_ptr + base_d + offset_q, d_vals, mask=q_mask,)


@triton.jit
def _sparse_attn_bwd_dkdv_kernel(Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dK_ptr, dV_ptr, k_row_offsets_ptr,
    k_col_indices_ptr, sm_scale, T: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, 
    BLOCK_D: tl.constexpr,):

    pid_k = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offset_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offset_q = tl.arange(0, BLOCK_Q)
    offset_d = tl.arange(0, BLOCK_D)

    k_mask = offset_k < T
    d_mask = offset_d < HEAD_DIM

    base = pid_bh * T * HEAD_DIM
    base_l = pid_bh * T

    offset = base + offset_k[:, None] * HEAD_DIM + offset_d[None, :]
    # load fixed K_j and V_j for this key block j
    k = tl.load(K_ptr + offset, mask=k_mask[:, None] & d_mask[None, :], other=0.0,)
    v = tl.load(V_ptr + offset, mask=k_mask[:, None] & d_mask[None, :], other=0.0,)

    dk_acc = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)

    # for key block j, loop over query blocks i that attend to j
    lo = tl.load(k_row_offsets_ptr + pid_k)
    hi = tl.load(k_row_offsets_ptr + pid_k + 1)

    for idx in range(lo, hi):
        q_blk = tl.load(k_col_indices_ptr + idx)
        q_rows = q_blk * BLOCK_Q + offset_q
        q_mask = q_rows < T

        offset_q_do = base + q_rows[:, None] * HEAD_DIM + offset_d[None, :]
        q = tl.load(Q_ptr + offset_q_do, mask=q_mask[:, None] & d_mask[None, :], other=0.0,)
        do = tl.load(dO_ptr + offset_q_do, mask=q_mask[:, None] & d_mask[None, :], other=0.0,)
        l_i = tl.load(L_ptr + base_l + q_rows, mask=q_mask, other=0.0,)
        d_i = tl.load(D_ptr + base_l + q_rows, mask=q_mask, other=0.0,)

        # recompute P_ij
        # forward stored L in log2 units so use exp2.
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        scores = scores * (sm_scale * 1.4426950408889634)
        scores = tl.where(q_mask[:, None] & k_mask[None, :], scores, -float("inf"))

        p = tl.math.exp2(scores - l_i[:, None])

        # dV_j += P_ij^T dO_i
        dv_acc += tl.dot(tl.trans(p.to(tl.float16)), do, out_dtype=tl.float32)

        # dP_ij = dO_i V_j^T
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        # dS_ij = P_ij * (dP_ij - D_i)
        ds = p * (dp - d_i[:, None])
        ds = tl.where(q_mask[:, None] & k_mask[None, :], ds, 0.0)

        # dK_j += dS_ij^T Q_i * sm_scale
        dk_acc += tl.dot(tl.trans(ds.to(tl.float16)), q, out_dtype=tl.float32) * sm_scale

    tl.store(dK_ptr + offset, dk_acc, mask=k_mask[:, None] & d_mask[None, :],)
    tl.store(dV_ptr + offset, dv_acc, mask=k_mask[:, None] & d_mask[None, :],)


@triton.jit
def _sparse_attn_bwd_dq_kernel(Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dQ_ptr,
    q_row_offsets_ptr, q_col_indices_ptr, sm_scale, T: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,):

    # one program computes one query block's dQ for one batch/head
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offset_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offset_k = tl.arange(0, BLOCK_K)
    offset_d = tl.arange(0, BLOCK_D)

    q_mask = offset_q < T
    d_mask = offset_d < HEAD_DIM

    base = pid_bh * T * HEAD_DIM
    base_l = pid_bh * T

    offset = base + offset_q[:, None] * HEAD_DIM + offset_d[None, :]
    q = tl.load(Q_ptr + offset, mask=q_mask[:, None] & d_mask[None, :], other=0.0,)
    do = tl.load(dO_ptr + offset, mask=q_mask[:, None] & d_mask[None, :], other=0.0,)
    l_i = tl.load(L_ptr + base_l + offset_q, mask=q_mask, other=0.0,)
    d_i = tl.load(D_ptr + base_l + offset_q, mask=q_mask, other=0.0,)

    dq_acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    # for query block i, loop over its live key blocks j
    lo = tl.load(q_row_offsets_ptr + pid_q)
    hi = tl.load(q_row_offsets_ptr + pid_q + 1)

    for idx in range(lo, hi):
        k_blk = tl.load(q_col_indices_ptr + idx)
        k_rows = k_blk * BLOCK_K + offset_k
        k_mask = k_rows < T
        kv_mask = (k_mask[:, None]) & (d_mask[None, :])
        offset_kv = base + k_rows[:, None] * HEAD_DIM + offset_d[None, :]

        k = tl.load(K_ptr + offset_kv, mask=kv_mask, other=0.0,)
        v = tl.load(V_ptr + offset_kv, mask=kv_mask,other=0.0,)

        # recompute P_ij
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        scores = scores * (sm_scale * 1.4426950408889634)
        scores = tl.where(q_mask[:, None] & k_mask[None, :], scores, -float("inf"))

        p = tl.math.exp2(scores - l_i[:, None])

        # dP_ij = dO_i V_j^T
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        # dS_ij = P_ij * (dP_ij - D_i)
        ds = p * (dp - d_i[:, None])
        ds = tl.where(q_mask[:, None] & k_mask[None, :], ds, 0.0)

        # dQ_i += dS_ij K_j * sm_scale
        dq_acc += tl.dot(ds.to(tl.float16), k, out_dtype=tl.float32) * sm_scale

    offset_dq = base + offset_q[:, None] * HEAD_DIM + offset_d[None, :]
    tl.store(dQ_ptr + offset_dq, dq_acc, mask=q_mask[:, None] & d_mask[None, :],
    )


def sparse_flash_backward(Q, K, V, O, L, dO,
                          k_row_offsets, k_col_indices,   # key-block view (sec 1)
                          q_row_offsets, q_col_indices,   # query-block view (sec 1)
                          sm_scale, BLOCK_Q, BLOCK_K):
    """A3 -- block-sparse flash attention backward. See ALGORITHMS.md sec 1, 4.

    Inputs:
      Q, K, V, O, dO (B, H, T, d)         fp16   O, dO are the forward output and its grad
      L              (B, H, T)            fp32   the forward residual
      k_row_offsets  (T//block + 1,)      int32  key-block view: for key block j,
      k_col_indices  (nnz,)               int32  the query blocks i that attend it
      q_row_offsets  (T//block + 1,)      int32  query-block view: for query block i,
      q_col_indices  (nnz,)               int32  its key blocks j (same as forward)
      sm_scale       float
      BLOCK_Q, BLOCK_K  ints                     == block
    Returns:
      dQ, dK, dV     (B, H, T, d)         fp16

    See ALGORITHMS.md sec 4 for the five gradient equations.

    TODO: implement.
    """
    B, H, T, d = Q.shape
    BH = B * H

    Qf = Q.reshape(BH, T, d)
    Kf = K.reshape(BH, T, d)
    Vf = V.reshape(BH, T, d)
    Of = O.reshape(BH, T, d)
    dOf = dO.reshape(BH, T, d)
    Lf = L.reshape(BH, T)

    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)

    dQf = dQ.reshape(BH, T, d)
    dKf = dK.reshape(BH, T, d)
    dVf = dV.reshape(BH, T, d)

    D = torch.empty((B, H, T), device=Q.device, dtype=torch.float32)
    Df = D.reshape(BH, T)

    BLOCK_D = triton.next_power_of_2(d)

    n_q = triton.cdiv(T, BLOCK_Q)
    n_k = triton.cdiv(T, BLOCK_K)

    grid_d = (n_q, BH)
    _sparse_attn_bwd_d_kernel[grid_d](Of, dOf, Df, T, d, BLOCK_Q,BLOCK_D,)

    grid_dkdv = (n_k, BH)
    _sparse_attn_bwd_dkdv_kernel[grid_dkdv](Qf, Kf, Vf, dOf, Lf, Df, dKf, dVf, k_row_offsets, k_col_indices,
        sm_scale, T, d, BLOCK_Q, BLOCK_K, BLOCK_D,)

    grid_dq = (n_q, BH)
    _sparse_attn_bwd_dq_kernel[grid_dq](Qf, Kf, Vf, dOf, Lf, Df, dQf, q_row_offsets, q_col_indices,
        sm_scale, T, d, BLOCK_Q, BLOCK_K, BLOCK_D,)

    return dQ, dK, dV