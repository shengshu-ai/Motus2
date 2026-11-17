"""Adjacent-pair vector RoPE kernels.

Read consecutive elements in one packed tile, split the final size-2
axis, and apply FP32 rotation with matching backward operations.
"""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N', 'L'])
def packed_rope_forward(X, F, Y, N, L, SHARED: tl.constexpr,
                        BLOCK: tl.constexpr):
    N = (N // 3072) * 3072
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // 3072
    frow = row % L if SHARED else row
    fi = frow * 128 + i % 128
    x = tl.load(X + i, i < N, other=0).to(tl.float32)
    f = tl.load(F + fi, i < N, other=0).to(tl.float32)
    xr, xi = tl.split(tl.reshape(x, (BLOCK // 2, 2)))
    fr, fv = tl.split(tl.reshape(f, (BLOCK // 2, 2)))
    rr = xr * fr
    ii = xi * fv
    ri = xr * fv
    ir = xi * fr
    yr = rr - ii
    yi = ri + ir
    y = tl.reshape(tl.join(yr, yi), (BLOCK,))
    tl.store(Y + i, y, i < N)


@triton.jit(do_not_specialize=['N', 'L'])
def packed_rope_backward(G, F, DX, N, L, SHARED: tl.constexpr,
                         BLOCK: tl.constexpr):
    N = (N // 3072) * 3072
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // 3072
    frow = row % L if SHARED else row
    fi = frow * 128 + i % 128
    g = tl.load(G + i, i < N, other=0).to(tl.float32)
    f = tl.load(F + fi, i < N, other=0).to(tl.float32)
    gr, gi = tl.split(tl.reshape(g, (BLOCK // 2, 2)))
    fr, fv = tl.split(tl.reshape(f, (BLOCK // 2, 2)))
    imag_fr = gi * fr
    imag_fi = gi * fv
    neg_real_fi = (-gr) * fv
    real_fr = gr * fr
    dx_im = imag_fr + neg_real_fi
    dx_re = imag_fi + real_fr
    dx_im = dx_im + 0.0
    dx_re = 0.0 + dx_re
    dx = tl.reshape(tl.join(dx_re, dx_im), (BLOCK,))
    tl.store(DX + i, dx, i < N)
