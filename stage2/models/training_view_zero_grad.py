"""Keep mixed-view data-parallel gradient layouts consistent for mono inputs."""

import torch


class _IdentityWithZeroViewGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, view_embedding):
        ctx.view_shape = tuple(view_embedding.shape)
        ctx.view_dtype = view_embedding.dtype
        ctx.view_device = view_embedding.device
        # Do not add a scalar +0: that can change negative-zero forward bits.
        return tokens

    @staticmethod
    def backward(ctx, grad_tokens):
        return grad_tokens, torch.zeros(
            ctx.view_shape, dtype=ctx.view_dtype, device=ctx.view_device
        )


def retain_zero_view_gradient(tokens, view_embedding):
    """Identity tokens with a zero local contribution to the view parameter.

    Mono data has no view offset, but must contribute its zero gradient to the
    same ZeRO reduction layout as stereo ranks. Tokens and their VJP are
    unchanged. In mixed-view DP the view parameter has a global gradient; this
    completes its local zero contribution. AdamW treats ``grad=None`` and an
    explicit zero gradient differently for an all-mono batch.
    """
    return _IdentityWithZeroViewGradient.apply(tokens, view_embedding)
