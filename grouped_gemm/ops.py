from grouped_gemm import backend
import torch


class GroupedGemm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x_a, w_b, batch_sizes, trans_b, tp_group, mode):
        ctx.save_for_backward(x_a, w_b, batch_sizes)
        ctx.trans_b = trans_b
        ctx.tp_group = tp_group
        ctx.mode = mode
        results = backend.gmm(x_a, w_b, batch_sizes, trans_a=False, trans_b=trans_b)
        return results 

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        x_a, w_b, batch_sizes = ctx.saved_tensors
        trans_b = ctx.trans_b

        x_agrad = None
        handle = None
        if ctx.needs_input_grad[0]:
            x_agrad = backend.gmm(
                grad, w_b, batch_sizes, trans_a=False, trans_b=not trans_b)
            if ctx.mode == 'column':
                handle = torch.distributed.all_reduce(
                    x_agrad, 
                    op=torch.distributed.ReduceOp.SUM, 
                    group=ctx.tp_group,
                    async_op=True,
                )

        w_bgrad = None
        if ctx.needs_input_grad[1]:
            lhs, rhs = (grad, x_a) if trans_b else (x_a, grad)
            w_bgrad = backend.gmm(
                lhs, rhs, batch_sizes, trans_a=True, trans_b=False)
        
        if handle is not None:
            handle.wait()

        return x_agrad, w_bgrad, None, None, None, None


def gmm(x_a, w_b, batch_sizes, trans_b=False, tp_group=None, mode=None):
    return GroupedGemm.apply(x_a, w_b, batch_sizes, trans_b, tp_group, mode)
