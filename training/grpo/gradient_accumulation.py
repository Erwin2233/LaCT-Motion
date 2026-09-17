"""DDP forward execution with the synchronization state set in advance."""


def forward_with_gradient_sync(model, *, sync_gradients, **inputs):
    """Select gradient synchronization before DDP prepares the backward pass.

    DDP reads this flag during forward, so changing it immediately before
    backward is too late. The final microbatch synchronizes accumulated gradients.
    """
    model.require_backward_grad_sync = sync_gradients
    return model(**inputs)
