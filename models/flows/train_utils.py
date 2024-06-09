
import jax
import jax.numpy as np
import flax
from ml_collections import ConfigDict

from functools import partial

def loss_flows(params, model, x, context):
    loss = -np.mean(model.apply(params, x, context, training=True))
    return loss

def train_step(state, batch, model, loss_fn):
    x, context = batch
    loss, grads = jax.value_and_grad(loss_fn)(
        state.params, model, x, context, training=True)
    new_state = state.apply_gradients(grads=grads)
    metrics = {"loss": loss}
    return new_state, metrics
