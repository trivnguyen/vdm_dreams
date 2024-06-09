
from typing import Any, List, Optional, Union
import dataclasses
from pathlib import Path

import yaml
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.linen.module import compact
from flax.training import train_state, checkpoints
import distrax
from ml_collections.config_dict import ConfigDict

from .bijectors import InverseConditional, ChainConditional, TransformedConditional, MaskedCouplingConditional

Array = Any

class Conditioner(nn.Module):
    event_shape: List[int]
    context_shape: List[int]
    hidden_dims: List[int]
    num_bijector_params: int
    activation: str = "relu"
    dropout_rate: float = 0.0  # default dropout rate
    batch_norm: bool = False

    @compact
    def __call__(self, x: Array, context=None, training: bool = True):
        # Infer batch dims
        batch_shape = x.shape[: -len(self.event_shape)]
        batch_shape_context = context.shape[: -len(self.context_shape)]
        assert batch_shape == batch_shape_context

        # Flatten event dims
        x = x.reshape(*batch_shape, -1)
        context = context.reshape(*batch_shape, -1)

        x = jnp.hstack([context, x])

        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = getattr(jax.nn, self.activation)(x)
            if self.batch_norm:
                x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

        x = nn.Dense(
            np.prod(self.event_shape) * self.num_bijector_params,
            kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros)(x)
        x = x.reshape(*batch_shape, *(tuple(self.event_shape) + (self.num_bijector_params,)))

        return x


class NeuralSplineFlow(nn.Module):
    """Based on the implementation in the Distrax repo, https://github.com/deepmind/distrax/blob/master/examples/flow.py"""

    n_dim: int
    n_context: int = 0
    n_transforms: int = 4
    hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [128, 128])
    activation: str = "gelu"
    n_bins: int = 8
    range_min: float = -1.0
    range_max: float = 1.0
    event_shape: Optional[List[int]] = None
    context_shape: Optional[List[int]] = None
    dropout_rate: float = 0.0
    batch_norm: bool = False

    @classmethod
    def from_path_to_model(
        cls,
        path_to_model: Union[str, Path],
        checkpoint_step: int = None,
    ) -> "NeuralSplineFlow":
        """load model from path where it is stored

        Args:
            path_to_model (Union[str, Path]): path to model

        Returns:
            Tuple[NeuralSplineFlow, np.array]: model, params
        """
        with open(path_to_model / "config.yaml", "r") as file:
            config = yaml.safe_load(file)
        config = ConfigDict(config)

        model = NeuralSplineFlow(
            n_dim=config.flows.n_dim,
            n_context=config.flows.n_context,
            hidden_dims=config.flows.hidden_dims,
            n_transforms=config.flows.n_transforms,
            activation=config.flows.activation,
            n_bins=config.flows.n_bins,
            range_min=config.flows.range_min,
            range_max=config.flows.range_max,
            batch_norm=config.flows.batch_norm,
            dropout_rate=config.flows.dropout_rate,
        )

        # initialize the model
        key = jax.random.PRNGKey(42)
        x_dummy = jax.random.uniform(key=key, shape=(64, config.flows.n_context))
        theta_dummy = jax.random.uniform(key=key, shape=(64, config.flows.n_dim))
        params = model.init(key, theta_dummy, x_dummy)

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.optim.learning_rate,
            warmup_steps=config.flow_training.warmup_steps,
            decay_steps=config.flow_training.n_train_steps,
        )
        tx = optax.adamw(learning_rate=schedule, weight_decay=config.optim.weight_decay)
        if hasattr(config.optim, "grad_clip"):
            if config.optim.grad_clip is not None:
                tx = optax.chain(
                    optax.clip(config.optim.grad_clip),
                    tx,
                )

        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        # Training config and state
        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=path_to_model,
            target=state,
            step=checkpoint_step,
        )
        if state is restored_state:
            raise FileNotFoundError(f"Did not load checkpoint correctly")
        return model, restored_state.params

    def setup(self):
        def bijector_fn(params: Array):
            return distrax.RationalQuadraticSpline(
                params, range_min=self.range_min, range_max=self.range_max)

        # If event shapes are not provided, assume single event and context dimensions
        event_shape = (self.n_dim,) if self.event_shape is None else self.event_shape
        context_shape = (self.n_context,) if self.context_shape is None else self.context_shape

        # Alternating binary mask
        mask = jnp.arange(0, np.prod(event_shape)) % 2
        mask = jnp.reshape(mask, event_shape)
        mask = mask.astype(bool)

        # Number of parameters for the rational-quadratic spline:
        # - `num_bins` bin widths
        # - `num_bins` bin heights
        # - `num_bins + 1` knot slopes
        # for a total of `3 * num_bins + 1` parameters
        num_bijector_params = 3 * self.n_bins + 1

        self.conditioner = [
            Conditioner(
                event_shape=event_shape, context_shape=context_shape, hidden_dims=self.hidden_dims,
                num_bijector_params=num_bijector_params, activation=self.activation,
                name="conditioner_{}".format(i)
            ) for i in range(self.n_transforms)
        ]

        bijectors = []
        for i in range(self.n_transforms):
            bijectors.append(
                MaskedCouplingConditional(
                    mask=mask, bijector=bijector_fn, conditioner=self.conditioner[i]))
            mask = jnp.logical_not(mask)  # Flip the mask after each layer

        self.bijector = InverseConditional(ChainConditional(bijectors))
        self.base_dist = distrax.MultivariateNormalDiag(jnp.zeros(event_shape), jnp.ones(event_shape))

        self.flow = TransformedConditional(self.base_dist, self.bijector)

    def __call__(self, x: Array, context: Array = None, training: bool = True) -> Array:
        return self.flow.log_prob(x, context=context, training=training)

    def sample(self, num_samples: int, rng: Array, context: Array = None) -> Array:
        return self.flow.sample(seed=rng, sample_shape=(num_samples,), context=context)
