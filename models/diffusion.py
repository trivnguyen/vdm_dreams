
import dataclasses
import os
from typing import Union
from pathlib import Path

import datasets
import flax.linen as nn
import jax
import jax.numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
import yaml
from absl import logging
from flax.core import FrozenDict
from flax.training import checkpoints, train_state
from ml_collections.config_dict import ConfigDict
from models.diffusion_utils import (
    NoiseScheduleFixedLinear, NoiseScheduleNet, NoiseScheduleScalar, alpha, sigma2,
                                    variance_preserving_map)
from models.mlp import MLPDecoder, MLPEncoder
from models.scores import GraphScoreNet, TransformerScoreNet

tfd = tfp.distributions


class VariationalDiffusionModel(nn.Module):
    """Variational Diffusion Model (VDM), adapted from https://github.com/google-research/vdm

    Attributes:
      d_feature: Number of features per set element.
      timesteps: Number of diffusion steps.
      gamma_min: Minimum log-SNR in the noise schedule (init if learned).
      gamma_max: Maximum log-SNR in the noise schedule (init if learned).
      antithetic_time_sampling: Antithetic time sampling to reduce variance.
      noise_schedule: Noise schedule; "learned_linear", "linear", or "learned_net"
      noise_scale: Std of Normal noise model.
      noise_scale_mass: Std of Normal noise model for mass.
      add_mass_recon_loss: Whether to add a mass reconstruction loss.
      add_mass_conservation_loss: Whether to add a mass conservation loss.
      mass_decay_rate: Decay rate for mass conservation loss.
      d_t_embedding: Dimensions the timesteps are embedded to.
      score: Score function; "transformer", "graph".
      score_dict: Dict of score arguments (see scores.py docstrings).
      n_classes: Number of classes in data. If >0, the first element of the conditioning vector is assumed to be integer class.
      embed_context: Whether to embed the conditioning context.
      use_encdec: Whether to use an encoder-decoder for latent diffusion.
      use_pos_enc: Whether to use positional encoding.
      norm_dict: Dict of normalization arguments (see datasets.py docstrings).
      n_pos_features: Number of positional features, for graph-building etc.
      n_vel_features: Number of velocity features.
      n_mass_features: Number of mass features.
      scale_non_linear_init: Whether to scale the initialization of the non-linear layers in the noise model.
    """

    d_feature: int = 3
    timesteps: int = 1000
    gamma_min: float = -8.0
    gamma_max: float = 14.0
    antithetic_time_sampling: bool = True
    noise_schedule: str = "linear"  # "linear", "learned_linear", or "learned_net"
    noise_scale: float = 1.0e-3
    noise_scale_mass: float = 1.0e-3
    add_mass_recon_loss: bool = True
    add_mass_conservation_loss: bool = False
    mass_decay_rate: float = 10.0
    d_t_embedding: int = 32
    score: str = "transformer"  # "transformer", "graph"
    score_dict: dict = dataclasses.field(
        default_factory=lambda: {
            "d_model": 256,
            "d_mlp": 512,
            "n_layers": 4,
            "n_heads": 4,
        }
    )
    encoder_dict: dict = dataclasses.field(
        default_factory=lambda: {"d_embedding": 12, "d_hidden": 256, "n_layers": 4}
    )
    decoder_dict: dict = dataclasses.field(
        default_factory=lambda: {"d_hidden": 256, "n_layers": 4}
    )
    embed_context: bool = False
    d_context_embedding: int = 32
    use_encdec: bool = True
    use_pos_enc: bool = True
    norm_dict: dict = dataclasses.field(
        default_factory=lambda: {
            "x_mean": 0.0, "x_std": 1.0, "cond_mean": 0.0, "cond_std": 1.0,
            "box_size": 1}
    )
    n_classes: int = 0
    n_pos_features: int = 3
    n_vel_features: int = 3
    n_mass_features: int = 1
    scale_non_linear_init: bool = False

    @classmethod
    def from_path_to_model(
        cls,
        path_to_model: Union[str, Path],
        checkpoint_step: int = None,
        norm_dict: dict = None,
    ) -> "VariationalDiffusionModel":
        """load model from path where it is stored

        Args:
            path_to_model (Union[str, Path]): path to model

        Returns:
            Tuple[VariationalDiffusionModel, np.array]: model, params
        """
        with open(os.path.join(path_to_model, "config.yaml"), "r") as file:
            config = yaml.safe_load(file)
        config = ConfigDict(config)

        # load in model and params
        score_dict = FrozenDict(config.score)
        encoder_dict = FrozenDict(config.encoder)
        decoder_dict = FrozenDict(config.decoder)
        if norm_dict is None:
            _, norm_dict = datasets.read_dataloader(
                config.data.dataset_root,
                config.data.dataset_name,
                config.data.n_features,
                config.data.n_particles,
                config.training.batch_size,
                seed=config.seed,
                shuffle=True,
                repeat=True,
                conditioning_parameters=config.data.conditioning_parameters,
            )
        norm_dict_input = FrozenDict(
            {
                "x_mean": tuple(map(float, norm_dict["mean"])),
                "x_std": tuple(map(float, norm_dict["std"])),
                "cond_mean": tuple(map(float, norm_dict["cond_mean"])),
                "cond_std": tuple(map(float, norm_dict["cond_std"])),
                "box_size": config.data.box_size
            }
        )
        vdm = VariationalDiffusionModel(
            d_feature=config.data.n_features,
            timesteps=config.vdm.timesteps,
            gamma_min=config.vdm.gamma_min,
            gamma_max=config.vdm.gamma_max,
            noise_schedule=config.vdm.noise_schedule,
            noise_scale=config.vdm.noise_scale,
            noise_scale_mass=config.vdm.noise_scale_mass,
            add_mass_recon_loss=config.vdm.add_mass_recon_loss,
            add_mass_conservation_loss=config.vdm.add_mass_conservation_loss,
            mass_decay_rate=config.vdm.mass_decay_rate,
            d_t_embedding=config.vdm.d_t_embedding,
            score=config.score.score,
            score_dict=score_dict,
            encoder_dict=encoder_dict,
            decoder_dict=decoder_dict,
            n_classes=config.vdm.n_classes,
            embed_context=config.vdm.embed_context,
            d_context_embedding=config.vdm.d_context_embedding,
            use_encdec=config.vdm.use_encdec,
            norm_dict=norm_dict_input,
            n_pos_features=config.data.n_pos_features,
            n_vel_features=config.data.n_vel_features,
            n_mass_features=config.data.n_mass_features,
        )
        rng = jax.random.PRNGKey(42)
        x_dummy = jax.random.normal(
            rng, (config.training.batch_size, config.data.n_particles, config.data.n_features))
        conditioning_dummy = jax.random.normal(
            rng, (config.training.batch_size, len(config.data.conditioning_parameters)))
        mask_dummy = np.ones((config.training.batch_size, config.data.n_particles))
        position_dummy = np.ones(
            (config.training.batch_size, config.data.n_particles, 2))
        _, params = vdm.init_with_output(
            {"sample": rng, "params": rng},
            x_dummy, conditioning_dummy, mask_dummy, position_dummy
        )

        # load in optimizer state
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.optim.learning_rate,
            warmup_steps=config.training.warmup_steps,
            decay_steps=config.training.n_train_steps,
        )
        tx = optax.adamw(learning_rate=schedule, weight_decay=config.optim.weight_decay)
        if hasattr(config.optim, "grad_clip"):
            if config.optim.grad_clip is not None:
                tx = optax.chain(
                    optax.clip(config.optim.grad_clip),
                    tx,
                )
        state = train_state.TrainState.create(apply_fn=vdm.apply, params=params, tx=tx)
        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=path_to_model, target=state, step=checkpoint_step,)
        if state is restored_state:
            raise FileNotFoundError(f"Did not load checkpoint correctly")

        # return model and params
        return vdm, restored_state.params

    def setup(self):
        # Noise schedule for diffusion
        if self.noise_schedule == "linear":
            self.gamma = NoiseScheduleFixedLinear(
                gamma_min=self.gamma_min, gamma_max=self.gamma_max
            )
        elif self.noise_schedule == "learned_linear":
            self.gamma = NoiseScheduleScalar(
                gamma_min=self.gamma_min, gamma_max=self.gamma_max
            )
        elif self.noise_schedule == "learned_net":
            self.gamma = NoiseScheduleNet(
                gamma_min=self.gamma_min,
                gamma_max=self.gamma_max,
                scale_non_linear_init=self.scale_non_linear_init,
            )
        else:
            raise NotImplementedError(f"Unknown noise schedule {self.noise_schedule}")

        # Score model specification
        if self.score == "transformer":
            self.score_model = TransformerScoreNet(
                d_t_embedding=self.d_t_embedding,
                score_dict=self.score_dict,
                adanorm=False,
            )
        elif self.score == "transformer_adanorm":
            self.score_model = TransformerScoreNet(
                d_t_embedding=self.d_t_embedding,
                score_dict=self.score_dict,
                adanorm=True,
            )
        elif self.score in ["graph", "chebconv", "edgeconv"]:
            self.score_model = GraphScoreNet(
                d_t_embedding=self.d_t_embedding,
                score_dict=self.score_dict,
                norm_dict=self.norm_dict,
                gnn_type=self.score,
            )
        else:
            raise NotImplementedError(f"Unknown score model {self.score}")

        # Optional encoder/decoder for latent diffusion
        if self.use_encdec:
            self.encoder = MLPEncoder(**self.encoder_dict)

            self.decoder = MLPDecoder(
                d_output=self.d_feature,
                noise_scale=self.noise_scale,
                **self.decoder_dict,
            )

        # Embedding for class and context
        if self.n_classes > 0:
            self.embedding_class = nn.Embed(self.n_classes, self.d_context_embedding)
        self.embedding_context = nn.Dense(self.d_context_embedding)

    def gammat(self, t):
        return self.gamma(t)

    def recon_mass_loss(self, x, z0, cond, mask=None):
        """ Additional term to the recon_loss that enforces mass conservation. """
        # get the mass features of each particle
        i_start = self.n_pos_features + self.n_vel_features
        i_stop = i_start + self.n_mass_features
        logm_scale = np.array(self.norm_dict['x_std'][i_start:i_stop])
        logm_loc = np.array(self.norm_dict['x_mean'][i_start:i_stop])
        logm_x = x[..., i_start:i_stop] * logm_scale + logm_loc
        logm_z0 = z0[..., i_start:i_stop] * logm_scale + logm_loc

        # reconstruction loss over the total mass
        # TODO: numerical stability; use logsumexp
        if mask is not None:
            logm_x = np.log10(np.sum(np.where(mask[..., None], 10**logm_x, 0), axis=-2))
            logm_z0 = np.log10(np.sum(np.where(mask[..., None], 10**logm_z0, 0), axis=-2))
        else:
            logm_x = np.log10(np.sum(10**logm_x, axis=-2))
            logm_z0 = np.log10(np.sum(10**logm_z0, axis=-2))
        logm_x_rescaled = (logm_x - logm_loc) / logm_scale
        logm_z0_rescaled = (logm_z0 - logm_loc) / logm_scale
        loss_recon_mass = -tfd.Normal(
            loc=logm_z0_rescaled, scale=self.noise_scale_mass).log_prob(logm_x_rescaled)

        # mass conservation penalty
        if self.add_mass_conservation_loss:
            # get the central subhalo mass and the halo masss
            cond_scale = np.array(self.norm_dict['cond_std'])[:self.n_mass_features]
            cond_loc = np.array(self.norm_dict['cond_mean'])[:self.n_mass_features]
            cond_unscaled = cond[..., :self.n_mass_features] * cond_scale + cond_loc
            dlogm = logm_z0 - cond_unscaled

            # exponential penalty
            loss_conserve = nn.relu(dlogm) * self.mass_decay_rate
            loss_recon_mass = loss_recon_mass + loss_conserve

        return loss_recon_mass

    def recon_loss(self, x, f, cond, mask=None):
        """The reconstruction loss measures the gap in the first step.
        We measure the gap from encoding the image to z_0 and back again.
        """
        g_0 = self.gamma(0.0)
        eps_0 = jax.random.normal(self.make_rng("sample"), shape=f.shape)
        z_0 = variance_preserving_map(f, g_0, eps_0)
        z_0_rescaled = z_0 / alpha(g_0)
        # z_0_rescaled = z_0

        # features reconstruction term
        loss_recon = -self.decode(z_0_rescaled, cond).log_prob(x)

        # add the mass reconstruction term
        if self.add_mass_recon_loss:
            loss_recon_mass = self.recon_mass_loss(x, z_0_rescaled, cond, mask)
        else:
            loss_recon_mass = None
        return loss_recon, loss_recon_mass

    def latent_loss(self, f):
        """The latent loss measures the gap in the last step, this is the KL
        divergence between the final sample from the forward process and starting
        distribution for the reverse process, here taken to be a N(0,1).
        """
        g_1 = self.gamma(1.0)
        var_1 = sigma2(g_1)
        mean1_sqr = (1.0 - var_1) * np.square(f)
        loss_klz = 0.5 * (mean1_sqr + var_1 - np.log(var_1) - 1.0)
        return loss_klz

    def diffusion_loss(self, t, f, cond, mask, position_enc):
        """The diffusion loss measures the gap in the intermediate steps."""
        # Sample z_t
        g_t = self.gamma(t)
        eps = jax.random.normal(self.make_rng("sample"), shape=f.shape)
        z_t = variance_preserving_map(f, g_t[:, None], eps)
        # Compute predicted noise
        eps_hat = self.score_model(
            z_t,
            g_t,
            cond,
            mask,
            position_enc
        )
        deps = eps - eps_hat
        loss_diff_mse = np.square(deps)  # Compute MSE of predicted noise
        T = self.timesteps
        # NOTE: retain dimension here so that mask can be applied later (hence dummy dims)
        # NOTE: opposite sign convention to official VDM repo!
        if T == 0:
            # Loss for infinite depth T, i.e. continuous time
            _, g_t_grad = jax.jvp(self.gamma, (t,), (np.ones_like(t),))
            loss_diff = -0.5 * g_t_grad[:, None, None] * loss_diff_mse
        else:
            # Loss for finite depth T, i.e. discrete time
            s = t - (1.0 / T)
            g_s = self.gamma(s)
            loss_diff = 0.5 * T * np.expm1(g_s - g_t)[:, None, None] * loss_diff_mse

        return loss_diff

    def __call__(self, x, conditioning=None, mask=None, position_encoding=None):
        d_batch = x.shape[0]

        # 1. Reconstruction loss
        # Add noise and reconstruct
        f = self.encode(x, conditioning)
        loss_recon, loss_recon_mass = self.recon_loss(x, f, conditioning, mask)

        # 2. Latent loss
        # KL z1 with N(0,1) prior
        loss_klz = self.latent_loss(f)

        # 3. Diffusion loss
        # Sample time steps
        rng1 = self.make_rng("sample")
        if self.antithetic_time_sampling:
            t0 = jax.random.uniform(rng1)
            t = np.mod(t0 + np.arange(0.0, 1.0, step=1.0 / d_batch), 1.0)
        else:
            t = jax.random.uniform(rng1, shape=(d_batch,))
        # Discretize time steps if we're working with discrete time
        T = self.timesteps
        if T > 0:
            t = np.ceil(t * T) / T
        cond = self.embed(conditioning)
        loss_diff = self.diffusion_loss(t, f, cond, mask, position_encoding)

        return (loss_diff, loss_klz, loss_recon, loss_recon_mass)

    def embed(self, conditioning):
        """Embed the conditioning vector, optionally including embedding a class
        assumed to be the first element of the context vector."""
        if not self.embed_context:
            return conditioning
        else:
            if (
                self.n_classes > 0 and conditioning.shape[-1] > 1
            ):  # If both classes and conditioning
                classes, conditioning = (
                    conditioning[..., 0].astype(np.int32),
                    conditioning[..., 1:],
                )
                class_embedding, context_embedding = self.embedding_class(
                    classes
                ), self.embedding_context(conditioning)
                return class_embedding + context_embedding
            elif (
                self.n_classes > 0 and conditioning.shape[-1] == 1
            ):  # If no conditioning but classes
                classes = conditioning[..., 0].astype(np.int32)
                class_embedding = self.embedding_class(classes)
                return class_embedding
            elif (
                self.n_classes == 0 and conditioning is not None
            ):  # If no classes but conditioning
                context_embedding = self.embedding_context(conditioning)
                return context_embedding
            else:  # If no conditioning
                return None

    def encode(self, x, conditioning=None, mask=None):
        """Encode an image x."""

        # Encode if using encoder-decoder; otherwise just return data sample
        if self.use_encdec:
            if conditioning is not None:
                cond = self.embed(conditioning)
            else:
                cond = None
            return self.encoder(x, cond, mask)
        else:
            return x

    def decode(
        self,
        z0,
        conditioning=None,
        mask=None,
    ):
        """Decode a latent sample z0."""

        # Decode if using encoder-decoder; otherwise just return last latent distribution
        if self.use_encdec:
            if conditioning is not None:
                cond = self.embed(conditioning)
            else:
                cond = None
            return self.decoder(z0, cond, mask)
        else:
            return tfd.Normal(loc=z0, scale=self.noise_scale)

    def sample_step(self, rng, i, T, z_t, conditioning=None, mask=None, position_encoding=None):
        """Sample a single step of the diffusion process."""
        rng_body = jax.random.fold_in(rng, i)
        eps = jax.random.normal(rng_body, z_t.shape)
        t = (T - i) / T
        s = (T - i - 1) / T

        g_s = self.gamma(s)
        g_t = self.gamma(t)
        cond = self.embed(conditioning)
        eps_hat_cond = self.score_model(
            z_t,
            g_t * np.ones((z_t.shape[0],), z_t.dtype),
            cond,
            mask,
            position_encoding
        )

        a = nn.sigmoid(g_s)
        b = nn.sigmoid(g_t)
        c = -np.expm1(g_t - g_s)
        sigma_t = np.sqrt(sigma2(g_t))
        z_s = (
            np.sqrt(a / b) * (z_t - sigma_t * c * eps_hat_cond)
            + np.sqrt((1.0 - a) * c) * eps
        )

        return z_s

    # def evaluate_score(
    #     self,
    #     z_t,
    #     g_t,
    #     cond,
    #     mask,
    #     position_enc,
    # ):
    #     return self.score_model(
    #         z=z_t,
    #         t=g_t,
    #         conditioning=cond,
    #         mask=mask,
    #         position_encoding=position_enc,
    #     )

    # def score_eval(self, z, t, conditioning, mask, position_enc):
        # """Evaluate the score model."""
        # cond = self.embed(conditioning)
        # return self.score_model(
        #     z=z,
        #     t=t,
        #     conditioning=cond,
        #     mask=mask,
        #     position_encoding=position_enc,
        # )

