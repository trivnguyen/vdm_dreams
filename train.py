
import os
import sys

import datasets
import flax
import jax
import jax.numpy as np
import ml_collections
import optax
import tensorflow as tf
import wandb
import yaml
from absl import flags, logging
from clu import metric_writers
from flax.core import FrozenDict
from flax.training import checkpoints, common_utils, train_state
from ml_collections import config_flags
from models.train_utils import (create_input_iter, param_count,
                                to_wandb_config, train_step)
from models.diffusion import VariationalDiffusionModel
from models.diffusion_utils import loss_vdm
from tqdm import trange

replicate = flax.jax_utils.replicate
unreplicate = flax.jax_utils.unreplicate

logging.set_verbosity(logging.INFO)

print("JAX devices: ", jax.devices())

def setup_model(
    config: ml_collections.ConfigDict, norm_dict: dict, batches: list,
    rng: jax.random.PRNGKey):
    """ Set up the VDM model """

    # Score and (optional) encoder model configs
    score_dict = FrozenDict(config.score)
    encoder_dict = FrozenDict(config.encoder)
    decoder_dict = FrozenDict(config.decoder)

    # Diffusion model
    norm_dict_input = FrozenDict({
        "x_mean": tuple(map(float, norm_dict["mean"])),
        "x_std": tuple(map(float, norm_dict["std"])),
        "cond_mean": tuple(map(float, norm_dict["cond_mean"])),
        "cond_std": tuple(map(float, norm_dict["cond_std"])),
        "box_size": config.data.box_size,
    })
    vdm = VariationalDiffusionModel(
        d_feature=config.data.n_features,
        timesteps=config.vdm.timesteps,
        noise_schedule=config.vdm.noise_schedule,
        noise_scale=config.vdm.noise_scale,
        noise_scale_mass=config.vdm.noise_scale_mass,
        add_mass_recon_loss=config.vdm.add_mass_recon_loss,
        add_mass_conservation_loss=config.vdm.add_mass_conservation_loss,
        mass_decay_rate=config.vdm.mass_decay_rate,
        d_t_embedding=config.vdm.d_t_embedding,
        gamma_min=config.vdm.gamma_min,
        gamma_max=config.vdm.gamma_max,
        score=config.score.score,
        score_dict=score_dict,
        embed_context=config.vdm.embed_context,
        d_context_embedding=config.vdm.d_context_embedding,
        n_classes=config.vdm.n_classes,
        use_encdec=config.vdm.use_encdec,
        encoder_dict=encoder_dict,
        decoder_dict=decoder_dict,
        norm_dict=norm_dict_input,
        n_pos_features=config.data.n_pos_features,
        n_vel_features=config.data.n_vel_features,
        n_mass_features=config.data.n_mass_features,
    )

    rng, rng_params = jax.random.split(rng)
    x_batch, conditioning_batch, mask_batch, position_batch = next(batches)
    _, params = vdm.init_with_output(
        {"sample": rng, "params": rng_params},
        x_batch[0],
        conditioning_batch[0],
        mask_batch[0],
        position_batch[0],
    )
    return vdm, params, rng

def train(
    config: ml_collections.ConfigDict, workdir: str = "./logging/"
) -> train_state.TrainState:
    """ Train the VDM model """

    # Set up wandb run
    if config.wandb.log_train and jax.process_index() == 0:
        wandb_config = to_wandb_config(config)
        run = wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            job_type=config.wandb.job_type,
            group=config.wandb.group,
            config=wandb_config,
        )
        wandb.define_metric(
            "*", step_metric="train/step"
        )  # Set default x-axis as 'train/step'
        workdir = os.path.join(workdir, run.group, run.name)
        os.makedirs(workdir, exist_ok=True)
    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0)
    with open(os.path.join(workdir, "config.yaml"), "w") as f:
        yaml.dump(config.to_dict(), f)

    # Load the dataset
    train_ds, norm_dict = datasets.read_dataloader(
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
    batches = create_input_iter(train_ds)
    logging.info("Loaded the %s dataset", config.data.dataset_name)

    # Set up the model
    rng = jax.random.PRNGKey(config.seed)
    vdm, params, rng = setup_model(config, norm_dict, batches, rng)

    logging.info("Instantiated the model")
    logging.info("Number of parameters: %d", param_count(params))

    # Set up the optimizer
    # Default schedule if not specified
    if not hasattr(config.optim, "lr_schedule"):
        config.optim.lr_schedule = "cosine"
    if config.optim.lr_schedule == "cosine":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.optim.learning_rate,
            warmup_steps=config.training.warmup_steps,
            decay_steps=config.training.n_train_steps,
        )
    elif config.optim.lr_schedule == "constant":
        schedule = optax.constant_schedule(config.optim.learning_rate)
    else:
        raise ValueError(f"Invalid learning rate schedule: {config.optim.lr_schedule}")

    tx = optax.adamw(learning_rate=schedule, weight_decay=config.optim.weight_decay)

    # Check if config.optim.grad_clip exists, if so add gradient clipping
    if hasattr(config.optim, "grad_clip"):
        if config.optim.grad_clip is not None:
            tx = optax.chain(
                optax.clip(config.optim.grad_clip),
                tx,
            )

    state = train_state.TrainState.create(apply_fn=vdm.apply, params=params, tx=tx)
    pstate = replicate(state)

    # Trainng loop
    logging.info("Starting training...")

    train_metrics = []
    with trange(config.training.n_train_steps) as steps:
        for step in steps:
            rng, *train_step_rng = jax.random.split(
                rng, num=jax.local_device_count() + 1
            )
            train_step_rng = np.asarray(train_step_rng)
            x, conditioning, mask, position_encoding = next(batches)
            if config.data.add_rotations or config.data.add_translations:
                x, conditioning, mask = datasets.augmentation.augment_data(
                    x=x,
                    mask=mask,
                    conditioning=conditioning,
                    rng=rng,
                    norm_dict=norm_dict,
                    n_pos_dim=config.data.n_pos_features,
                    box_size=config.data.box_size,
                    rotations=config.data.add_rotations,
                    translations=config.data.add_translations,
                )
            pstate, metrics = train_step(
                pstate,
                (x, conditioning, mask, position_encoding),
                train_step_rng,
                vdm,
                loss_vdm,
                config.training.unconditional_dropout,
                config.training.p_uncond,
            )
            steps.set_postfix(val=unreplicate(metrics["loss"]))
            train_metrics.append(metrics)

            # Log periodically
            if (
                (step % config.training.log_every_steps == 0)
                and (step != 0)
                and (jax.process_index() == 0)
            ):
                train_metrics = common_utils.get_metrics(train_metrics)
                summary = {
                    f"train/{k}": v
                    for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
                }

                writer.write_scalars(step, summary)
                train_metrics = []

                if config.wandb.log_train:
                    wandb.log({"train/step": step, **summary})

            # Save checkpoints periodically
            if (
                (step % config.training.save_every_steps == 0)
                and (step != 0)
                and (jax.process_index() == 0)
            ):
                state_ckpt = unreplicate(pstate)
                checkpoints.save_checkpoint(
                    ckpt_dir=workdir,
                    target=state_ckpt,
                    step=step,
                    overwrite=True,
                    keep=np.inf,
                )

    logging.info("All done! Have a great day.")

    return unreplicate(pstate)


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training or sampling hyperparameter configuration.",
        lock_config=True,
    )

    # Parse flags
    FLAGS(sys.argv)

    # Ensure TF does not see GPU and grab all GPU memory
    tf.config.experimental.set_visible_devices([], "GPU")

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())
    logging.info("JAX total visible devices: %r", jax.device_count())

    # Start training run
    train(config=FLAGS.config, workdir=FLAGS.config.wandb.workdir)
