
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, List

import datasets
import eval
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow as tf
import models.diffusion
import numpy as np
import yaml
from absl import flags, logging
from ml_collections.config_dict import ConfigDict
from ml_collections import config_flags
from models.diffusion_utils import generate
from models.flows import maf, nsf
from models.train_utils import create_input_iter
from tqdm import tqdm

logging.set_verbosity(logging.INFO)


def infer(config: ConfigDict):

    # set the random seed
    rng = jax.random.PRNGKey(config.seed)

    # load the dataset
    logging.info("Loading the dataset...")
    x, conditioning, mask, position_encoding, norm_dict = datasets.read_dataset(
        config.data.dataset_root,
        config.data.dataset_name,
        config.data.n_features,
        config.data.n_particles,
        conditioning_parameters=config.data.conditioning_parameters,
    )
    x = x * norm_dict['std'] + norm_dict['mean']

    # load the VDM
    logging.info("Loading the VDM")
    path_to_vdm = Path(os.path.join(config.workdir, config.vdm_name))
    vdm, vdm_params = models.diffusion.VariationalDiffusionModel.from_path_to_model(
        path_to_model=path_to_vdm, norm_dict=norm_dict)

    # Iterate over the entire dataset and start generation
    dset = datasets.make_dataloader(
        (x, conditioning, mask, position_encoding), batch_size=config.batch_size,
        seed=config.seed, shuffle=False, repeat=False)
    dset = create_input_iter(dset)

    truth_samples = []
    truth_cond = []
    truth_mask = []
    truth_pos_enc = []
    vdm_samples = []
    vdm_cond = []
    vdm_mask = []
    vdm_pos_enc = []

    for batch in tqdm(dset):
        x_batch, cond_batch, mask_batch, position_encoding = batch
        x_batch = jnp.repeat(x_batch[0], config.n_repeats, axis=0)
        position_encoding = jnp.repeat(position_encoding[0], config.n_repeats, axis=0)
        truth_cond_batch = jnp.repeat(cond_batch[0], config.n_repeats, axis=0)
        truth_mask_batch = jnp.repeat(mask_batch[0], config.n_repeats, axis=0)
        num_batch = len(truth_cond_batch)

        vdm_samples_batch = eval.generate_samples(
                vdm=vdm,
                params=vdm_params,
                rng=rng,
                n_samples=num_batch,
                n_particles=config.data.n_particles,
                conditioning=truth_cond_batch,
                mask=truth_mask_batch,
                position_encoding=position_encoding,
                steps=config.steps,
                norm_dict=norm_dict,
            )

        # denormalize the conditioning vector
        truth_cond_batch = truth_cond_batch * norm_dict['cond_std'] + norm_dict['cond_mean']

        # store data
        truth_samples.append(x_batch)
        truth_cond.append(truth_cond_batch)
        truth_mask.append(truth_mask_batch)
        truth_pos_enc.append(position_encoding)
        vdm_samples.append(vdm_samples_batch)
        vdm_cond.append(truth_cond_batch)
        vdm_mask.append(truth_mask_batch)
        vdm_pos_enc.append(position_encoding)

    truth_samples = jnp.concatenate(truth_samples, axis=0)
    truth_mask = jnp.concatenate(truth_mask, axis=0)
    truth_cond = jnp.concatenate(truth_cond, axis=0)
    truth_pos_enc = jnp.concatenate(truth_pos_enc, axis=0)
    vdm_samples = jnp.concatenate(vdm_samples, axis=0)
    vdm_cond = jnp.concatenate(vdm_cond, axis=0)
    vdm_mask = jnp.concatenate(vdm_mask, axis=0)
    vdm_pos_enc = jnp.concatenate(vdm_pos_enc, axis=0)

    # Save the samples
    if config.output_name is None:
        vdm_base = os.path.basename(config.vdm_name)
        output_name = f'vdm/{vdm_base}.npz'
    else:
        output_name = config.output_name
    output_path = os.path.join(config.outdir, output_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logging.info("Saving the generated samples to %s", output_path)
    np.savez(
        output_path, samples=vdm_samples, cond=vdm_cond, mask=vdm_mask,
        truth=truth_samples, truth_cond=truth_cond, truth_mask=truth_mask,
        pos_enc=truth_pos_enc, vdm_pos_enc=vdm_pos_enc
    )

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
    infer(config=FLAGS.config)
