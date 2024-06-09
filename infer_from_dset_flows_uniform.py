
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, List

import datasets
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow as tf
import models.eval_utils
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

@partial(jax.vmap, in_axes=(0, None))
def create_mask(n, num_particles):
    # Create an array [0, 1, 2, ..., num_particles-1]
    indices = jnp.arange(num_particles)
    # Compare each index to n, resulting in True (1) if index < n, else False (0)
    mask = indices < n
    return mask.astype(jnp.float32)


def infer(config: ConfigDict):

    # set the random seed
    rng = jax.random.PRNGKey(config.seed)

    # load the dataset
    logging.info("Loading the dataset...")
    _, _, _, _, norm_dict = datasets.read_dataset(
        config.data.dataset_root,
        config.data.dataset_name,
        config.data.n_features,
        config.data.n_particles,
        config.data.conditioning_parameters,
    )
    _, flows_conditioning, _, _, flows_norm_dict = datasets.read_dataset(
        config.data.dataset_root,
        config.data.dataset_name,
        config.data.n_features,
        config.data.n_particles,
        config.data.flows_labels + config.data.flows_conditioning_parameters,
    )
    num_flows_conditioning = len(config.data.flows_conditioning_parameters)
    num_flows_labels = len(config.data.flows_labels)

    # split the flows conditioning into the conditioning and labels
    flows_labels = flows_conditioning[:, :num_flows_labels]
    flows_conditioning = flows_conditioning[:, num_flows_labels:]

    # load the VDM and the normalizing flows
    logging.info("Loading the VDM and the normalizing flows...")
    path_to_vdm = Path(os.path.join(config.workdir, config.vdm_name))
    vdm, vdm_params = models.diffusion.VariationalDiffusionModel.from_path_to_model(
        path_to_model=path_to_vdm, norm_dict=norm_dict, checkpoint_step=config.vdm_checkpoint_step)

    path_to_flows = Path(os.path.join(config.workdir, config.flows_name))
    flows, flows_params = nsf.NeuralSplineFlow.from_path_to_model(
        path_to_model=path_to_flows, checkpoint_step=config.flows_checkpoint_step)

    # Create the sampling function based on the normalizing flows
    @partial(jax.vmap, in_axes=(0, None, 0))
    def sample_from_flow(context, n_samples=10_000, key=jax.random.PRNGKey(42)):
        """Helper function to sample from the flow model.
        """
        def sample_fn(flows):
            x_samples = flows.sample(
                num_samples=n_samples, rng=key,
                context=context * jnp.ones((n_samples, 1)))
            return x_samples

        x_samples = nn.apply(sample_fn, flows)(flows_params)
        return x_samples

    # First, use the flows to generate VDM conditioning samples
    logging.info("Generating VDM conditioning samples...")
    flows_conditioning_samples = jax.random.uniform(
        rng, shape=(config.n_samples, num_flows_conditioning),
        minval=jnp.min(flows_conditioning, axis=0),
        maxval=jnp.max(flows_conditioning, axis=0)
    )
    flows_labels_samples = sample_from_flow(
        flows_conditioning_samples, 1, jax.random.split(rng, config.n_samples))
    flows_labels_samples = flows_labels_samples.reshape(-1, flows_labels.shape[1])

    # create the conditioning samples for the VDM
    conditioning = jnp.concatenate(
        [flows_labels_samples, flows_conditioning_samples], axis=1)
    conditioning = conditioning * flows_norm_dict['cond_std'] + flows_norm_dict['cond_mean']

    # Iterate over the entire dataset and start generation
    dset = datasets.make_dataloader(
        (conditioning, ), batch_size=config.batch_size,
        seed=config.seed, shuffle=False, repeat=False)
    dset = create_input_iter(dset)

    vdm_samples = []
    vdm_cond = []
    vdm_mask = []

    for batch in tqdm(dset):
        cond_batch = batch[0][0]
        num_batch = len(cond_batch)

        log_num_subhalos = cond_batch[:, num_flows_labels-1]
        # cond_batch = jnp.delete(cond_batch, num_flows_labels-1, axis=1)
        cond_batch = (cond_batch - norm_dict['cond_mean']) / norm_dict['cond_std']

        # get the total number of particles
        num_subhalos = 10**log_num_subhalos
        num_subhalos = jnp.clip(num_subhalos, 1, config.data.n_particles)
        num_subhalos = jnp.round(num_subhalos).astype(jnp.int32)
        mask_batch = create_mask(num_subhalos, config.data.n_particles)

        vdm_samples_batch = models.eval_utils.generate_samples(
            vdm=vdm,
            params=vdm_params,
            rng=rng,
            n_samples=num_batch,
            n_particles=config.data.n_particles,
            conditioning=cond_batch,
            mask=mask_batch,
            position_encoding=None,
            steps=config.steps,
            norm_dict=norm_dict,
        )

        # denormalize the conditioning vector
        cond_batch = cond_batch * norm_dict['cond_std'] + norm_dict['cond_mean']

        # store data
        vdm_samples.append(vdm_samples_batch)
        vdm_mask.append(mask_batch)
        vdm_cond.append(cond_batch)

    vdm_samples = jnp.concatenate(vdm_samples, axis=0)
    vdm_mask = jnp.concatenate(vdm_mask, axis=0)
    vdm_cond = jnp.concatenate(vdm_cond, axis=0)

    # Save the samples
    if config.output_name is None:
        vdm_base = os.path.basename(config.vdm_name)
        flows_base = os.path.basename(config.flows_name)
        output_name = f'vdm-flows-uniform/{vdm_base}_{flows_base}.npz'
    else:
        output_name = config.output_name
    output_path = os.path.join(config.outdir, output_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logging.info("Saving the generated samples to %s", output_path)
    np.savez(
        output_path, samples=vdm_samples, cond=vdm_cond, mask=vdm_mask,
        flow_samples=conditioning[:, :num_flows_labels],
        flow_cond=conditioning[:, num_flows_labels:],
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
