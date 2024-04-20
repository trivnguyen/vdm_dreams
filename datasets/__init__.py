
import os
import time
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as np
import numpy as vnp
import pandas as pd
import tensorflow as tf
from absl import logging

from . import augmentation

EPS = 1e-7

def make_dataloader(data, batch_size, seed=None, shuffle=True, repeat=True):
    n_train = len(data[0])

    train_ds = tf.data.Dataset.from_tensor_slices(data)
    train_ds = train_ds.cache()
    if repeat:
        train_ds = train_ds.repeat()

    batch_dims = [jax.local_device_count(), batch_size // jax.device_count()]

    for _batch_size in reversed(batch_dims):
        train_ds = train_ds.batch(_batch_size, drop_remainder=False)

    if shuffle:
        train_ds = train_ds.shuffle(n_train, seed=seed)
    return train_ds

def read_dataset(
    dataset_root,
    dataset_name,
    n_features,
    n_particles,
    conditioning_parameters,
    norm_dict=None,
):
    # Read in the dataset
    data = np.load(os.path.join(dataset_root, f"{dataset_name}.npz"))
    x = data['features'][:, :n_particles, :n_features]
    mask = data['mask'][:, :n_particles]
    position_encoding = data['position_encoding'][:, :n_particles]

    # Read in the conditioning features
    conditioning = pd.read_csv(
        os.path.join(dataset_root, f"{dataset_name}_cond.csv"))
    conditioning = np.array(conditioning[conditioning_parameters].values)

    # Standardize per-feature (over datasets and particles)
    if norm_dict is None:
        x_mean = x.mean(axis=(0, 1))
        x_std = x.std(axis=(0, 1))
        cond_mean = conditioning.mean(axis=0)
        cond_std = conditioning.std(axis=0)
        norm_dict = {
            "mean": x_mean,
            "std": x_std,
            "cond_mean": cond_mean,
            "cond_std": cond_std,
        }
    else:
        x_mean = norm_dict["mean"]
        x_std = norm_dict["std"]
        cond_mean = norm_dict.get("cond_mean", 0)
        cond_std = norm_dict.get("cond_std", 1)
    x = (x - x_mean + EPS) / (x_std + EPS)
    conditioning = (conditioning - cond_mean + EPS) / (cond_std + EPS)

    # Finalize
    return x, conditioning, mask, position_encoding, norm_dict

def read_dataloader(
    dataset_root,
    dataset_name,
    n_features,
    n_particles,
    batch_size,
    conditioning_parameters,
    seed=None,
    shuffle=True,
    repeat=True,
    norm_dict=None,
):
    x, conditioning, mask, position_encoding, norm_dict = read_dataset(
        dataset_root,
        dataset_name,
        n_features,
        n_particles,
        conditioning_parameters,
        norm_dict=norm_dict,
    )
    train_ds = make_dataloader(
        (x, conditioning, mask, position_encoding),
        batch_size,
        seed=seed,
        shuffle=shuffle,
        repeat=repeat,
    )
    return train_ds, norm_dict
