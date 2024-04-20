
import os
from typing import Optional

import jax
import jax.numpy as np
import tensorflow as tf


def random_symmetry_matrix(key):
    # 8 possible sign combinations for reflections
    signs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]
    )

    # 6 permutations for axis swapping
    perms = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]])

    # Randomly select one sign combination and one permutation
    sign = signs[jax.random.randint(key, (), 0, 8)]
    perm = perms[jax.random.randint(key, (), 0, 6)]

    # Combine them to form the random symmetry matrix
    matrix = np.eye(3)[perm] * sign
    return matrix

def augment_with_translations(
    x,
    conditioning,
    mask,
    rng,
    norm_dict,
    n_pos_dim: int = 3,
    box_size: float = 1,
):
    rng, _ = jax.random.split(rng)
    x = x * norm_dict["std"] + norm_dict["mean"]

    # Draw N random translations
    translations = jax.random.uniform(
        rng, minval=-box_size / 2, maxval=box_size / 2, shape=(*x.shape[:2], 3)
    )
    x = x.at[..., :n_pos_dim].set(
        (x[..., :n_pos_dim] + translations[..., None, :]) % box_size
    )
    x = (x - norm_dict["mean"]) / norm_dict["std"]
    return x, conditioning, mask

def augment_with_symmetries(
    x,
    conditioning,
    mask,
    rng,
    norm_dict,
    n_pos_dim: int = 3,
    n_vel_dim: int = 3,
    box_size: float = 1,
):
    rng, _ = jax.random.split(rng)
    # Rotations and reflections that respect boundary conditions
    matrix = random_symmetry_matrix(rng)
    x = x.at[..., :n_pos_dim].set(np.dot(x[..., :n_pos_dim], matrix.T))
    if n_vel_dim > 0:
        # Rotate velocities too
        x = x.at[..., n_pos_dim : n_pos_dim + n_vel_dim].set(
            np.dot(x[..., n_pos_dim : n_pos_dim + n_vel_dim], matrix.T)
        )
    return x, conditioning, mask


def augment_data(
    x,
    conditioning,
    mask,
    rng,
    norm_dict,
    rotations: bool = True,
    translations: bool = True,
    n_pos_dim: int = 3,
    n_vel_dim: int = 3,
    box_size: float = 1,
):
    if rotations:
        x, conditioning, mask = augment_with_symmetries(
            x=x,
            mask=mask,
            conditioning=conditioning,
            rng=rng,
            norm_dict=norm_dict,
            n_pos_dim=n_pos_dim,
            n_vel_dim=n_vel_dim,
            box_size=box_size,
        )
    if translations:
        x, conditioning, mask = augment_with_translations(
            x=x,
            mask=mask,
            conditioning=conditioning,
            rng=rng,
            norm_dict=norm_dict,
            n_pos_dim=n_pos_dim,
            box_size=box_size,
        )
    return x, conditioning, mask
