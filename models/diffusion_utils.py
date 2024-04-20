
import jax
import jax.numpy as np
import flax.linen as nn


class NoiseScheduleNet(nn.Module):
    gamma_min: float = -6.0
    gamma_max: float = 7.0
    n_features: int = 1024
    nonlinear: bool = True
    scale_non_linear_init: bool = False

    def setup(self):
        init_bias = self.gamma_max
        init_scale = self.gamma_min - init_bias

        self.l1 = DenseMonotone(
            1, kernel_init=nn.initializers.constant(init_scale),
            bias_init=nn.initializers.constant(init_bias))
        if self.nonlinear:
            if self.scale_non_linear_init:
                stddev_l2 = init_scale
                stddev_l3 = init_scale
            else:
                stddev_l2 = stddev_l3 = 0.01
            self.l2 = DenseMonotone(
                self.n_features, kernel_init=nn.initializers.normal(stddev=stddev_l2))
            self.l3 = DenseMonotone(
                1, kernel_init=nn.initializers.normal(stddev=stddev_l3),
                use_bias=False, decreasing=False)

    @nn.compact
    def __call__(self, t):
        assert np.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1

        if np.isscalar(t) or len(t.shape) == 0:
            t = t * np.ones((1, 1))
        else:
            t = np.reshape(t, (-1, 1))

        h = self.l1(t)
        if self.nonlinear:
            _h = 2.0 * (t - 0.5)  # Scale input to [-1, +1]
            _h = self.l2(_h)
            _h = 2 * (nn.sigmoid(_h) - 0.5)
            _h = self.l3(_h) / self.n_features
            h += _h

        return np.squeeze(h, axis=-1)


class DenseMonotone(nn.Dense):
    """Strictly decreasing Dense layer."""

    decreasing: bool = True

    @nn.compact
    def __call__(self, inputs):
        inputs = np.asarray(inputs, self.dtype)
        kernel = self.param("kernel", self.kernel_init, (inputs.shape[-1], self.features))
        kernel = abs(np.asarray(kernel, self.dtype))  # Use -abs for strictly decreasing
        if self.decreasing:
            kernel = -kernel
        y = jax.lax.dot_general(inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())), precision=self.precision)
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,))
            bias = np.asarray(bias, self.dtype)
            y = y + bias
        return y


class NoiseScheduleScalar(nn.Module):
    gamma_min: float = -6.0
    gamma_max: float = 7.0

    def setup(self):
        init_bias = self.gamma_max
        init_scale = self.gamma_min - self.gamma_max
        self.w = self.param("w", nn.initializers.constant(init_scale), (1,))
        self.b = self.param("b", nn.initializers.constant(init_bias), (1,))

    @nn.compact
    def __call__(self, t):
        # gamma = self.gamma_max - |self.gamma_min - self.gamma_max| * t
        return self.b + -abs(self.w) * t


class NoiseScheduleFixedLinear(nn.Module):
    gamma_min: float = -6.0
    gamma_max: float = 6.0

    @nn.compact
    def __call__(self, t):
        return self.gamma_max + (self.gamma_min - self.gamma_max) * t


def gamma(ts, gamma_min=-6, gamma_max=6):
    return gamma_max + (gamma_min - gamma_max) * ts


def sigma2(gamma):
    return jax.nn.sigmoid(-gamma)


def alpha(gamma):
    return np.sqrt(1 - sigma2(gamma))


def variance_preserving_map(x, gamma, eps):
    a = alpha(gamma)
    var = sigma2(gamma)
    x_shape = x.shape
    x = x.reshape(x.shape[0], -1)
    eps = eps.reshape(eps.shape[0], -1)
    noise_augmented = a * x + np.sqrt(var) * eps
    return noise_augmented.reshape(x_shape)


def get_timestep_embedding(timesteps, embedding_dim: int, dtype=np.float32):
    """Build sinusoidal embeddings (from Fairseq)."""

    assert len(timesteps.shape) == 1
    timesteps *= 1000

    half_dim = embedding_dim // 2
    emb = np.log(10_000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim, dtype=dtype) * -emb)
    emb = timesteps.astype(dtype)[:, None] * emb[None, :]
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # Zero pad
        emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def loss_vdm(params, model, rng, x, conditioning=None, mask=None, position_encoding=None, beta=1.0):
    """
    Compute the loss for a VDM model, sum of diffusion, latent,
    and reconstruction losses, appropriately masked.
    """
    loss_diff, loss_klz, loss_recon, loss_recon_mass = model.apply(
        params, x, conditioning, mask, position_encoding, rngs={"sample": rng})

    if mask is None:
        mask = np.ones(x.shape[:-1])

    loss_batch = (
        ((loss_diff + loss_klz) * mask[:, :, None]).sum((-1, -2)) / beta +
        (loss_recon * mask[:, :, None]).sum((-1, -2))
    ) / mask.sum(-1)
    if loss_recon_mass is not None:
        loss_batch += (loss_recon_mass).sum(-1) / mask.sum(-1)
    return loss_batch.mean()


def generate(vdm, params, rng, shape, conditioning=None, mask=None, position_encoding=None, steps=None):
    """Generate samples from a VDM model."""
    # Generate latents
    rng, spl = jax.random.split(rng)

    # If using a latent projection, use embedding size; otherwise, use feature size
    zt = jax.random.normal(spl, shape + (vdm.encdec_dict["d_embedding"] if vdm.use_encdec else vdm.d_feature,))
    if vdm.timesteps == 0:
        if steps is None:
            raise Exception("Need to specify steps argument for continuous-time VLB")
        else:
            timesteps = steps
    else:
        timesteps = vdm.timesteps

    def body_fn(i, z_t):
        return vdm.apply(
            params, rng, i, timesteps, z_t, conditioning, mask=mask,
            position_encoding=position_encoding, method=vdm.sample_step)
    z0 = jax.lax.fori_loop(lower=0, upper=timesteps, body_fun=body_fn, init_val=zt)

    g0 = vdm.apply(params, 0.0, method=vdm.gammat)
    var0 = sigma2(g0)
    z0_rescaled = z0 / np.sqrt(1.0 - var0)
    return vdm.apply(params, z0_rescaled, conditioning, method=vdm.decode)
