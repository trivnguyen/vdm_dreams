

from models.diffusion_utils import generate

def generate_samples(
    vdm,
    params,
    rng,
    n_samples,
    n_particles,
    conditioning,
    mask,
    position_encoding,
    steps,
    norm_dict,
):
    generated_samples = generate(
        vdm,
        params,
        rng,
        (n_samples, n_particles),
        conditioning=conditioning,
        mask=mask,
        position_encoding=position_encoding,
        steps=steps,
    )
    generated_samples = generated_samples.mean()
    generated_samples = generated_samples * norm_dict["std"] + norm_dict["mean"]
    return generated_samples
