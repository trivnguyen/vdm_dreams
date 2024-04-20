
import dataclasses
import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 52

    # Wandb logging
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.entity = None
    wandb.project = "set-diffusion"
    wandb.job_type = "training"
    wandb.name = None
    wandb.log_train = True
    wandb.workdir = "/mnt/ceph/users/tnguyen/dark_camels/point-cloud-diffusion-testing"
    wandb.group = "vdm"

    # Inference
    config.logdir = "/mnt/ceph/users/tnguyen/dark_camels/point-cloud-diffusion-testing"
    config.workdir = "/mnt/ceph/users/tnguyen/dark_camels/point-cloud-diffusion-testing/outputs"
    config.output_name = "test"
    config.vdm_name = ""
    config.steps = 1000
    config.batch_size = 256
    config.n_repeats = 1

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.dataset_root = "/mnt/ceph/users/tnguyen/dark_camels/" \
        "point-cloud-diffusion-datasets/processed_datasets/"
    data.dataset_name = "mw_zooms-wdm-galprop/nmax50-vmaxtilde-pad-v4"
    data.n_particles = 51  # Select the first n_particles particles
    data.n_features = 8  # Select the first n_features features
    data.n_pos_features = 3
    data.n_vel_features = 3
    data.n_mass_features = 2
    data.box_size = 1.0  # Need to know the box size for augmentations
    data.add_augmentations = True
    data.add_rotations = True
    data.add_translations = False
    data.conditioning_parameters = [
        "halo_mvir", "halo_mstar", "inv_wdm_mass", "log_sn1", "log_sn2", "log_agn1"]
    data.flows_conditioning_parameters = [
        "halo_mvir", "halo_mstar", "inv_wdm_mass", "log_sn1", "log_sn2", "log_agn1"]
    data.flows_labels = ["log_num_subhalo", ]
    data.kwargs = {}

    # Flows model for number of particles
    config.flows = flows = ml_collections.ConfigDict()
    flows.flows = "neural_spline_flow"
    flows.n_dim = len(data.flows_labels)
    flows.n_context = len(data.flows_conditioning_parameters)
    flows.hidden_dims = [128, 128]
    flows.n_transforms = 8
    flows.activation = "relu"
    flows.n_bins = 4
    flows.range_min = -5.0
    flows.range_max = 5.0

    # Vartiational diffusion model
    config.vdm = vdm = ml_collections.ConfigDict()
    vdm.gamma_min = -16.0
    vdm.gamma_max = 10.0
    vdm.noise_schedule = "learned_linear"
    vdm.noise_scale = 1e-3
    vdm.noise_scale_mass = 1e-3
    vdm.add_mass_recon_loss = True
    vdm.add_mass_conservation_loss = True
    vdm.mass_decay_rate = 1e3
    vdm.timesteps = 0  # 0 for continuous-time VLB
    vdm.embed_context = True
    vdm.d_context_embedding = 16
    vdm.d_t_embedding = 16  # Timestep embedding dimension
    vdm.n_classes = 0
    vdm.use_encdec = False  # keep this as False

    # Encoder and decoder
    config.encoder = encoder = ml_collections.ConfigDict()
    encoder.d_hidden = 256
    encoder.n_layers = 4
    encoder.d_embedding = 12
    config.decoder = decoder = ml_collections.ConfigDict()
    decoder.d_hidden = 256
    decoder.n_layers = 4

    # Transformer score model
    config.score = score = ml_collections.ConfigDict()
    score.score = "transformer"
    score.induced_attention = False
    score.n_inducing_points = 100
    score.d_model = 128
    score.d_mlp = 256
    score.n_layers = 6
    score.n_heads = 4
    score.concat_conditioning = False
    score.d_conditioning = 128
    score.adanorm = False

    # VDM Training
    config.training = training = ml_collections.ConfigDict()
    training.half_precision = False
    training.batch_size = 64
    training.n_train_steps = 20_001
    training.warmup_steps = 5000
    training.log_every_steps = 500
    training.eval_every_steps = 5000   # removed
    training.save_every_steps = 5000
    training.unconditional_dropout = False  # Set to True to use unconditional dropout (randomly zero out conditioning vectors)
    training.p_uncond = 0.0  # Fraction of conditioning vectors to zero out if unconditional_dropout is True

    # Flow Training
    config.flow_training = flow_training = ml_collections.ConfigDict()
    flow_training.half_precision = False
    flow_training.batch_size = 128
    flow_training.n_train_steps = 20_001
    flow_training.warmup_steps = 5000
    flow_training.log_every_steps = 500
    flow_training.eval_every_steps = 5000
    flow_training.save_every_steps = 5000

    # Optimizer (AdamW)
    config.optim = optim = ml_collections.ConfigDict()
    optim.learning_rate = 3e-4
    optim.weight_decay = 1e-4
    optim.grad_clip = 0.5
    optim.lr_schedule = "cosine"

    return config
