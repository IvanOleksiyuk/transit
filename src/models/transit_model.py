import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

# Some standard libraries
import numpy as np
import matplotlib.pyplot as plt
import PIL
import copy
from functools import partial
from typing import Any, Mapping
import pickle
from types import SimpleNamespace

# torch related
from torch import nn
import wandb
import torch 
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch.nn.functional import normalize, mse_loss, cosine_similarity
import torch.distributed as dist

# Local sorce
import transit.src.models.distance_correlation as dcor
from transit.src.models.pearson_correlation import PearsonCorrelation

# Libraries
from transit.mattstools.mattstools.simple_transformers import TransformerEncoder, FullEncoder, TransformerVectorEncoder
from transit.mltools.modules import IterativeNormLayer
from transit.mltools.torch_utils import get_sched, get_loss_fn
from transit.mltools.mlp import MLP
from transit.src.models.dequantization import SelectiveDequantizationTransform

def to_np(inpt) -> np.ndarray:
    """More consicse way of doing all the necc steps to convert a pytorch
    tensor to numpy array.

    - Includes gradient deletion, and device migration
    """
    if isinstance(inpt, (tuple, list)):
        return type(inpt)(to_np(x) for x in inpt)
    if inpt.dtype == torch.bfloat16:  # Numpy conversions don't support bfloat16s
        inpt = inpt.half()
    return inpt.detach().cpu().numpy()

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def pairwise_distances(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pairwise Euclidean distances between rows of a and b."""
    a2 = (a * a).sum(dim=-1, keepdim=True)
    b2 = (b * b).sum(dim=-1, keepdim=True).T
    ab = a @ b.T
    dist2 = (a2 + b2 - 2 * ab).clamp(min=0)
    return torch.sqrt(dist2 + eps)


def energy_distance(
    p_samples: torch.Tensor,
    q_samples: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Unbiased-style energy distance estimator between two sample sets."""
    cross = pairwise_distances(p_samples, q_samples, eps=eps).mean()
    intra_p = pairwise_distances(p_samples, p_samples, eps=eps).mean()
    intra_q = pairwise_distances(q_samples, q_samples, eps=eps).mean()
    return 2 * cross - intra_p - intra_q


class EnergyDistanceLoss(nn.Module):
    """Energy distance loss between true and generated batches."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, true_batch: torch.Tensor, generated_batch: torch.Tensor) -> torch.Tensor:
        return energy_distance(true_batch, generated_batch, eps=self.eps)

class TRANSIT(LightningModule):
    
    def __init__(
        self,
        *,
        inpt_dim: int,

        latent_dim: int,
        latent_norm: bool,
        optimizer: partial,
        scheduler: Mapping,
        encoder_cfg,
        decoder_cfg,
        network_type = "partial_context",
        var_group_list: list = None,
        loss_cfg: Mapping = None,
        transport_loss: str = "adversarial",
        mmd_cfg: Mapping | None = None,
        energy_cfg: Mapping | None = None,
        use_m_encodig = True,
        input_noise_cfg=None,
        reverse_pass_mode=None,
        seed=42,
        valid_plots = True,
        adversarial_cfg = None,
        discriminator_latent_cfg = None,
        add_standardizing_layer = False,
        afterglow_epoch = np.inf,
        second_input_mask = False,
        total_skip = False,
        input_type = "default",
        valid_plot_freq = 1,
        dequantization_cfg = None,
        true_trajectory_pickle_file = None, # Path to a pickle file containing a function that provides true trajectories
        do_switch_off_adversary_in_case_of_instability = False,
        ema_cfg: Mapping | None = None,
        
    ) -> None:
        """
        Args:
            inpt_dim: Number of edge, node and high level features
            normaliser_config: Config for the IterativeNormLayer
            optimizer: Partially initialised optimiser
            scheduler: How the sceduler should be used
        """
        
        super().__init__()
        # TODO need to preprocess the data properly!
        torch.manual_seed(seed)
        self.valid_plot_freq = valid_plot_freq
        self.input_type=input_type
        self.afterglow_epoch = afterglow_epoch
        self.transport_loss_mode = transport_loss
        if ema_cfg is None:
            self.ema_cfg = SimpleNamespace()
        elif isinstance(ema_cfg, Mapping):
            self.ema_cfg = SimpleNamespace(**dict(ema_cfg))
        else:
            self.ema_cfg = ema_cfg

        self.use_ema = bool(self._cfg_get(self.ema_cfg, "enabled", False))
        self.ema_decay = float(self._cfg_get(self.ema_cfg, "decay", 0.999))
        self.ema_update_every = max(1, int(self._cfg_get(self.ema_cfg, "update_every", 1)))
        self.ema_start_step = max(0, int(self._cfg_get(self.ema_cfg, "start_step", 0)))
        self.ema_use_in_eval = bool(self._cfg_get(self.ema_cfg, "use_in_eval", True))
        self.ema_include_encoder2 = bool(self._cfg_get(self.ema_cfg, "include_encoder2", True))
        self._ema_shadow = {}
        self._ema_backup = None
        self._ema_num_updates = 0
        self._ema_eval_active = False
        self._ema_should_update_this_batch = False
        if mmd_cfg is None:
            self.mmd_cfg = SimpleNamespace()
        elif isinstance(mmd_cfg, Mapping):
            self.mmd_cfg = SimpleNamespace(**dict(mmd_cfg))
        else:
            self.mmd_cfg = mmd_cfg

        if energy_cfg is None:
            self.energy_cfg = SimpleNamespace()
        elif isinstance(energy_cfg, Mapping):
            self.energy_cfg = SimpleNamespace(**dict(energy_cfg))
        else:
            self.energy_cfg = energy_cfg

        self.energy_distance_loss = EnergyDistanceLoss(
            eps=float(self._cfg_get(self.energy_cfg, "eps", 1e-8))
        )

        if self.transport_loss_mode in ["mmd", "energy"]:
            self.adversarial = False
            self.automatic_optimization = True
            self.adversarial_cfg = None
        elif adversarial_cfg is not None:
            if hasattr(adversarial_cfg, "mode"):
                self.adversarial = adversarial_cfg.mode
            else:
                self.adversarial = "default"
            self.automatic_optimization = False
            self.adversarial_cfg = adversarial_cfg
            self.disc_input_noise_std = getattr(adversarial_cfg, "disc_input_noise_std", 0.0)

            if not hasattr(adversarial_cfg, "g_loss_weight_in_warmup"):
                setattr(adversarial_cfg, "g_loss_weight_in_warmup", True)
            if not hasattr(adversarial_cfg, "train_dis_in_warmup"):
                setattr(adversarial_cfg, "train_dis_in_warmup", False)
            if not hasattr(adversarial_cfg, "loss_function"):
                setattr(adversarial_cfg, "loss_function", "binary_cross_entropy")
            if not hasattr(adversarial_cfg, "gradient_clip_val"):
                self.gradient_clip_val = 5
            else:
                self.gradient_clip_val = adversarial_cfg.gradient_clip_val
            if not hasattr(adversarial_cfg, "label_smoothing_eps"):
                setattr(adversarial_cfg, "label_smoothing_eps", 0)
        else:
            self.adversarial = False
            self.disc_input_noise_std = 0.0
        if true_trajectory_pickle_file is not None:
            self.true_trajectory_function = pickle.load(open(true_trajectory_pickle_file, "rb"))
        else:
            self.true_trajectory_function = None
        self.use_m_encodig = use_m_encodig
        self.loss_cfg = loss_cfg
        self.valid_plots = valid_plots
        self.save_hyperparameters(logger=False)
        self.network_type = network_type
        print("SELECTED NETWORK TYPE: ", network_type)
        self.latent_norm_enc1 = latent_norm
        self.latent_norm_enc2 = latent_norm
        self.total_skip = total_skip
        self.do_switch_off_adversary_in_case_of_instability = do_switch_off_adversary_in_case_of_instability
        # Initialise the networks
        if hasattr(inpt_dim[0], "__getitem__"):
            x_dim = inpt_dim[0][0]
            context_dim = inpt_dim[1][0]
        else:
            x_dim = inpt_dim[0]
            context_dim = inpt_dim[1]
        self.second_input_mask = second_input_mask #By default
        
        if network_type == "partial_context":
            self.decoder_out_m = False
            self.encoder1 = encoder_cfg(inpt_dim=x_dim, ctxt_dim=context_dim, outp_dim=latent_dim)
            self.decoder = decoder_cfg(inpt_dim=latent_dim, ctxt_dim=context_dim, outp_dim=x_dim)
            self.style_injection_cond = True
            if self.adversarial:
                if hasattr(adversarial_cfg, "discriminator"):
                    self.discriminator = adversarial_cfg.discriminator(inpt_dim=latent_dim, ctxt_dim=context_dim)
                    self.use_disc_lat = True
                else:
                    self.use_disc_lat = False
                if hasattr(adversarial_cfg, "discriminator2"):
                    if adversarial_cfg.get("reco_both_m_mhat", False):
                        context_dim_dis2=context_dim*2
                        self.use_disc_reco_doublecond=True
                    else:
                        context_dim_dis2=context_dim
                        self.use_disc_reco_doublecond=False
                    self.discriminator2 = adversarial_cfg.discriminator2(inpt_dim=x_dim, ctxt_dim=context_dim_dis2)
                    self.use_disc_reco = True
                else:
                    self.use_disc_reco = False
            self.encoder2 = lambda x: x
        else:
            assert False, "Unknown network type"

        self.add_standardizing_layer = add_standardizing_layer
        if add_standardizing_layer:
            self.std_layer_x = IterativeNormLayer(x_dim)
            self.std_layer_ctxt = IterativeNormLayer(context_dim)
            
        if dequantization_cfg is not None:
            self.do_dequantization = True
            self.dequantization_layer = SelectiveDequantizationTransform(
                dequantization_cfg["discrete_indices"], 
                dequantization_cfg["discrete_shift"], 
                dequantization_cfg["discrete_scale"]
            )
            print("Dequantization layer added")
        else:
            self.do_dequantization = False
            self.dequantization_layer = None
        
        self.var_group_list = var_group_list
        self.reverse_pass_mode = reverse_pass_mode
        self.input_noise_cfg = input_noise_cfg
        
        if hasattr(loss_cfg, "DisCO_loss_cfg"):
            self.DisCO_loss = dcor.DistanceCorrelation()
        if hasattr(loss_cfg, "pearson_loss_cfg"):
            self.pearson_loss = PearsonCorrelation()

        # For more stable checks in the shared step
        expected_attrs = ["reco", 
                "reco_l1",
                "consistency_x",
                "consistency_x_l1",
                "consistency_normalised_x",
                "consistency_xx", 
                "consistency_cont", 
                "latent_variance_cfg", 
                "l1_reg", 
                "DisCO_loss_cfg", 
                "pearson_loss_cfg", 
                "attractive", 
                "repulsive", 
                "second_derivative_smoothness",
                "third_derivative_smoothness",
                "noised_reco",
                "consistency_noised",
                "latent_ed_gaussian",
                "consistency_noise",
                "projected_context_ed_gaussian",
                "latent_mean_cfg",
                "latent_covariance_cfg",
                "decoder_jacobian_cfg",
                "parallelity_loss_cfg",
                "rot_loss_cfg"]
        for attr in list(self.loss_cfg.keys()):
            if attr in expected_attrs:
                setattr(self.loss_cfg, attr, getattr(self.loss_cfg, attr, None))
            else:
                assert False, f"Unexpected loss config attribute: {attr}"
        for attr in expected_attrs:
            if not hasattr(self.loss_cfg, attr):
                setattr(self.loss_cfg, attr, None)

        if hasattr(self.loss_cfg.DisCO_loss_cfg, "w"):
            if not hasattr(self.loss_cfg.DisCO_loss_cfg, "mode"):
                self.loss_cfg.DisCO_loss_cfg.mode = "e1_vs_e2"
        self.dis_steps_per_gen = 0

        if self.use_ema:
            self._init_ema_state()

    def encode_content(self, x_inp, m_pair, mask=None):
        if not self.use_m_encodig:
            en = self.encoder1(x_inp, mask=mask)
        if self.style_injection_cond:
            if self.second_input_mask:
                en = self.encoder1(x_inp, mask=mask, ctxt=m_pair)
            else:
                en = self.encoder1(x_inp, ctxt=m_pair)
        else:
            en = self.encoder1(torch.cat([x_inp, m_pair], dim=1))
        
        if self.latent_norm_enc1:
            return normalize(en)
        else:
            return en

    def get_latent(self, x_inp, m_pair, mask=None):
        
        #Make sure the inputs are in the right shape
        m_pair=m_pair.reshape([x_inp.shape[0], -1])
        
        if self.do_dequantization:
            x_inp = self.dequantization_layer(x_inp)
        
        if self.add_standardizing_layer:
            x_inp = self.std_layer_x(x_inp, mask=mask)
            m_pair = self.std_layer_ctxt(m_pair)
        
        return self.encode_content(x_inp, m_pair, mask=mask)
    
    def inn_transport(self, xs, ms_from, ms_to, mask=None):
        # xs and ms are already stadardized and dequantized if those layers are present, so we can just encode, decode and return the reconstruction
        content = self.encode_content(xs, ms_from, mask=mask)
        style = self.encode_style(ms_to)
        recon = self.decode(content, style)
        return recon
    
    def inn_second_latent(self, xs, ms_from, ms_to, mask=None):
        # xs and ms are already stadardized and dequantized if those layers are present, so we can just encode, decode and return the reconstruction
        content = self.encode_content(xs, ms_from, mask=mask)
        style = self.encode_style(ms_to)
        recon = self.decode(content, style)
        content_n = self.encode_content(recon, ms_to, mask=mask)
        style_n = self.encode_style(ms_to)
        return content_n, style_n
    
    def get_second_latent(self, x_inp, m_from, m_to, mask=None):
        # Function fro outside use to get the second latent encoding for a given input and two different contexts. Useful for transport in the latent space.
        m_from = m_from.reshape([x_inp.shape[0], -1])
        m_to = m_to.reshape([x_inp.shape[0], -1])
        if self.do_dequantization:
            x_inp = self.dequantization_layer(x_inp)
        if self.add_standardizing_layer:
            x_inp = self.std_layer_x(x_inp, mask=mask)
            m_from = self.std_layer_ctxt(m_from)
            m_to = self.std_layer_ctxt(m_to)
        return self.inn_second_latent(x_inp, m_from, m_to, mask=mask)[0]

    def encode_style(self, m):
        en = self.encoder2(m)
        if self.latent_norm_enc2:
            return normalize(en)
        else:
            return en

    def decode(self, content, style, mask=None):
        if self.style_injection_cond:
            if self.second_input_mask:
                return self.decoder(content, mask=mask, ctxt=style)
            else:
                return self.decoder(content, ctxt=style)
        else:
            return self.decoder(torch.cat([content, style], dim=1))

    @staticmethod
    def _cfg_get(cfg, key, default):
        if cfg is None:
            return default
        if isinstance(cfg, Mapping):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _add_disc_noise(self, *tensors):
        if self.disc_input_noise_std <= 0:
            return tensors if len(tensors) > 1 else tensors[0]
        noisy = [t + torch.randn_like(t) * self.disc_input_noise_std for t in tensors]
        return noisy if len(noisy) > 1 else noisy[0]

    def _ema_named_parameters(self):
        modules = [("encoder1", self.encoder1), ("decoder", self.decoder)]
        if self.ema_include_encoder2 and hasattr(self.encoder2, "named_parameters"):
            modules.append(("encoder2", self.encoder2))

        for module_name, module in modules:
            if not hasattr(module, "named_parameters"):
                continue
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    yield f"{module_name}.{param_name}", param

    def _init_ema_state(self):
        self._ema_shadow = {
            name: param.detach().float().cpu().clone()
            for name, param in self._ema_named_parameters()
        }

    @torch.no_grad()
    def _ema_update(self):
        if not self.use_ema:
            return

        if not self._ema_shadow:
            self._init_ema_state()

        one_minus_decay = 1.0 - self.ema_decay
        for name, param in self._ema_named_parameters():
            shadow = self._ema_shadow.get(name, None)
            target_device = shadow.device if shadow is not None else torch.device("cpu")
            current = param.detach().to(device=target_device, dtype=torch.float32)
            if name not in self._ema_shadow:
                self._ema_shadow[name] = current.clone()
                continue
            self._ema_shadow[name].mul_(self.ema_decay).add_(current, alpha=one_minus_decay)
        self._ema_num_updates += 1

    @torch.no_grad()
    def _ema_apply_eval_weights(self):
        if not (self.use_ema and self.ema_use_in_eval):
            return
        if self._ema_eval_active:
            return
        if not self._ema_shadow:
            return

        self._ema_backup = {}
        for name, param in self._ema_named_parameters():
            if name not in self._ema_shadow:
                continue
            self._ema_backup[name] = param.detach().clone()
            param.copy_(self._ema_shadow[name].to(device=param.device, dtype=param.dtype))
        self._ema_eval_active = True

    @torch.no_grad()
    def _ema_restore_train_weights(self):
        if not self._ema_eval_active:
            return
        if self._ema_backup is None:
            self._ema_eval_active = False
            return

        for name, param in self._ema_named_parameters():
            if name in self._ema_backup:
                param.copy_(self._ema_backup[name])
        self._ema_backup = None
        self._ema_eval_active = False

    def _compute_mmd(self, x_real, m_real, x_fake, m_fake):
        # Product of RBF kernels on features and mass; multi-bandwidth for robustness.
        eps = 1e-12

        def _bandwidth_list(base_sigma, multipliers, device, dtype):
            mult = torch.as_tensor(multipliers, device=device, dtype=dtype)
            return base_sigma * mult

        def _rbf_kernel(d2, sigmas):
            kernels = [torch.exp(-d2 / (2.0 * (sigma ** 2) + eps)) for sigma in sigmas]
            return torch.stack(kernels, dim=0).mean(dim=0)

        def _pairwise_d2(a, b):
            return torch.cdist(a, b, p=2) ** 2

        sigma_x_base = self._cfg_get(self.mmd_cfg, "sigma_x", None)
        sigma_m_base = self._cfg_get(self.mmd_cfg, "sigma_m", None)
        sig_mult_x = self._cfg_get(self.mmd_cfg, "sigma_x_multipliers", [0.5, 1.0, 2.0])
        sig_mult_m = self._cfg_get(self.mmd_cfg, "sigma_m_multipliers", [0.5, 1.0, 2.0])

        if sigma_x_base is None:
            d_real = torch.cdist(x_real, x_real, p=2)
            sigma_x_base = torch.median(d_real.detach()) + eps
        else:
            sigma_x_base = torch.tensor(float(sigma_x_base), device=x_real.device, dtype=x_real.dtype)

        if sigma_m_base is None:
            d_m_real = torch.cdist(m_real, m_real, p=2)
            sigma_m_base = torch.median(d_m_real.detach()) + eps
        else:
            sigma_m_base = torch.tensor(float(sigma_m_base), device=m_real.device, dtype=m_real.dtype)

        sigma_x_list = _bandwidth_list(sigma_x_base, sig_mult_x, x_real.device, x_real.dtype)
        sigma_m_list = _bandwidth_list(sigma_m_base, sig_mult_m, m_real.device, m_real.dtype)

        k_x_rr = _rbf_kernel(_pairwise_d2(x_real, x_real), sigma_x_list)
        k_x_ff = _rbf_kernel(_pairwise_d2(x_fake, x_fake), sigma_x_list)
        k_x_rf = _rbf_kernel(_pairwise_d2(x_real, x_fake), sigma_x_list)

        k_m_rr = _rbf_kernel(_pairwise_d2(m_real, m_real), sigma_m_list)
        k_m_ff = _rbf_kernel(_pairwise_d2(m_fake, m_fake), sigma_m_list)
        k_m_rf = _rbf_kernel(_pairwise_d2(m_real, m_fake), sigma_m_list)

        K_rr = k_x_rr * k_m_rr
        K_ff = k_x_ff * k_m_ff
        K_rf = k_x_rf * k_m_rf

        n_r = x_real.shape[0]
        n_f = x_fake.shape[0]
        if n_r < 2 or n_f < 2:
            return torch.tensor(0.0, device=x_real.device, dtype=x_real.dtype)

        def _off_diag_mean(mat):
            diag_sum = torch.diagonal(mat).sum()
            return (mat.sum() - diag_sum) / (mat.numel() - mat.shape[0])

        mmd = _off_diag_mean(K_rr) + _off_diag_mean(K_ff) - 2.0 * K_rf.mean()
        return mmd

    def _compute_energy_distance(self, x_real, m_real, x_fake, m_fake):
        x_real_flat = x_real.reshape(x_real.shape[0], -1)
        m_real_flat = m_real.reshape(m_real.shape[0], -1)
        x_fake_flat = x_fake.reshape(x_fake.shape[0], -1)
        m_fake_flat = m_fake.reshape(m_fake.shape[0], -1)

        true_batch = torch.cat([x_real_flat, m_real_flat], dim=-1)
        generated_batch = torch.cat([x_fake_flat, m_fake_flat], dim=-1)
        return self.energy_distance_loss(true_batch, generated_batch)

    def _compute_projected_context_ed_gaussian(self, content, num_directions: int = 1, eps: float = 1e-8):
        """Energy distance to a unit Gaussian in 1D random projections of content."""
        content_flat = content.reshape(content.shape[0], -1)

        n_dims = content_flat.shape[1]
        num_directions = max(int(num_directions), 1)
        if n_dims == 1:
            z = torch.randn_like(content_flat)
            return energy_distance(content_flat, z, eps=eps)

        directions = torch.randn(
            num_directions,
            n_dims,
            device=content_flat.device,
            dtype=content_flat.dtype,
        )
        directions = F.normalize(directions, dim=1)

        proj_real = content_flat @ directions.T
        proj_gauss = torch.randn_like(proj_real)
        losses = [
            energy_distance(proj_real[:, i:i+1], proj_gauss[:, i:i+1], eps=eps)
            for i in range(num_directions)
        ]
        return torch.stack(losses).mean()

    def disc_lat(self, e1, e2):
        if self.style_injection_cond:
            return self.discriminator(e1, ctxt=e2)
        else:
            return self.discriminator(torch.cat([e1, e2], dim=1))

    def disc_reco(self, w1, w2):
        if self.style_injection_cond:
            return self.discriminator2(w1, ctxt=w2)
        else:
            return self.discriminator2(torch.cat([w1, w2], dim=1))

    def interprete_input(self, sample, phase="train"):
        if self.second_input_mask:
            x_inp, mask, m_pair, m_add = sample
        elif self.input_type=="sky_train":
            if phase=="train":
                x_inp, m_pair, m_add = sample[0][:, :-1], sample[0][:, -1:], sample[0][:, -1:]
                mask = None
            elif phase=="generate":
                x_inp, m_pair, m_add = sample[0][:, :-1], sample[0][:, -1:], sample[2]
                mask = None                
        else:
            x_inp, m_pair, m_add = sample
            mask = None
        return x_inp, mask, m_pair, m_add

    def parralelity_loss(self, encoder, decoder, x_inp, m_pair, eps_scale=1e-2):
        eps = torch.randn_like(x_inp)
        eps = eps / (eps.norm(dim=1, keepdim=True) + 1e-8)
        x_bar = x_inp + eps*eps_scale
        d_x = (x_bar - x_inp).reshape(x_inp.shape[0], -1)
        if self.style_injection_cond:
            z = encoder(x_inp, ctxt=m_pair)
            z_bar = encoder(x_bar, ctxt=m_pair)
            m_pair_shuffled = m_pair[torch.randperm(m_pair.shape[0])]
            x_rec = decoder(z, ctxt=m_pair_shuffled)
            x_bar_rec = decoder(z_bar, ctxt=m_pair_shuffled)
        else:
            z = encoder(torch.cat([x_inp, m_pair], dim=1))
            z_bar = encoder(torch.cat([x_bar, m_pair], dim=1))
            x_rec = decoder(z)
            x_bar_rec = decoder(z_bar)
        dx_rec = (x_bar_rec - x_rec).reshape(x_inp.shape[0], -1)
        return (dx_rec-d_x).norm(dim=1).mean()/eps_scale

    def rotationality_loss(self, encoder, decoder, x_inp, m_pair, eps_scale=1e-2):
        B = x_inp.shape[0]

        u = torch.randn_like(x_inp)
        u = u / (u.norm(dim=1, keepdim=True) + 1e-8)

        v = torch.randn_like(x_inp)
        v = v - (u * v).sum(dim=1, keepdim=True) * u
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)

        x_u = x_inp + eps_scale * u
        x_v = x_inp + eps_scale * v

        if self.style_injection_cond:
            m_pair_shuffled = m_pair[torch.randperm(m_pair.shape[0])]

            z = encoder(x_inp, ctxt=m_pair)
            z_u = encoder(x_u, ctxt=m_pair)
            z_v = encoder(x_v, ctxt=m_pair)

            x_rec = decoder(z, ctxt=m_pair_shuffled)
            x_u_rec = decoder(z_u, ctxt=m_pair_shuffled)
            x_v_rec = decoder(z_v, ctxt=m_pair_shuffled)
        else:
            z = encoder(torch.cat([x_inp, m_pair], dim=1))
            z_u = encoder(torch.cat([x_u, m_pair], dim=1))
            z_v = encoder(torch.cat([x_v, m_pair], dim=1))

            x_rec = decoder(z)
            x_u_rec = decoder(z_u)
            x_v_rec = decoder(z_v)

        du_rec = (x_u_rec - x_rec).reshape(B, -1)
        dv_rec = (x_v_rec - x_rec).reshape(B, -1)

        u_flat = u.reshape(B, -1)
        v_flat = v.reshape(B, -1)

        skew = (u_flat * dv_rec).sum(dim=1) - (v_flat * du_rec).sum(dim=1)

        return (skew.abs() / eps_scale).mean()


    def decoder_smoothness_penalty(self, decoder, z, context, eps_scale=1e-2):
            """
            Finite-difference approximation of decoder Jacobian norm.
            z: [B, Dz]
            """
            eps = torch.randn_like(z)
            eps = eps / (eps.norm(dim=1, keepdim=True) + 1e-8)
            eps = eps * eps_scale

            x1 = decoder(z, context)
            x2 = decoder(z + eps, context)

            dx = (x2 - x1).reshape(z.shape[0], -1)
            dz = eps.reshape(z.shape[0], -1)

            return ((dx.norm(dim=1) / (dz.norm(dim=1) + 1e-8)) ** 2).mean()

    def _shared_step(self, sample: tuple, _batch_index = None, step_type="none") -> torch.Tensor:
        self.switch_off_adversary_in_case_of_instability = True
        self.log(f"{step_type}_debug/global_step", self.global_step)
        batch_size=sample[0].shape[0]
        
        x_inp, mask, m_pair, m_add = self.interprete_input(sample, phase="train")

        if self.input_noise_cfg is not None and self.training:
            noise_std = self.input_noise_cfg.get("noise_std", 0.1)
            x_inp = x_inp + torch.randn_like(x_inp)*noise_std

        #Make sure the inputs are in the right shape
        m_pair=m_pair.reshape([x_inp.shape[0], -1])
        m_add=m_add.reshape([x_inp.shape[0], -1])
        
        # DELETE THIS, just for debugging to check the mass values are in the right range and not all smaller than 0.5 for example
        self.log(f"{step_type}_debug/batch_size", batch_size)
        self.log(f"{step_type}_debug/m_pair_larger0.5", ( (m_pair>0.5).sum()/m_pair.numel() ).item())
        self.log(f"{step_type}_debug/m_add_larger0.5", ( (m_add>0.5).sum()/m_add.numel() ).item())

        if self.do_dequantization:
            x_inp = self.dequantization_layer(x_inp)
        
        if self.add_standardizing_layer:
            x_inp = self.std_layer_x(x_inp, mask=mask)
            m_pair = self.std_layer_ctxt(m_pair)
            m_add = self.std_layer_ctxt(m_add)
        
        content = self.encode_content(x_inp, m_pair, mask=mask)
        style = self.encode_style(m_pair)
        
        if self.total_skip:
            recon = x_inp*self.total_skip + self.decode(content, style)
        else:
            recon = self.decode(content, style)
        
        # Reverse pass
        rpm = torch.randperm(batch_size)
        if self.reverse_pass_mode == "additional_input":
            style_p = self.encode_style(m_add)[rpm]
        elif self.reverse_pass_mode == "additional_input_noise":
            style_p = self.encode_style(m_add)[rpm]
            max_p = style_p.max()
            min_p = style_p.min()
            style_p_n = style_p + torch.randn_like(m_add)*0.05
            style_p_n[style_p_n>max_p] = style_p[style_p_n>max_p]
            style_p_n[style_p_n<min_p] = style_p[style_p_n<min_p]
            style_p = style_p_n
        else:
            style_p = style[rpm]

        recon_p = self.decode(content, style_p)
        if self.decoder_out_m:
            x_n = recon_p[:, :x_inp.shape[1]]
            m_n = recon_p[:, x_inp.shape[1]:]
        else:
            x_n = recon_p
            if self.reverse_pass_mode == "additional_input":
                m_n = m_add[rpm]
            else:
                m_n = m_pair[rpm]

        content_n = self.encode_content(x_n, m_n, mask=mask)
        style_n = self.encode_style(m_n)

        #### Losses
        total_loss = 0
        self.switch_off_adversary_in_case_of_instability = False

        # Reconstruction loss
        if self.use_m_encodig and self.decoder_out_m:
            loss_reco = mse_loss(recon, torch.cat([x_inp, m_pair], dim=1)).mean()
        else:
            loss_reco = mse_loss(recon, x_inp).mean()
        total_loss += loss_reco*self.loss_cfg.reco.w
        self.log(f"{step_type}/loss_reco", loss_reco)

        # Reco L1
        if self.loss_cfg.reco_l1 is not None:
            if self.use_m_encodig and self.decoder_out_m:
                loss_reco_l1 = F.l1_loss(recon, torch.cat([x_inp, m_pair], dim=1)).mean()
            else:
                loss_reco_l1 = F.l1_loss(recon, x_inp).mean()
            total_loss += loss_reco_l1*self.loss_cfg.reco_l1.w
            self.log(f"{step_type}/loss_reco_l1", loss_reco_l1)

        if self.do_switch_off_adversary_in_case_of_instability:
            if loss_reco > 0.0001:
                self.switch_off_adversary_in_case_of_instability = True

        # Second derivative smoothness
        if self.loss_cfg.second_derivative_smoothness is not None:
            e2_p_pl = self.encode_style(m_add + self.loss_cfg.second_derivative_smoothness.step)[rpm]
            e2_p_mi = self.encode_style(m_add - self.loss_cfg.second_derivative_smoothness.step)[rpm]
            recon_p_pl = self.decode(content, e2_p_pl)
            recon_p_mi = self.decode(content, e2_p_mi)
            loss_sec_der = (recon_p_pl+recon_p_mi-2*recon_p)/(self.loss_cfg.second_derivative_smoothness.step**2)
            loss_sec_der = loss_sec_der.abs().mean()
            self.log(f"{step_type}/loss_sec_der_smooth", loss_sec_der)
            if self.loss_cfg.second_derivative_smoothness.w is not None:
                if isinstance(self.loss_cfg.second_derivative_smoothness.w, float) or isinstance(self.loss_cfg.second_derivative_smoothness.w, int):
                    total_loss += loss_sec_der*self.loss_cfg.second_derivative_smoothness.w
                else:
                    w = self.loss_cfg.second_derivative_smoothness.w(self.global_step)
                    total_loss += loss_sec_der*w
                    self.log(f"{step_type}_debug/second_derivative_smoothnessw", w)

        # Third derivative smoothness (5-point central finite difference)
        if self.loss_cfg.third_derivative_smoothness is not None:
            step_val = self.loss_cfg.third_derivative_smoothness.step
            if step_val == 0:
                raise ValueError("third_derivative_smoothness.step must be non-zero")

            e2_m2 = self.encode_style(m_add - 2 * step_val)[rpm]
            e2_m1 = self.encode_style(m_add - step_val)[rpm]
            e2_p1 = self.encode_style(m_add + step_val)[rpm]
            e2_p2 = self.encode_style(m_add + 2 * step_val)[rpm]

            recon_m2 = self.decode(content, e2_m2)
            recon_m1 = self.decode(content, e2_m1)
            recon_p1 = self.decode(content, e2_p1)
            recon_p2 = self.decode(content, e2_p2)

            loss_third_der = (recon_m2 - 2 * recon_m1 + 2 * recon_p1 - recon_p2) / (2 * (step_val ** 3))
            loss_third_der = loss_third_der.abs().mean()
            self.log(f"{step_type}/loss_third_der_smooth", loss_third_der)

            if self.loss_cfg.third_derivative_smoothness.w is not None:
                if isinstance(self.loss_cfg.third_derivative_smoothness.w, (float, int)):
                    total_loss += loss_third_der * self.loss_cfg.third_derivative_smoothness.w
                else:
                    w = self.loss_cfg.third_derivative_smoothness.w(self.global_step)
                    total_loss += loss_third_der * w
                    self.log(f"{step_type}_debug/third_derivative_smoothnessw", w)

        # Consistency losses 
        if self.loss_cfg.consistency_x is not None:
            loss_back_vec = mse_loss(content, content_n).mean()
            self.log(f"{step_type}/loss_back_vec", loss_back_vec)
            if self.loss_cfg.consistency_x.w is not None:
                if isinstance(self.loss_cfg.consistency_x.w, float) or isinstance(self.loss_cfg.consistency_x.w, int):
                    total_loss += loss_back_vec*self.loss_cfg.consistency_x.w
                else:
                    total_loss += loss_back_vec*self.loss_cfg.consistency_x.w(self.global_step)
            if self.do_switch_off_adversary_in_case_of_instability:
                if loss_back_vec > 0.0001:
                    self.switch_off_adversary_in_case_of_instability = True

        if self.loss_cfg.decoder_jacobian_cfg is not None:
            loss_decoder_jacobian = self.decoder_smoothness_penalty(
                self.decoder,
                content,
                context=style,
                eps_scale=self.loss_cfg.decoder_jacobian_cfg.eps_scale,
            )

            if self.loss_cfg.decoder_jacobian_cfg.w is not None:
                total_loss += loss_decoder_jacobian * self.loss_cfg.decoder_jacobian_cfg.w

            self.log(f"{step_type}/decoder_jacobian_regularization", loss_decoder_jacobian)

        if self.loss_cfg.parallelity_loss_cfg is not None:
            loss_parallelity = self.parralelity_loss(self.encoder1, self.decoder, x_inp, m_pair, eps_scale=self.loss_cfg.parallelity_loss_cfg.eps_scale)

            if self.loss_cfg.parallelity_loss_cfg.w is not None:
                total_loss += loss_parallelity * self.loss_cfg.parallelity_loss_cfg.w

            self.log(f"{step_type}/parallelity_loss", loss_parallelity)

        if self.loss_cfg.rot_loss_cfg is not None:
            loss_rotationality = self.rotationality_loss(self.encoder1, self.decoder, x_inp, m_pair, eps_scale=self.loss_cfg.rot_loss_cfg.eps_scale)

            if self.loss_cfg.rot_loss_cfg.w is not None:
                total_loss += loss_rotationality * self.loss_cfg.rot_loss_cfg.w

            self.log(f"{step_type}/rotationality_loss", loss_rotationality)

        #Consistency L1
        if self.loss_cfg.consistency_x_l1 is not None:
            loss_back_vec_l1 = F.l1_loss(content, content_n).mean()
            self.log(f"{step_type}/loss_back_vec_l1", loss_back_vec_l1)
            if self.loss_cfg.consistency_x_l1.w is not None:
                if isinstance(self.loss_cfg.consistency_x_l1.w, float) or isinstance(self.loss_cfg.consistency_x_l1.w, int):
                    total_loss += loss_back_vec_l1*self.loss_cfg.consistency_x_l1.w
                else:
                    total_loss += loss_back_vec_l1*self.loss_cfg.consistency_x_l1.w(self.global_step)
            if self.do_switch_off_adversary_in_case_of_instability:
                if loss_back_vec_l1 > 0.0001:
                    self.switch_off_adversary_in_case_of_instability = True

        # Consistency losses 
        if self.loss_cfg.consistency_normalised_x is not None:
            var = content.var(dim=0, unbiased=False, keepdim=True).detach()
            self.log(f"{step_type}/mean_variance", var.mean())
            diff = content - content_n
            loss_back_vec_normalised = diff**2 / (var + 0.00001)
            loss_back_vec_normalised = (loss_back_vec_normalised).mean()
            self.log(f"{step_type}/loss_back_vec_explicit", (diff**2).mean())
            self.log(f"{step_type}/loss_back_vec_explicit_simpnorm", (diff**2).mean()/var.mean())
            self.log(f"{step_type}/loss_back_vec_normalised", loss_back_vec_normalised)
            if self.loss_cfg.consistency_normalised_x.w is not None:
                if isinstance(self.loss_cfg.consistency_normalised_x.w, float) or isinstance(self.loss_cfg.consistency_normalised_x.w, int):
                    total_loss += loss_back_vec_normalised*self.loss_cfg.consistency_normalised_x.w
                else:
                    total_loss += loss_back_vec_normalised*self.loss_cfg.consistency_normalised_x.w(self.global_step)
            if self.do_switch_off_adversary_in_case_of_instability:
                if loss_back_vec_normalised > 0.0001:
                    self.switch_off_adversary_in_case_of_instability = True

        # Nosed reconstruction loss
        if self.loss_cfg.noised_reco is not None:
            x_inp_noised = x_inp + torch.randn_like(x_inp)*self.loss_cfg.noised_reco.noise_std
            content_noised = self.encode_content(x_inp_noised, m_pair, mask=mask)
            recon_noised = self.decode(content_noised, style)
            loss_reco_noised = mse_loss(x_inp_noised, recon_noised).mean()
            self.log(f"{step_type}/loss_reco_noised", loss_reco_noised)
            if self.loss_cfg.noised_reco.w is not None:
                if isinstance(self.loss_cfg.noised_reco.w, float) or isinstance(self.loss_cfg.noised_reco.w, int):
                    total_loss += loss_reco_noised*self.loss_cfg.noised_reco.w
                else:
                    total_loss += loss_reco_noised*self.loss_cfg.noised_reco.w(self.global_step)

        # Noised consistency loss
        if self.loss_cfg.consistency_noised is not None:
            content_noised_2 = content + torch.randn_like(content)*self.loss_cfg.consistency_noised.noise_std
            recon_noised_2 = self.decode(content_noised_2, style_p)
            content_noised_n = self.encode_content(recon_noised_2, m_n, mask=mask)
            loss_back_vec_noised = mse_loss(content_noised_2, content_noised_n).mean()
            self.log(f"{step_type}/loss_back_vec_noised", loss_back_vec_noised)
            if self.loss_cfg.consistency_noised.w is not None:
                if isinstance(self.loss_cfg.consistency_noised.w, float) or isinstance(self.loss_cfg.consistency_noised.w, int):
                    total_loss += loss_back_vec_noised*self.loss_cfg.consistency_noised.w
                else:
                    total_loss += loss_back_vec_noised*self.loss_cfg.consistency_noised.w(self.global_step)

        # Gaussian consistency loss in the latent space
        if self.loss_cfg.consistency_noise is not None:
            content_noise = torch.randn_like(content)
            recon_noise = self.decode(content_noise, style_p)
            content_noise_n = self.encode_content(recon_noise, m_n, mask=mask)
            loss_back_vec_noise = mse_loss(content_noise, content_noise_n).mean()
            self.log(f"{step_type}/loss_back_vec_noise", loss_back_vec_noise)
            if self.loss_cfg.consistency_noise.w is not None:
                if isinstance(self.loss_cfg.consistency_noise.w, float) or isinstance(self.loss_cfg.consistency_noise.w, int):
                    total_loss += loss_back_vec_noise*self.loss_cfg.consistency_noise.w
                else:
                    total_loss += loss_back_vec_noise*self.loss_cfg.consistency_noise.w(self.global_step)


        self.log(f"{step_type}/switch_off_adversary_in_case_of_instability", int(self.switch_off_adversary_in_case_of_instability))

        if self.loss_cfg.consistency_cont is not None:
            loss_back_cont = mse_loss(style_p, style_n).mean()
            self.log(f"{step_type}/loss_back_cont", loss_back_cont)
            if self.loss_cfg.consistency_cont.w is not None:
                if isinstance(self.loss_cfg.consistency_cont.w, float) or isinstance(self.loss_cfg.consistency_cont.w, int):
                    total_loss += loss_back_cont*self.loss_cfg.consistency_cont.w
                else:
                    total_loss += loss_back_cont*self.loss_cfg.consistency_cont.w(self.global_step)

        # Full forward-backward consistency
        if self.loss_cfg.consistency_xx is not None:
            # s_con = content.std()
            # self.log(f"{step_type}/s_con", s_con)
            reco_2 = self.decode(content_n, style)
            loss_back_vec = mse_loss(x_inp, reco_2).mean()
            self.log(f"{step_type}/loss_cons_xx", loss_back_vec)
            if self.loss_cfg.consistency_xx.w is not None:
                if isinstance(self.loss_cfg.consistency_xx.w, float) or isinstance(self.loss_cfg.consistency_xx.w, int):
                    total_loss += loss_back_vec*self.loss_cfg.consistency_xx.w
                else:
                    total_loss += loss_back_vec*self.loss_cfg.consistency_xx.w(self.global_step)

        if self.loss_cfg.latent_ed_gaussian is not None:
            # Sample from unit Gaussian
            z = torch.randn_like(content)
            loss_ed_gaussian = self.energy_distance_loss(content, z)
            self.log(f"{step_type}/loss_ed_gaussian", loss_ed_gaussian)
            if self.loss_cfg.latent_ed_gaussian.w is not None:
                if isinstance(self.loss_cfg.latent_ed_gaussian.w, float) or isinstance(self.loss_cfg.latent_ed_gaussian.w, int):
                    total_loss += loss_ed_gaussian*self.loss_cfg.latent_ed_gaussian.w
                else:
                    total_loss += loss_ed_gaussian*self.loss_cfg.latent_ed_gaussian.w(self.global_step)

        # latent variance loss
        if self.loss_cfg.latent_variance_cfg is not None:
            content_for_var = content
            if getattr(self.loss_cfg.latent_variance_cfg, "random_rotation", False):
                # Random orthogonal matrix via QR decomposition; encourages isotropy
                D = content.shape[1]
                Q, _ = torch.linalg.qr(torch.randn(D, D, device=content.device, dtype=content.dtype))
                content_for_var = content @ Q
            var = content_for_var.var(dim=0)
            loss_latent_variance = torch.mean(torch.abs(1 - var)**self.loss_cfg.latent_variance_cfg.pow)
            if self.loss_cfg.latent_variance_cfg.w is not None:
                total_loss += loss_latent_variance*self.loss_cfg.latent_variance_cfg.w
            self.log(f"{step_type}/variance_regularization", loss_latent_variance)

        # latent covariance loss
        if self.loss_cfg.latent_covariance_cfg is not None:
            z = content - content.mean(dim=0, keepdim=True)   # [B, D]
            cov = (z.T @ z) / (z.shape[0] - 1)                # [D, D]

            off_diag = cov - torch.diag(torch.diag(cov))
            loss_latent_covariance = torch.mean(torch.abs(off_diag)**self.loss_cfg.latent_covariance_cfg.pow)

            if self.loss_cfg.latent_covariance_cfg.w is not None:
                total_loss += loss_latent_covariance * self.loss_cfg.latent_covariance_cfg.w

            self.log(f"{step_type}/covariance_regularization", loss_latent_covariance)

        # Mean loss
        if self.loss_cfg.latent_mean_cfg is not None:
            mean = content.mean(dim=0)
            loss_latent_mean = torch.mean(mean**2)
            if self.loss_cfg.latent_mean_cfg.w is not None:
                total_loss += loss_latent_mean*self.loss_cfg.latent_mean_cfg.w
            self.log(f"{step_type}/mean_regularization", loss_latent_mean)

        # L1 regularization
        if self.loss_cfg.l1_reg is not None:
            all_params = torch.cat([x.view(-1) for x in self.parameters()])
            l1_regularization = self.loss_cfg.l1_reg*torch.norm(all_params, 1)
            total_loss += l1_regularization
            self.log(f"{step_type}/l1_regularization", l1_regularization)

        # Attractive and repulsive loss that are parts of triplet loss
        if self.loss_cfg.attractive is not None:
            loss_attractive = -cosine_similarity(content, style[torch.randperm(batch_size)]).mean()
            self.log(f"{step_type}/loss_attractive", loss_attractive)
            if self.loss_cfg.attractive.w is not None:
                if isinstance(self.loss_cfg.attractive.w, float) or isinstance(self.loss_cfg.attractive.w, int):
                    total_loss += loss_attractive*self.loss_cfg.attractive.w
                else:
                    total_loss += loss_attractive*self.loss_cfg.attractive.w(self.global_step)
        
        if self.loss_cfg.repulsive is not None:
            loss_repulsive = torch.abs(cosine_similarity(content, style)).mean()
            self.log(f"{step_type}/loss_repulsive", loss_repulsive)
            if self.loss_cfg.repulsive.w is not None:
                if isinstance(self.loss_cfg.repulsive.w, float) or isinstance(self.loss_cfg.repulsive.w, int):
                    total_loss += loss_repulsive*self.loss_cfg.repulsive.w
                else:
                    total_loss += loss_repulsive*self.loss_cfg.repulsive.w(self.global_step)

        if self.transport_loss_mode == "mmd":
            loss_mmd = self._compute_mmd(x_inp, m_pair, x_n, m_n)
            w_mmd = self._cfg_get(self.mmd_cfg, "w", 1.0)
            total_loss += loss_mmd * w_mmd
            self.log(f"{step_type}/loss_mmd", loss_mmd)
        if self.transport_loss_mode == "energy":
            loss_energy = self._compute_energy_distance(x_inp, m_pair, x_n, m_n)
            w_energy = self._cfg_get(self.energy_cfg, "w", 1.0)
            total_loss += loss_energy * w_energy
            self.log(f"{step_type}/loss_energy", loss_energy)
        if self.transport_loss_mode == "energy_pair":
            # Split the batch in half and compute energy distance between the two halves, to avoid self-distance problems
            x_inp_half1, x_inp_half2 = torch.chunk(x_inp, 2, dim=0)
            m_pair_half1, m_pair_half2 = torch.chunk(m_pair, 2, dim=0)
            x_n_half1, x_n_half2 = torch.chunk(x_n, 2, dim=0)
            m_n_half1, m_n_half2 = torch.chunk(m_n, 2, dim=0)
            loss_energy_pair = self._compute_energy_distance(x_inp_half1, m_pair_half1, x_n_half2, m_n_half2) + self._compute_energy_distance(x_inp_half2, m_pair_half2, x_n_half1, m_n_half1)
            w_energy_pair = self._cfg_get(self.energy_cfg, "w", 1.0)
            total_loss += loss_energy_pair * w_energy_pair
            self.log(f"{step_type}/loss_energy", loss_energy_pair)

        if self.loss_cfg.projected_context_ed_gaussian is not None:
            n_dirs = self._cfg_get(self.loss_cfg.projected_context_ed_gaussian, "num_directions", 1)
            proj_eps = self._cfg_get(self.loss_cfg.projected_context_ed_gaussian, "eps", 1e-8)
            loss_projected_context_ed_gaussian = self._compute_projected_context_ed_gaussian(
                content,
                num_directions=n_dirs,
                eps=proj_eps,
            )
            self.log(f"{step_type}/loss_projected_context_ed_gaussian", loss_projected_context_ed_gaussian)

            w_proj_ed = self._cfg_get(self.loss_cfg.projected_context_ed_gaussian, "w", None)
            if w_proj_ed is not None:
                if isinstance(w_proj_ed, (float, int)):
                    total_loss += loss_projected_context_ed_gaussian * w_proj_ed
                else:
                    total_loss += loss_projected_context_ed_gaussian * w_proj_ed(self.global_step)

        # DisCO loss
        if self.loss_cfg.DisCO_loss_cfg is not None:
            if self.loss_cfg.DisCO_loss_cfg.mode == "e1_vs_e2":
                loss_disco = self.DisCO_loss(content, style)
            elif self.loss_cfg.DisCO_loss_cfg.mode == "e1_vs_w2":
                loss_disco = self.DisCO_loss(content, m_pair)
            self.log(f"{step_type}/DisCO_loss", loss_disco)
            if self.loss_cfg.DisCO_loss_cfg.w is not None:
                if isinstance(self.loss_cfg.DisCO_loss_cfg.w, float) or isinstance(self.loss_cfg.DisCO_loss_cfg.w, int):
                        total_loss += loss_disco*self.loss_cfg.DisCO_loss_cfg.w
                else:
                    w = self.loss_cfg.DisCO_loss_cfg.w(self.global_step)
                    total_loss += loss_disco*w
                    self.log(f"{step_type}_debug/DisCO_lossw", w)

        # Pearson loss
        if self.loss_cfg.pearson_loss_cfg is not None:
            loss_disco = self.pearson_loss(content, style)
            self.log(f"{step_type}/pearson_loss", loss_disco)
            if self.loss_cfg.pearson_loss_cfg.w is not None:
                if isinstance(self.loss_cfg.pearson_loss_cfg.w, float) or isinstance(self.loss_cfg.pearson_loss_cfg.w, int):
                        total_loss += loss_disco*self.loss_cfg.pearson_loss_cfg.w
                else:
                    total_loss += loss_disco*self.loss_cfg.pearson_loss_cfg.w(self.global_step)

        # Log the total loss
        self.log(f"{step_type}/total_loss", total_loss)
        if self.adversarial:
            return total_loss, content, style, x_inp, m_pair
        else:
            return total_loss

    def adversarial_loss(self, y_hat, y):
        # Ensure consistent shapes
        y_hat = y_hat.view(-1, 1)
        y = y.view(-1, 1)

        if self.adversarial_cfg.loss_function=="binary_cross_entropy":
            return F.binary_cross_entropy(y_hat, y.reshape((-1, 1)))
        if self.adversarial_cfg.loss_function=="binary_cross_entropy_with_logits":
            return F.binary_cross_entropy_with_logits(y_hat, y.reshape((-1, 1)))
        elif self.adversarial_cfg.loss_function=="mse":
            return mse_loss(y_hat, y.reshape((-1, 1)))
        elif self.adversarial_cfg.loss_function=="WGAN":
            return - 2 * torch.mean(y_hat * (y.reshape((-1, 1))-0.5))
        elif self.adversarial_cfg.loss_function == "hinge":
            # Convert {0,1} -> {-1,+1}
            y_signed = 2 * y - 1
            return torch.mean(F.relu(1 - y_signed * y_hat))
        elif self.adversarial_cfg.loss_function == "huberised_hinge":
            y_signed = 2 * y - 1   # {0,1} -> {-1,+1}
            margin = y_signed * y_hat
            z = 1 - margin

            delta = getattr(self.adversarial_cfg, "huber_delta", 1.0)

            loss = torch.where(
                z <= 0,
                torch.zeros_like(z),
                torch.where(
                    z < delta,
                    0.5 * z**2 / delta,
                    z - 0.5 * delta
                )
            )
            return loss.mean()
        else:
            raise ValueError(f"Unknown loss function: {self.adversarial_cfg.loss_function}")

    def training_step(self, sample: tuple, batch_idx: int) -> torch.Tensor:
        self._ema_should_update_this_batch = False

        if isinstance(self.adversarial, str) and "double_discriminator" in self.adversarial: 
            if self.use_disc_lat and self.use_disc_reco:
                optimizer_g, optimizer_d, optimizer_d2 = self.optimizers()
            elif self.use_disc_lat and not self.use_disc_reco:
                optimizer_g, optimizer_d = self.optimizers()
            elif not self.use_disc_lat and self.use_disc_reco:
                optimizer_g, optimizer_d2 = self.optimizers()
            else:
                optimizer_g = self.optimizers()
            # adversarial loss is binary cross-entropy
            total_loss, e1, e2, w1, w2 = self._shared_step(sample, step_type="train", _batch_index=batch_idx)
            batch_size=sample[0].shape[0]
            rpm = torch.randperm(batch_size)
            w2_perm = w2.clone()
            w2_perm = w2_perm[rpm]
            if self.adversarial_cfg.label_smoothing_eps>0:
                labels = torch.cat([torch.ones(batch_size) - self.adversarial_cfg.label_smoothing_eps, torch.zeros(batch_size) + self.adversarial_cfg.label_smoothing_eps]).type_as(w2_perm)
            else:
                labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).type_as(w2_perm)
            e1_copy = e1.clone()
            generated = self.decode(e1, e2[rpm])
            if self.adversarial_cfg.loss_function in ["binary_cross_entropy", "binary_cross_entropy_with_logits"]:
                threshold=np.log(2)
            elif self.adversarial_cfg.loss_function=="mse":
                threshold=0.25
            elif self.adversarial_cfg.loss_function=="WGAN":
                threshold=100000
            # train discriminator
            # Measure discriminator's ability to classify encoded samples with correct mass and encoded samples with incorrect mass
            allow_gen_train = True
            if self.current_epoch>=self.adversarial_cfg.warmup or self.adversarial_cfg.train_dis_in_warmup:
                if self.use_disc_lat:
                    # Train discriminator for latent space (with optional input noise)
                    e_lat = torch.cat([e1, e1_copy], dim=0)
                    m_lat = torch.cat([w2, w2_perm], dim=0)
                    e_lat, m_lat = self._add_disc_noise(e_lat, m_lat)
                    d_loss = self.adversarial_loss(self.disc_lat(e_lat, m_lat), labels)
                    self.toggle_optimizer(optimizer_d)
                    self.log("d_loss", d_loss, prog_bar=True)
                    self.zero_grad()
                    self.manual_backward(d_loss, retain_graph=True)
                    self.clip_gradients(optimizer_d, gradient_clip_val=self.gradient_clip_val)
                    optimizer_d.step()
                    self.untoggle_optimizer(optimizer_d)
                else:
                    d_loss = 0
                
                if self.use_disc_reco:
                    # Train discriminator for reconstruction/transport (with optional input noise)
                    reco_in = torch.cat([w1, generated], dim=0)
                    if self.use_disc_reco_doublecond:
                        reco_ctxt = torch.cat([torch.cat([w2, w2_perm], dim=0), torch.cat([w2_perm, w2], dim=0)], dim=1)
                    else:
                        reco_ctxt = torch.cat([w2, w2_perm], dim=0)
                    reco_in, reco_ctxt = self._add_disc_noise(reco_in, reco_ctxt)
                    d_loss_gen = self.adversarial_loss(self.disc_reco(reco_in, reco_ctxt),  labels)  
                        
                    self.toggle_optimizer(optimizer_d2)
                    self.log("d_loss_gen", d_loss_gen, prog_bar=True)
                    self.zero_grad()
                    self.manual_backward(d_loss_gen, retain_graph=True)
                    self.clip_gradients(optimizer_d2, gradient_clip_val=self.gradient_clip_val)
                    optimizer_d2.step()
                    self.untoggle_optimizer(optimizer_d2)
                    self.dis_steps_per_gen+=1
                else:
                    d_loss_gen = 0
                
                # Pause genrator training if discriminator is too weak
                if self.adversarial=="double_discriminator_priority":
                    if d_loss>threshold or d_loss_gen>threshold:
                        allow_gen_train = False
                if self.adversarial=="double_discriminator_priority_balancing":
                    if d_loss<0.69 or d_loss_gen<0.69:
                        self.adversarial_cfg.every_n_steps_g = max(self.adversarial_cfg.every_n_steps_g-1, 1)
                    elif d_loss>threshold or d_loss_gen>threshold:
                        self.adversarial_cfg.every_n_steps_g = min(self.adversarial_cfg.every_n_steps_g+1, 5)
                    
                    if d_loss>threshold or d_loss_gen>threshold:
                        allow_gen_train = False
 
                        
                if self.dis_steps_per_gen<self.adversarial_cfg.every_n_steps_g:
                    allow_gen_train = False
                        
            if self.current_epoch>self.afterglow_epoch:
                allow_gen_train = False

            # Train generator
            if self.current_epoch<self.adversarial_cfg.warmup or allow_gen_train:
                total_loss2 = total_loss
                if (self.current_epoch>self.adversarial_cfg.warmup or self.adversarial_cfg.g_loss_weight_in_warmup) and not self.switch_off_adversary_in_case_of_instability:
                    if isinstance(self.adversarial_cfg.g_loss_weight, float) or isinstance(self.adversarial_cfg.g_loss_weight, int):
                        g_loss_weight = self.adversarial_cfg.g_loss_weight
                    else:
                        g_loss_weight = self.adversarial_cfg.g_loss_weight(self.global_step)
                        self.log("g_loss_weight", g_loss_weight)
                    if isinstance(self.adversarial_cfg.g_loss_gen_weight, float) or isinstance(self.adversarial_cfg.g_loss_gen_weight, int):
                        g_loss_gen_weight = self.adversarial_cfg.g_loss_gen_weight
                    else:
                        g_loss_gen_weight = self.adversarial_cfg.g_loss_gen_weight(self.global_step)
                        self.log("g_loss_gen_weight", g_loss_gen_weight)
                    if self.use_disc_lat:
                        e_lat = torch.cat([e1, e1_copy], dim=0)
                        m_lat = torch.cat([w2, w2_perm], dim=0)
                        e_lat, m_lat = self._add_disc_noise(e_lat, m_lat)
                        g_loss = - self.adversarial_loss(self.disc_lat(e_lat, m_lat), labels)
                        total_loss2 += g_loss*g_loss_weight
                    if self.use_disc_reco:
                        reco_in = torch.cat([w1, generated], dim=0)
                        if self.use_disc_reco_doublecond:
                            reco_ctxt = torch.cat([torch.cat([w2, w2_perm], dim=0), torch.cat([w2_perm, w2], dim=0)], dim=1)
                        else:
                            reco_ctxt = torch.cat([w2, w2_perm], dim=0)
                        reco_in, reco_ctxt = self._add_disc_noise(reco_in, reco_ctxt)
                        g_loss_gen = - self.adversarial_loss(self.disc_reco(reco_in, reco_ctxt),  labels)  
                        total_loss2 += g_loss_gen*g_loss_gen_weight
                else:
                    total_loss2 = total_loss
                self.log("total_loss2", total_loss2, prog_bar=True)
                self.zero_grad()
                self.manual_backward(total_loss2)
                self.clip_gradients(optimizer_g, gradient_clip_val=self.gradient_clip_val)
                optimizer_g.step()
                self.untoggle_optimizer(optimizer_g)
                self._ema_should_update_this_batch = True
                self.log("dis_steps_per_gen", self.dis_steps_per_gen)
                self.dis_steps_per_gen = 0
        elif self.adversarial:
            assert False, "Adversarial mode not implemented"
        else:	
            self._ema_should_update_this_batch = True
            total_loss = self._shared_step(sample, step_type="train", _batch_index=batch_idx)
            return total_loss

    def _draw_event_transport_trajectories(self, w1_, m_pair_, var, var_name, masses="auto", max_traj=20, plot_second_derivative=True, return_type="PIL"):
        import gc
        if self.true_trajectory_function is not None and max_traj>10:
            max_traj=10

        max_traj = min(max_traj, w1_.shape[0])
        gen = torch.Generator(device="cpu")
        gen.manual_seed(1)
        idx = torch.randperm(w1_.shape[0], generator=gen)[:max_traj]
        w1 = w1_[idx].detach()
        m_pair = m_pair_[idx]
        content = self.encode_content(w1, m_pair).detach()
        recons = []
        zs = [] if self.adversarial else None

        device = w1.device
        if masses == "auto":
            interval= max(m_pair_.flatten().cpu().numpy()) - min(m_pair_.flatten().cpu().numpy())
            masses = np.linspace(min(m_pair_.flatten().cpu().numpy())-interval/10, max(m_pair_.flatten().cpu().numpy())+interval/10, 126)
        for m in masses:
            w2 = torch.full((w1.shape[0], 1), float(m), dtype=torch.float32, device=device)
            style = self.encode_style(w2).detach()
            recon = self.decode(content, style)
            recon = self.std_layer_x.reverse(recon) if self.add_standardizing_layer else recon
            recon = recon.detach()
            recons.append(recon)

            if self.adversarial:
                if self.use_disc_lat:
                    zs.append(self.disc_lat(content, style).detach())
                elif self.use_disc_reco:
                    zs.append(self.disc_reco(w1, w2).detach())

            del style, recon, w2
            torch.cuda.empty_cache()

        all_z = None
        if self.adversarial:
            all_z = torch.stack(zs)
            vmin = float(all_z.min())
            vmax = float(all_z.max())

        plt.figure()

        x = self.std_layer_ctxt.reverse(torch.tensor(masses).to(w1.device)).cpu().numpy().reshape(-1) if self.add_standardizing_layer else masses.cpu().numpy().reshape(-1)
        for i in range(max_traj):
            y = [float(recon[i, var].cpu().numpy()) for recon in recons]
            if self.adversarial:
                z = [float(z[i].cpu().numpy()) for z in zs]
                plt.plot(x, y, "black", zorder=i*2+1)
                plt.scatter(x, y, c=z, cmap="turbo", s=2, zorder=i*2+2, vmin=vmin, vmax=vmax)
                if i == 0:
                    plt.colorbar()
            else:
                plt.plot(x, y, "r")

        x_pair = self.std_layer_ctxt.reverse(m_pair).cpu().numpy() if self.add_standardizing_layer else m_pair.cpu().numpy()
        y_pair = self.std_layer_x.reverse(w1).cpu().numpy() if self.add_standardizing_layer else w1.cpu().numpy()
        plt.scatter(x_pair, y_pair[:, var], marker="x", label="originals", c="green")
        plt.xlabel("mass")
        plt.ylabel(f"dim{var}")
        plt.title(f"Event transport for {var_name}, global step: {self.global_step}, epoch: {self.current_epoch}")

        if self.true_trajectory_function is not None:
            if not hasattr(self, "cached_true_trajectories"):
                self.cached_true_trajectories = []
                true_y0 = []
                for x_start, y_strat in zip(x_pair, y_pair):
                    true_y = self.true_trajectory_function.inverse(y_strat, x_start)
                    true_y0.append(true_y)
                for i in range(max_traj):
                    traj=[]
                    for x_val in x:
                        true_y = self.true_trajectory_function.forward(true_y0[i], x_val)
                        traj.append(true_y)
                    self.cached_true_trajectories.append(traj)
                self.cached_true_trajectories = np.array(self.cached_true_trajectories)
            for i in range(max_traj):
                plt.plot(x, self.cached_true_trajectories[i][:, var], "b--", label="true trajectory" if i==0 else None, color="gray")

        fig = plt.gcf()
        fig.tight_layout()
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(height, width, 4)
        buf = buf[:, :, [1, 2, 3, 0]]  # ARGB to RGBA
        img = PIL.Image.fromarray(buf, "RGBA")
        plt.close("all")

        if plot_second_derivative:
            plt.figure()
            for i in range(max_traj):
                y = np.array([float(recon[i, var].cpu().numpy()) for recon in recons])
                plt.plot(x[1:-1], (-2*y[1:-1]+y[:-2]+y[2:])/(x[1]-x[0])**2, "r")
            plt.xlabel("mass")
            plt.ylabel(f"dim{var}")
            plt.title(f"Event transport 2nd derivative for {var_name}, global step: {self.global_step}, epoch: {self.current_epoch}")
            fig = plt.gcf()
            fig.tight_layout()
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(height, width, 4)
            buf = buf[:, :, [1, 2, 3, 0]]  # ARGB to RGBA
            img2 = PIL.Image.fromarray(buf, "RGBA")
            plt.close("all")
            # Force release memory
            del w1, content, recons, zs, all_z
            gc.collect()
            torch.cuda.empty_cache()
            return img, img2

        # Force release memory
        if all_z is not None:
            del all_z
        del w1, content, recons, zs
        gc.collect()
        torch.cuda.empty_cache()
        return img, None

    def validation_step (self, sample: tuple, batch_idx: int) -> torch.Tensor:
        if not self.adversarial:
            total_loss = self._shared_step(sample, step_type="valid", _batch_index=batch_idx)
            if batch_idx == 0 and self.valid_plots and self.current_epoch % self.valid_plot_freq == 0:
                with torch.no_grad():
                    x_inp, mask, m_pair, _ = self.interprete_input(sample, phase="train")
                    m_pair = m_pair.reshape([x_inp.shape[0], -1])
                    if self.do_dequantization:
                        x_inp = self.dequantization_layer(x_inp)
                    if self.add_standardizing_layer:
                        x_inp = self.std_layer_x(x_inp, mask=mask)
                        m_pair = self.std_layer_ctxt(m_pair)
                    for var in range(x_inp.shape[1]):
                        image_traj, images_2der = self._draw_event_transport_trajectories(
                            x_inp,
                            m_pair,
                            var=var,
                            var_name=self.var_group_list[0][var] if self.var_group_list else f"var{var}",
                            max_traj=20,
                        )
                        if wandb.run is not None:
                            image = wandb.Image(image_traj)
                            wandb.run.log({f"valid_images/transport_{self.var_group_list[0][var] if self.var_group_list else var}": image})
                            if images_2der is not None:
                                image_2der = wandb.Image(images_2der)
                                wandb.run.log({f"valid_images/transport_2der_{self.var_group_list[0][var] if self.var_group_list else var}": image_2der})
            return total_loss

        total_loss, e1, e2, w1, w2 = self._shared_step(sample, step_type="valid", _batch_index=batch_idx)
        batch_size=sample[0].shape[0]
        rpm = torch.randperm(batch_size)
        w2_perm = w2.clone()
        w2_perm = w2_perm[rpm]
        labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).type_as(w2_perm)
        e1_copy = e1.clone()
        generated = self.decode(e1, e2[rpm])
        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        if "double_discriminator" in self.adversarial or self.adversarial=="discriminator_priority":
                if self.use_disc_lat:
                    # Train discriminator for latent space
                    d_loss = self.adversarial_loss(self.disc_lat(torch.cat([e1, e1_copy], dim=0), torch.cat([w2, w2_perm], dim=0)), labels)
                    self.log("valid\d_loss", d_loss, prog_bar=True)
                
                if self.use_disc_reco:
                    # Train discriminator for reconstruction/transport
                    if self.use_disc_reco_doublecond:
                        d_loss_gen = self.adversarial_loss(self.disc_reco(torch.cat([w1, generated], dim=0), torch.cat([torch.cat([w2, w2_perm], dim=0), torch.cat([w2_perm, w2], dim=0)], dim=1)),  labels) 
                    else:
                        d_loss_gen = self.adversarial_loss(self.disc_reco(torch.cat([w1, generated], dim=0), torch.cat([w2, w2_perm], dim=0)),  labels)  
                    self.log("valid\d_loss_gen", d_loss_gen, prog_bar=True)
            
        if batch_idx == 0 and self.valid_plots and self.current_epoch%self.valid_plot_freq==0:
            for var in range(w1.shape[1]):
                image_traj, images_2der = self._draw_event_transport_trajectories(w1, w2, var=var, var_name=self.var_group_list[0][var], max_traj=20)
                if wandb.run is not None:
                    image = wandb.Image(image_traj)
                    wandb.run.log({f"valid_images/transport_{self.var_group_list[0][var]}": image})
                    if images_2der is not None:
                        image_2der = wandb.Image(images_2der)
                        wandb.run.log({f"valid_images/transport_2der_{self.var_group_list[0][var]}": image_2der})
        return total_loss

    def on_fit_start(self, *_args) -> None:
        """Function to run at the start of training."""
        # Define the metrics for wandb (otherwise the min wont be stored!)
        if wandb.run is not None:
            for step_type in ["train", "valid"]:
                wandb.define_metric(f"{step_type}/total_loss", summary="min")

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if not self.use_ema:
            return
        if not self._ema_should_update_this_batch:
            return
        if self.global_step < self.ema_start_step:
            return
        if (self.global_step - self.ema_start_step) % self.ema_update_every != 0:
            return
        self._ema_update()

    def on_validation_epoch_start(self) -> None:
        self._ema_apply_eval_weights()

    def on_validation_epoch_end(self) -> None:
        self._ema_restore_train_weights()

    def on_test_epoch_start(self) -> None:
        self._ema_apply_eval_weights()

    def on_test_epoch_end(self) -> None:
        self._ema_restore_train_weights()

    def on_predict_start(self) -> None:
        self._ema_apply_eval_weights()

    def on_predict_end(self) -> None:
        self._ema_restore_train_weights()

    def on_save_checkpoint(self, checkpoint) -> None:
        if not self.use_ema:
            return
        checkpoint["ema_state"] = {
            "shadow": self._ema_shadow,
            "num_updates": self._ema_num_updates,
        }

    def on_load_checkpoint(self, checkpoint) -> None:
        if "ema_state" not in checkpoint:
            return
        ema_state = checkpoint["ema_state"]
        self._ema_shadow = ema_state.get("shadow", {})
        self._ema_num_updates = int(ema_state.get("num_updates", 0))

    def configure_optimizers(self) -> dict:
        """Configure the optimisers and learning rate sheduler for this
        model."""
        if self.adversarial=="3optim_normal":
            enc2_params =  list(self.encoder2.parameters()) if hasattr(self.encoder2, "parameters") else []
            enc_dec_params = list(self.encoder1.parameters()) + enc2_params + list(self.decoder.parameters())
            opt_e = self.hparams.optimizer(params=self.encoder1.parameters())
            opt_g = self.hparams.optimizer(params=enc_dec_params)
            opt_d = self.hparams.optimizer(params=self.discriminator.parameters())
            if getattr(self.adversarial_cfg, "scheduler", None) is None:
                return [opt_e, opt_g, opt_d], []
            elif self.adversarial_cfg.scheduler == "same_given":
                sched_g = self.hparams.scheduler.scheduler(opt_g)
                sched_d = self.hparams.scheduler.scheduler(opt_d)
                sched_e = self.hparams.scheduler.scheduler(opt_e)
                return [opt_e, opt_g, opt_d], [sched_e, sched_g, sched_d]
            else: 
                sched_g = self.adversarial_cfg.scheduler.scheduler_g(opt_g)
                sched_d = self.adversarial_cfg.scheduler.scheduler_d(opt_d)
                sched_e = self.adversarial_cfg.scheduler.scheduler_e(opt_e)
                return [opt_e, opt_g, opt_d], [sched_e, sched_g, sched_d]            
        elif isinstance(self.adversarial, str) and "double_discriminator" in self.adversarial:
            if hasattr(self, "encoder2"):
                enc2_params =  list(self.encoder2.parameters()) if hasattr(self.encoder2, "parameters") else []
            else:
                enc2_params = []
            enc_dec_params = list(self.encoder1.parameters()) + enc2_params + list(self.decoder.parameters())
            optimisers, schedulers = [], []
            opt_g= self.hparams.optimizer(params=enc_dec_params)
            optimisers.append(opt_g)
            if self.use_disc_lat: 
                opt_d = self.hparams.optimizer(params=self.discriminator.parameters())
                optimisers.append(opt_d)
            if self.use_disc_reco: 
                opt_d2 = self.hparams.optimizer(params=self.discriminator2.parameters())
                optimisers.append(opt_d2)
            if getattr(self.adversarial_cfg, "scheduler", None) is None:
                pass
            elif self.adversarial_cfg.scheduler == "same_given":
                schedulers.append( self.hparams.scheduler.scheduler(opt_g))
                if self.use_disc_lat: schedulers.append(self.hparams.scheduler.scheduler(opt_d))
                if self.use_disc_reco: schedulers.append(self.hparams.scheduler.scheduler(opt_d2))
            else: 
                schedulers.append(self.adversarial_cfg.scheduler.scheduler_g(opt_g))
                if self.use_disc_lat: schedulers.append(self.adversarial_cfg.scheduler.scheduler_d(opt_d))
                if self.use_disc_reco: schedulers.append(self.adversarial_cfg.scheduler.scheduler_d2(opt_d2))
            return optimisers, schedulers
        elif self.adversarial:
            enc2_params =  list(self.encoder2.parameters()) if hasattr(self.encoder2, "parameters") else []
            enc_dec_params = list(self.encoder1.parameters()) + enc2_params + list(self.decoder.parameters())
            opt_g = self.hparams.optimizer(params=enc_dec_params)
            opt_d = self.hparams.optimizer(params=self.discriminator.parameters())
            if getattr(self.adversarial_cfg, "scheduler", None) is None:
                return [opt_g, opt_d], []
            elif self.adversarial_cfg.scheduler == "same_given":
                sched_g = self.hparams.scheduler.scheduler(opt_g)
                sched_d = self.hparams.scheduler.scheduler(opt_d)
                return [opt_g, opt_d], [sched_g, sched_d]
            else: 
                sched_g = self.adversarial_cfg.scheduler.scheduler_g(opt_g)
                sched_d = self.adversarial_cfg.scheduler.scheduler_d(opt_d)
                return [opt_g, opt_d], [sched_g, sched_d]
        else:
            # Finish initialising the partialy created methods
            opt = self.hparams.optimizer(params=self.parameters())

            sched = self.hparams.scheduler.scheduler(opt)

            # Return the dict for the lightning trainer
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, **self.hparams.scheduler.lightning},
            }

    def on_train_epoch_end(self) -> None:
        """Makes several plots of the jets and how they are reconstructed.
        """
        if self.adversarial:
            if self.lr_schedulers() is not None:
                for sched in self.lr_schedulers():
                    sched.step()
    
    def generate(self, sample: tuple) -> torch.Tensor:
        x_inp, mask, y_pair, y_new = self.interprete_input(sample, phase="generate")
        
        if self.do_dequantization:
            x_inp = self.dequantization_layer(x_inp)
        
        if self.add_standardizing_layer:
            x_inp = self.std_layer_x(x_inp)
            y_pair = self.std_layer_ctxt(y_pair)
            y_new = self.std_layer_ctxt(y_new)
        
        content = self.encode_content(x_inp, y_pair, mask=mask)
        style = self.encode_style(y_new)

        if self.total_skip is not None:
            recon = x_inp*self.total_skip + self.decode(content, style)
        else:
            recon = self.decode(content, style)
        
        if self.add_standardizing_layer:
            recon = self.std_layer_x.reverse(recon)
        
        if self.do_dequantization:
            recon = self.dequantization_layer.inverse(recon)
        
        return recon

    def predict_step(self, batch: tuple, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if self.second_input_mask:
            batch[3] = batch[3].reshape([len(batch[0]), -1])
            batch[2] = batch[2].reshape([len(batch[0]), -1])
            context = batch[3]
        else:
            batch[2] = batch[2].reshape([len(batch[0]), -1])
            batch[1] = batch[1].reshape([len(batch[0]), -1])
            context = batch[2]
        if self.var_group_list is not None:
            sample = self.generate(batch).squeeze(1)
            if len(sample.shape)==1:
                sample = sample.unsqueeze(1)
            sample = sample.reshape(-1, sample.shape[-1])
            result = {var_name: column.reshape(-1, 1) for var_name, column in zip(self.var_group_list[0], sample.T)}
            result.update({var_name: column.reshape(-1, 1) for var_name, column in zip(self.var_group_list[1], context.T)})                
        else:
            sample = self.generate(batch)
            result = sample
        return result

