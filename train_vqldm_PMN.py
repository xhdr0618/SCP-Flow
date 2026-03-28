"""
tHPM-LDM:
train & interface

dataset requirement:
(1) process logitudial data into npz files using ./datamodule/make_data_SIGF.py (path: xxxx/SIGF_make)
(2) the structure of SIGF_make will be:
SIGF_make/
├── test/
│   ├── SD3434_OS_0.npz
│   ├── SD3434_OS_1.npz
│   └── ...
├── train/
│   ├── SD1284_OS_0.npz
│   ├── SD1284_OS_1.npz
│   └── ...
└── validation/
    ├── SD1006_OD_0.npz
    ├── SD1006_OD_1.npz
    └── ...
    
Note: 
We evaluate the performace of the generated results via `metrics.calculate_metric.py`.
Only using the strucure of `on_test_start`, `test_step` and `on_test_epoch_end` to get *_gen.png and save *_gt.png.
We use diff_time=50 for validation (DDIM, just for quick check), and use diff_time=1000 for training and testing(DDPM)
"""
# -*- coding:utf-8 -*-
import os
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import partial
import numpy as np
import cv2
from tqdm import tqdm

import pytorch_lightning as pl
import torch
import pandas as pd
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.optim.lr_scheduler import LambdaLR

from base.init_experiment import initExperiment
from datamodule.seq_fundus_2D_datamodule import SeqFundusDatamodule
from ddpm_default import DiffusionWrapper, make_beta_schedule
from ldm.lr_scheduler import LambdaLinearScheduler
from ldm.modules.condition_gen_MSTFCM_PopuMemory import ConditionGenMSTFCMPopuMemory
from ldm.modules.diffusionmodules.util import extract_into_tensor, noise_like
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.ema_hash import LitEma
from ldm.util import default
from models.flow_forecaster import ProgressionFlowForecaster

from train_vqgan import VQGAN
from metrics.calculate_metric import calculate_fid, calculate_ssim, calculate_psnr
from utils.util import draw_loss_fig, draw_psnr_fig, draw_ssim_fig


def get_parser():
    parser = ArgumentParser()

    parser.add_argument("--exp_name", type=str, default='DBGLAtMSHFPMQM')
    parser.add_argument('--first_stage_ckpt', type=str, default='pre-trained/VQGAN/vqgan.ckpt', help="path to your VQGAN ckpt")
    parser.add_argument('--data_root', type=str, default='your_SIFG_make_dataset_path', help="path to your dataset(maked into npz)")
    parser.add_argument('--result_root', type=str, default='results/LAtMSHFPMQM', help="path of your training results(relative path)")
    parser.add_argument('--image_save_dir', type=str, default='results/Eval-LAtMSHFPMQM',help="path of your testing results")

    parser.add_argument("--command", default="fit", help="running mode: (fit/test)")
    parser.add_argument("--image_size", default=(256, 256), help="image input size")
    parser.add_argument("--latent_size", default=(32, 32), help="the size of the latent code")
    parser.add_argument("--latent_channel", default=4, help="the channel of latent code")

    # train args
    parser.add_argument("--max_epochs", type=int, default=700)
    parser.add_argument("--limit_train_batches", type=int, default=1000)
    parser.add_argument("--limit_val_batches", type=int, default=3)  
    parser.add_argument("--base_learning_rate", type=float, default=5.0e-05)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--scale_lr', type=bool, default=False, help="scale base-lr by ngpu * batch_size * n_accumulate")  # use as multi-GPU (not test)
    parser.add_argument('--num_workers', type=int, default=2)

    # lightning args
    parser.add_argument('--profiler', default='simple')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', default="auto")
    parser.add_argument('--reproduce', type=int, default=True)

    # resume of LDM for testing
    parser.add_argument('--resume', default='', type=str, help='Path for checkpoint to load and resume') # "" for training and "pre-trained/tHPM-LDM/tHPM-LDM.ckpt" for testing
    parser.add_argument('--test_type', default='test', type=str, help='Data type for testing')

    # diffusion sampling steps
    parser.add_argument('--diff_time', default=1000, type=int, help='Sampling time steps for diffusion model')  # using 50 for validation (DDIM) and using 1000 for training and testing(DDPM)
    parser.add_argument('--diff_pred_cond', default='LAtMSHFPMQM', type=str, help='Condition setting for diffusion model')  # (for tHPM-LDM, we use "LAtMSHFPMQM")
    parser.add_argument('--predictor_type', default='diffusion', type=str, choices=['diffusion', 'scpflow'], help='future predictor backbone')
    parser.add_argument('--condition_dim', default=128, type=int, help='condition embedding dim for flow backbone')
    parser.add_argument('--flow_hidden_channels', default=128, type=int, help='hidden channels of SCP-Flow backbone')
    parser.add_argument('--flow_tau_embed_dim', default=128, type=int, help='tau embedding dim of SCP-Flow backbone')
    parser.add_argument('--flow_bridge_sigma', default=0.05, type=float, help='bridge noise std for SCP-Flow')
    parser.add_argument('--flow_velocity_weight', default=1.0, type=float, help='velocity loss weight')
    parser.add_argument('--flow_target_weight', default=1.0, type=float, help='target latent loss weight')
    parser.add_argument('--flow_recon_weight', default=1.0, type=float, help='reconstruction loss weight')
    parser.add_argument('--interval_loss_weight', default=0.2, type=float, help='next-visit interval loss weight')
    parser.add_argument('--uncertainty_loss_weight', default=0.1, type=float, help='uncertainty loss weight')
    parser.add_argument('--consistency_loss_weight', default=0.1, type=float, help='progression consistency loss weight')
    parser.add_argument('--structure_loss_weight', default=0.0, type=float, help='structure supervision loss weight')
    parser.add_argument('--structure_dim', default=128, type=int, help='structure condition embedding dim')
    parser.add_argument('--structure_roi_size', default=96, type=int, help='ROI crop size for structure condition extraction')
    parser.add_argument('--missing_visit_count', default=0, type=int, help='number of historical visits to hide by duplication/blanking')
    parser.add_argument('--missing_visit_strategy', default='none', type=str, choices=['none', 'tail', 'uniform', 'random'], help='strategy for missing-visit robustness evaluation')
    parser.add_argument('--missing_seed', default=0, type=int, help='seed for random missing-visit sampling')
    return parser

def from_argparse_args(config, callbacks=None):
    """
    pytorch_lighning 2.4.0 - preparing args for Trainer
    """
    args = {
        "callbacks": callbacks,
        "max_epochs": config.max_epochs,
        "accelerator": config.accelerator,
        "limit_train_batches": config.limit_train_batches,
        "limit_val_batches": config.limit_val_batches,  # the number of batches for validation
        "accumulate_grad_batches": config.accumulate_grad_batches,
        "profiler": config.profiler,
        "devices": config.devices if config.devices != "auto" else "auto",
        "default_root_dir": config.default_root_dir if hasattr(config, "default_root_dir") else None,
        "check_val_every_n_epoch": 1 if getattr(config, "predictor_type", "diffusion") == "scpflow" else 10,
    }
    # Disable logger if we're in test mode (not training)
    if config.command == "test":
        args["logger"] = False
    return args


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class VQModelInterface(VQGAN):
    """
    interface of VQGAN
    """
    def __init__(self):
        super().__init__()

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def encode_only(self, x):
        h = self.encoder(x)
        return h

    def quant_only(self, x):
        h = self.quant_conv(x)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class SCPFlowLDM(pl.LightningModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.instantiate_first_stage(opts)
        self.condition_model = ConditionGenMSTFCMPopuMemory(d_model=opts.condition_dim)
        self.flow_model = ProgressionFlowForecaster(
            latent_channels=opts.latent_channel,
            cond_dim=opts.condition_dim,
            hidden_channels=opts.flow_hidden_channels,
            tau_embed_dim=opts.flow_tau_embed_dim,
            struct_dim=opts.structure_dim,
        )
        self.scale_by_std = True
        self.register_buffer('scale_factor', torch.tensor(1.0))
        self.use_ema = False
        self.test_records = []

    def instantiate_first_stage(self, opts):
        model = VQModelInterface()
        states = torch.load(opts.first_stage_ckpt, map_location=self.device)
        model.load_state_dict(states['state_dict'])
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        return self.first_stage_model.decode(z)

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0:
            x = batch['image'].to(self.device)
            encoder_posterior = self.encode_first_stage(x[:, -1, :, :, :])
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1.0 / z.flatten().std())
            self.log('val/loss', 0.0, prog_bar=False, logger=True)

    def save_image(self, img, path):
        if img.ndim == 4:
            img = img[0, :, :, :]
        img = img[:, :, :] * 255
        img = img.permute(1, 2, 0)
        img[img > 255] = 255
        img[img < 0] = 0
        img = np.uint8(img.data.cpu().numpy())
        cv2.imwrite(path, img)

    def _build_structure_condition(self, batch):
        struct_cond = {}
        found = False
        for key in ["vcdr", "ocod", "disc_roi", "polar_roi"]:
            if key in batch:
                struct_cond[key] = batch[key]
                found = True
        return struct_cond if found else None

    def get_input(self, batch):
        x_full = batch['image'].to(self.device)
        t_full = batch['time'].to(self.device).float()
        l_full = batch['label'].to(self.device)
        x_id = batch['image_id']

        x_cond = x_full.clone()
        t_cond = t_full.clone()
        l_cond = l_full.clone()
        x_cond[:, -1, :, :, :] = x_full[:, -2, :, :, :]
        t_cond[:, -1] = t_full[:, -2]
        l_cond[:, -1] = l_full[:, -2]

        prev_latent = self.get_first_stage_encoding(self.encode_first_stage(x_full[:, -2, :, :, :])).detach()
        target_latent = self.get_first_stage_encoding(self.encode_first_stage(x_full[:, -1, :, :, :])).detach()
        prev_prev_latent = self.get_first_stage_encoding(self.encode_first_stage(x_full[:, -3, :, :, :])).detach()

        target_interval = (t_full[:, -1] - t_full[:, -2]).float()
        struct_cond = self._build_structure_condition(batch)

        cond = {
            "x_seq": x_cond,
            "t_seq": t_cond,
            "l_seq": l_cond,
            "prev_latent": prev_latent,
            "prev_prev_latent": prev_prev_latent,
            "target_interval": target_interval,
            "struct_cond": struct_cond,
            "x_id": x_id,
        }
        return target_latent, cond, x_full[:, -1, :, :, :], x_id

    def _get_conditions(self, cond, current_step=None):
        x_seq = cond["x_seq"]
        t_seq = cond["t_seq"]
        l_seq = cond["l_seq"]
        if self.training:
            ch, cp = self.condition_model(x_seq, t_seq, l_seq, current_step=current_step)
        else:
            ch, cp = self.condition_model(x_seq, t_seq, l_seq)
        return ch, cp

    def _sample_bridge(self, prev_latent, target_latent):
        batch_size = prev_latent.shape[0]
        tau = torch.rand(batch_size, device=prev_latent.device, dtype=prev_latent.dtype)
        tau = tau.clamp(1e-3, 1 - 1e-3)
        mean = (1.0 - tau[:, None, None, None]) * prev_latent + tau[:, None, None, None] * target_latent
        bridge_std = self.opts.flow_bridge_sigma * torch.sqrt(tau * (1.0 - tau))
        z_t = mean + bridge_std[:, None, None, None] * torch.randn_like(prev_latent)
        target_velocity = target_latent - prev_latent
        return tau, z_t, target_velocity

    def _forward_losses(self, target_latent, cond, target_img, current_step=None):
        prev_latent = cond["prev_latent"]
        prev_prev_latent = cond["prev_prev_latent"]
        target_interval = cond["target_interval"]
        struct_cond = cond["struct_cond"]

        ch, cp = self._get_conditions(cond, current_step=current_step)
        tau, z_t, target_velocity = self._sample_bridge(prev_latent, target_latent)
        outputs = self.flow_model(
            z_t=z_t,
            tau=tau,
            ch=ch,
            cp=cp,
            struct_cond=struct_cond,
            prev_latent=prev_latent,
        )

        pred_latent = outputs["pred_target"]
        pred_img = self.decode_first_stage(pred_latent)

        velocity_loss = F.mse_loss(outputs["velocity"], target_velocity)
        target_loss = F.mse_loss(pred_latent, target_latent)
        recon_loss = F.l1_loss(pred_img, target_img)
        interval_loss = F.mse_loss(outputs["pred_interval"], target_interval)

        latent_residual = (pred_latent - target_latent).pow(2).mean(dim=[1, 2, 3]).detach()
        uncertainty = outputs["pred_uncertainty"]
        uncertainty_loss = torch.mean(latent_residual / uncertainty + torch.log(uncertainty))

        pred_delta = (pred_latent - prev_latent).mean(dim=[2, 3])
        hist_delta = (prev_latent - prev_prev_latent).mean(dim=[2, 3]).detach()
        consistency_loss = F.mse_loss(pred_delta, hist_delta)

        structure_loss = torch.tensor(0.0, device=self.device)
        if struct_cond is not None and struct_cond.get("vcdr") is not None and struct_cond.get("ocod") is not None:
            struct_target = torch.cat(
                [
                    struct_cond["vcdr"].to(self.device).view(-1, 1).float(),
                    struct_cond["ocod"].to(self.device).view(-1, 1).float(),
                ],
                dim=-1,
            )
            structure_loss = F.mse_loss(outputs["pred_structure"], struct_target)

        loss = (
            self.opts.flow_velocity_weight * velocity_loss
            + self.opts.flow_target_weight * target_loss
            + self.opts.flow_recon_weight * recon_loss
            + self.opts.interval_loss_weight * interval_loss
            + self.opts.uncertainty_loss_weight * uncertainty_loss
            + self.opts.consistency_loss_weight * consistency_loss
            + self.opts.structure_loss_weight * structure_loss
        )

        return loss, {
            "pred_latent": pred_latent,
            "pred_img": pred_img,
            "pred_interval": outputs["pred_interval"],
            "pred_uncertainty": outputs["pred_uncertainty"],
            "velocity_loss": velocity_loss.detach(),
            "target_loss": target_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "interval_loss": interval_loss.detach(),
            "uncertainty_loss": uncertainty_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
            "structure_loss": structure_loss.detach(),
        }

    def training_step(self, batch, batch_idx):
        target_latent, cond, target_img, _ = self.get_input(batch)
        loss, aux = self._forward_losses(target_latent, cond, target_img, current_step=self.global_step)
        self.log("train/loss", loss, prog_bar=True, logger=True)
        self.log("train/velocity_loss", aux["velocity_loss"], prog_bar=False, logger=True)
        self.log("train/target_loss", aux["target_loss"], prog_bar=False, logger=True)
        self.log("train/recon_loss", aux["recon_loss"], prog_bar=True, logger=True)
        self.log("train/interval_loss", aux["interval_loss"], prog_bar=False, logger=True)
        self.log("train/uncertainty_loss", aux["uncertainty_loss"], prog_bar=False, logger=True)
        self.log("train/consistency_loss", aux["consistency_loss"], prog_bar=False, logger=True)
        self.log("train/structure_loss", aux["structure_loss"], prog_bar=False, logger=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        target_latent, cond, target_img, _ = self.get_input(batch)
        loss, aux = self._forward_losses(target_latent, cond, target_img)
        psnr_value = calculate_psnr(target_img[0].detach().cpu(), aux["pred_img"][0].detach().cpu(), data_range=1)
        self.log("val/loss", loss, prog_bar=True, logger=True)
        self.log("val/recon_loss", aux["recon_loss"], prog_bar=False, logger=True)
        self.log("val/psnr", psnr_value, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        target_latent, cond, target_img, x_id = self.get_input(batch)
        prev_latent = cond["prev_latent"]
        ch, cp = self._get_conditions(cond)
        tau = torch.zeros(prev_latent.shape[0], device=self.device, dtype=prev_latent.dtype)
        outputs = self.flow_model(
            z_t=prev_latent,
            tau=tau,
            ch=ch,
            cp=cp,
            struct_cond=cond["struct_cond"],
            prev_latent=prev_latent,
        )
        pred_latent = outputs["pred_target"]
        pred_img = self.decode_first_stage(pred_latent)
        rec_img = self.decode_first_stage(target_latent)

        save_dir = self.opts.image_save_dir
        os.makedirs(save_dir, exist_ok=True)
        img_save_dir = os.path.join(save_dir, "image")
        os.makedirs(img_save_dir, exist_ok=True)

        for each_idx in range(target_img.shape[0]):
            case_name = os.path.basename(x_id[each_idx]).split(".")[0]
            self.save_image(pred_img[each_idx], os.path.join(img_save_dir, f"{case_name}_gen.png"))
            self.save_image(target_img[each_idx], os.path.join(img_save_dir, f"{case_name}_gt.png"))
            self.save_image(rec_img[each_idx], os.path.join(img_save_dir, f"{case_name}_rec.png"))
            missing_mask = batch.get("missing_mask")
            missing_count = 0.0
            if missing_mask is not None:
                missing_count = missing_mask[each_idx].float().sum().detach().cpu().item()
            self.test_records.append(
                {
                    "case_id": case_name,
                    "pred_interval": outputs["pred_interval"][each_idx].detach().cpu().item(),
                    "target_interval": cond["target_interval"][each_idx].detach().cpu().item(),
                    "pred_uncertainty": outputs["pred_uncertainty"][each_idx].detach().cpu().item(),
                    "missing_count": missing_count,
                    "missing_strategy": self.opts.missing_visit_strategy,
                }
            )

    def on_test_start(self):
        self.test_records = []

    def on_test_epoch_end(self):
        save_dir = self.opts.image_save_dir
        os.makedirs(save_dir, exist_ok=True)
        if len(self.test_records) > 0:
            df = pd.DataFrame(self.test_records)
            df["abs_interval_error"] = (df["pred_interval"] - df["target_interval"]).abs()
            df.to_csv(os.path.join(save_dir, "flow_predictions.csv"), index=False)
        print("All Done!")

    def configure_optimizers(self):
        lr = self.opts.learning_rate
        params = list(self.flow_model.parameters()) + list(self.condition_model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        scheduler_config = {'warm_up_steps': [10000],
                            'cycle_lengths': [10000000000000],
                            'f_start': [0.1],
                            'f_max': [1.0],
                            'f_min': [0.5]}
        scheduler = LambdaLinearScheduler(**scheduler_config)
        scheduler = [
            {
                'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                'interval': 'step',
                'frequency': 1
            }]
        return [opt], scheduler


class LDM(pl.LightningModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        unet_config = {
            'image_size': opts.latent_size,
            'in_channels': opts.latent_channel,
            'out_channels': opts.latent_channel,
            'model_channels': 192,
            'attention_resolutions': [1, 2, 4, 8],
            'num_res_blocks': 2,
            'channel_mult': [1, 2, 2, 4, 4],
            'num_heads': 8,  # head of cross attention condition
            'use_scale_shift_norm': True,
            'resblock_updown': True,
            "dropout": 0.5,
            "context_dim": 128,  # dim of cross attention condition
            "use_spatial_transformer": True  # Cross attention in LDM
        }
        self.instantiate_first_stage(opts)

        """ set up diffusion model w.r.t. diff_pred_cond """
        if hasattr(self.opts, "diff_pred_cond"):
            if self.opts.diff_pred_cond == "default":
                self.model = DiffusionWrapper(unet_config, conditioning_key=None)
            elif self.opts.diff_pred_cond in ["LAtMSHFPMQM"]:
                unet_config["in_channels"] = opts.latent_channel * 2  # we need 2*latent_channel since latent alignment
                self.model = DiffusionWrapper(unet_config, conditioning_key=self.opts.diff_pred_cond)
            else:
                raise RuntimeError(f"diff pred condition: {self.opts.diff_pred_cond} is not supported")
        else:
            self.model = DiffusionWrapper(unet_config, conditioning_key=None)

        self.latent_size = opts.latent_size
        self.channels = opts.latent_channel

        self.parameterization = "eps"  # all assuming fixed variance schedules
        self.loss_type = "l2"
        self.use_ema = True
        self.use_positional_encodings = False
        self.v_posterior = 0.  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        self.original_elbo_weight = 0.
        self.l_simple_weight = 1.
        self.l_swav_weight = 0.1
        self.scale_by_std = True
        self.log_every_t = 100

        scale_factor = 1.0
        if not self.scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        self.register_schedule()
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")


        self.train_psnr = []
        self.train_ssim = []
        self.train_loss = []

    def setup(self, stage=None):
        if stage == 'fit':
            # using validation dataset during training
            if hasattr(self.trainer, 'val_dataloaders'):
                self._val_dataloader = self.trainer.val_dataloaders

    def register_schedule(self,
                          given_betas=None,
                          beta_schedule="linear",
                          timesteps=1000,
                          linear_start=0.0015,
                          linear_end=0.0155,
                          cosine_s=8e-3):
        """
        Difffusion - Noise Schedule
        """
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def instantiate_first_stage(self, opts):
        """
        loading pre-trained VQGAN
        """
        model = VQModelInterface()
        states = torch.load(opts.first_stage_ckpt, map_location=self.device)
        model.load_state_dict(states['state_dict'])
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    @torch.no_grad()
    def encode_first_stage_only(self, x):
        return self.first_stage_model.encode_only(x)

    @torch.no_grad()
    def encode_first_stage_quant(self, x):
        return self.first_stage_model.quant_only(x)

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()  # the latent of diffusion, do the gaussian sampling
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        """
        register the scale_factor as the STD-RESCALLING [1. / z.flatten().std()] using
        the fist batch of the first epoch
        """
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = batch['image']
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x[:, -1, :, :, :])
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")
            
            # init log metric for checkpointing
            self.log('psnr_mean', 0, prog_bar=False, logger=True)
            self.log('ssim_mean', 0, prog_bar=False, logger=True)
            self.log("fid_mean", 10000, prog_bar=False, logger=True)

    @torch.no_grad()
    def get_input(self, batch):
        """
        get the input of model according to the input batch
        Be very careful for the last image and its encoding
        """
        x = batch['image'].to(self.device)
        c2 = batch['time'].to(self.device)
        c3 = batch['label'].to(self.device)

        """ matching diff predict condition options """
        if self.opts.diff_pred_cond in ["LAtMSHFPMQM"]:
            f5 = self.scale_factor * self.get_first_stage_encoding(self.encode_first_stage(x[:, -2, :, :, :])).detach()
            c = {'c1': x, 'c2': c2, 'c3': c3, "f5": f5, "x_id": batch['image_id']}
        else:
            raise RuntimeError(f"diff pred condition: {self.opts.diff_pred_cond} is not supported")

        encoder_posterior = self.encode_first_stage(x[:, -1, :, :, :])  # the last one img
        z = self.scale_factor * encoder_posterior.detach()  # the latent code for the last one img

        x_id = batch['image_id']
        return z, c, x[:, -1, :, :, :], x_id  # latent-code of last one img, conditions, last one img, image_id(patient id - seq)

    def forward(self, x, c, current_step=None):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, c, t, current_step)

    def training_step(self, batch, batch_idx):
        """
        train method for pl trainer, return loss
        """
        z, c, _, _ = self.get_input(batch)  # encoding and conditions
        loss, loss_dict = self(z, c, current_step=self.global_step)  # excute self.forward(), go to self.p_losses(), return losses

        if batch_idx == 0:
            self.sample_batch = batch
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def save_image(self, img, path):
        if img.ndim == 4:
            img = img[0, :, :, :]
        img = img[:, :, :] * 255
        img = img.permute(1, 2, 0)
        img[img > 255] = 255
        img[img < 0] = 0
        img = np.uint8(img.data.cpu().numpy())
        cv2.imwrite(path, img)

    def on_validation_start(self):
        if self.current_epoch > 100:  # do validation after 100 epoches to save time  
            self.training = False
            self.each_psnr = []
            self.each_ssim = []
            self.each_loss = []
            self.each_fid = []

    def validation_step(self, batch, batch_idx):
        if self.current_epoch > 100:  # do validation after 100 epoches to save time
            with self.ema_scope("Plotting"):
                img_save_dir = os.path.join(self.opts.default_root_dir, 'train_progress', str(self.current_epoch))
                os.makedirs(img_save_dir, exist_ok=True)
                n_batches = batch['image'].shape[0]
                z, c, x, _ = self.get_input(batch)
                loss, _ = self(z, c, current_step=None)
                self.each_loss.append(loss.item())

                """ sampling with diff_time in val stage """
                if self.opts.diff_time == self.num_timesteps:
                    x_samples = self.sample(c=c, batch_size=n_batches, return_intermediates=False, clip_denoised=True)
                elif self.opts.diff_time in range(0, self.num_timesteps):
                    x_samples = self.sample_ddim(c=c, batch_size=n_batches, n_steps=self.opts.diff_time)
                else:
                    raise RuntimeError(f"Receive a wrong range of diff_time: {self.opts.diff_time}")

                img_samples = self.decode_first_stage(x_samples)
                x_rec = self.decode_first_stage(z)

                for idx in range(n_batches):
                    name = batch["image_id"][idx].split(".")[0]
                    self.save_image(img_samples[idx], os.path.join(img_save_dir, f'{name}_gen_sample.png'))
                    self.save_image(x[idx], os.path.join(img_save_dir, f'{name}_x_sample.png'))
                    self.save_image(x_rec[idx], os.path.join(img_save_dir, f'{name}_x_rec.png'))
                    self.each_psnr.append(calculate_psnr(img_samples[idx], x[idx]))
                    self.each_ssim.append(calculate_ssim(img_samples[idx], x[idx]))
                    self.each_fid.append(calculate_fid(img_samples[idx], x[idx]))

    def on_validation_epoch_end(self):
        if self.current_epoch > 100:  # do validation after 100 epoches to save time
            self.train_psnr.append([self.current_epoch, np.mean(self.each_psnr), np.std(self.each_psnr)])
            self.train_ssim.append([self.current_epoch, np.mean(self.each_ssim), np.std(self.each_ssim)])

            draw_psnr_fig(self.train_psnr, saving_path=os.path.join(self.opts.default_root_dir, 'train_psnr.png'))
            draw_ssim_fig(self.train_ssim, saving_path=os.path.join(self.opts.default_root_dir, 'train_ssim.png'))
            self.log('psnr_mean', np.mean(self.each_psnr), prog_bar=False, logger=True, on_step=False, on_epoch=True)
            self.log('ssim_mean', np.mean(self.each_ssim), prog_bar=False, logger=True, on_step=False, on_epoch=True)
            self.log("fid_mean", np.mean(self.each_fid), prog_bar=True, logger=True)
            self.training = True

    def on_train_epoch_end(self):
        print(f"End current epoch: {self.current_epoch}")
        metrics = self.trainer.callback_metrics
        if "train/loss" in metrics:
            current_loss = metrics["train/loss"]
            print(self.current_epoch, current_loss.item())
            self.train_loss.append([self.current_epoch, current_loss.item() if torch.is_tensor(current_loss) else current_loss])
        if len(self.train_loss) != 0:
            draw_loss_fig(self.train_loss, saving_path=os.path.join(self.opts.default_root_dir, 'train_loss.png'))

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def p_losses(self, x_start, cond, t, current_step=None):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t, cond, current_step=current_step)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")
        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])  # loss for epsilon-prediction

        log_prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})

        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = self.get_loss(model_out, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        # loss for swav
        swav = self.model.diffusion_model.condition.pmn.mce.swav_loss_value
        if swav is None:
            loss = loss_simple + self.original_elbo_weight * loss_vlb
        else:
            loss_swav = swav
            loss_dict.update({f'{log_prefix}/loss_swav': loss_swav})
            loss = loss_simple + self.original_elbo_weight * loss_vlb + self.l_swav_weight * loss_swav

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
        return loss

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    """ Sampling schedule for DDPM """
    def p_mean_variance(self, x, c, t, clip_denoised):
        model_out = self.model(x, t, c)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised, temperature=1., noise_dropout=0., repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x0 = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0

    @torch.no_grad()
    def p_sample_loop(self, c, shape, return_intermediates=False, log_every_t=100, clip_denoised=True):
        device = self.betas.device
        b = shape[0]
        x = torch.randn(shape, device=device)
        # intermediates = [x]
        intermediates_x0 = [x]
        with tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps) as pbar:
            for i in pbar:
                x, x0 = self.p_sample(x, c, torch.full((b,), i, device=device, dtype=torch.long),
                                      clip_denoised=clip_denoised)
                if i % log_every_t == 0 or i == self.num_timesteps - 1:
                    # intermediates.append(x)
                    intermediates_x0.append(x0)
        if return_intermediates:
            return x, intermediates_x0
        return x

    @torch.no_grad()
    def sample(self, c=None, batch_size=1, return_intermediates=False, clip_denoised=True):
        return self.p_sample_loop(c,
                                  [batch_size, self.channels] + list(self.latent_size),
                                  return_intermediates=return_intermediates,
                                  clip_denoised=clip_denoised)

    """ Sampling schedule for DDIM """
    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, t_next, clip_denoised):
        b, *_, device = *x.shape, x.device
        model_output, x0 = self.predict_eps_from_z(x, t, c, clip_denoised)
        alpha_bar = self.alphas_cumprod[t]
        alpha_bar_next = self.alphas_cumprod[t_next]
        while len(alpha_bar.shape) < len(x.shape):
            alpha_bar = alpha_bar.unsqueeze(-1)
            alpha_bar_next = alpha_bar_next.unsqueeze(-1)
        sigma = 0.0  #  deterministic sampling
        c1 = torch.sqrt(alpha_bar_next)
        c2 = torch.sqrt(1 - alpha_bar_next - sigma ** 2)
        pred_x0 = x0
        dir_xt = torch.sqrt(1 - alpha_bar) * model_output
        x_next = c1 * pred_x0 + c2 * dir_xt

        return x_next, pred_x0

    def predict_eps_from_z(self, z, t, c, clip_denoised=True):
        model_output = self.model(z, t, c)
        alpha_bar = self.alphas_cumprod[t]
        while len(alpha_bar.shape) < len(z.shape):
            alpha_bar = alpha_bar.unsqueeze(-1)
        pred_x0 = (z - torch.sqrt(1 - alpha_bar) * model_output) / torch.sqrt(alpha_bar)

        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1., 1.)

        return model_output, pred_x0

    @torch.no_grad()
    def p_sample_loop_ddim(self, c, shape, n_steps, return_intermediates=False, log_every_t=100, clip_denoised=True):
        device = self.betas.device
        b = shape[0]
        x = torch.randn(shape, device=device)
        intermediates_x0 = [x]
        seq = torch.linspace(0, self.num_timesteps - 1, n_steps).flip(0).long()
        next_seq = torch.cat([seq[1:], torch.tensor([0])]).to(device)

        with tqdm(range(len(seq)), desc='Sampling t', total=len(seq)) as pbar:
            for i in pbar:
                t = torch.full((b,), seq[i], device=device, dtype=torch.long)
                t_next = torch.full((b,), next_seq[i], device=device, dtype=torch.long)

                x, x0 = self.p_sample_ddim(x, c, t, t_next, clip_denoised=clip_denoised)

                if i % log_every_t == 0 or i == len(seq) - 1:
                    intermediates_x0.append(x0)

        if return_intermediates:
            return x, intermediates_x0
        return x

    @torch.no_grad()
    def sample_ddim(self, c=None, batch_size=1, n_steps=50, return_intermediates=False, clip_denoised=True):
        """
        using DDIM in reverse sampling
        Args:
            c: context
            batch_size: batch size
            n_steps: sampling steps for DDIM
            return_intermediates: True/False return intermediated result
            clip_denoised: True/False
        """
        return self.p_sample_loop_ddim(c, [batch_size, self.channels] + list(self.latent_size), n_steps=n_steps,
                                       return_intermediates=return_intermediates,
                                       clip_denoised=clip_denoised)

    def test_step(self, batch, batch_idx):
        z, c, x, x_id = self.get_input(batch)

        """ sampling with diff_time in test stage """
        if self.opts.diff_time == self.num_timesteps:
            # print("Using DDPM sampling schedule")
            x_samples = self.sample(c=c, batch_size=self.opts.batch_size)
        elif self.opts.diff_time in range(0, self.num_timesteps):
            # print("Using DDIM sampling schedule")
            x_samples = self.sample_ddim(c=c, batch_size=self.opts.batch_size, n_steps=self.opts.diff_time)
        else:
            raise RuntimeError(f"Receive a wrong range of diff_time: {self.opts.diff_time}")
        img_samples = self.decode_first_stage(x_samples)

        x_rec = self.decode_first_stage(z)
        save_dir = self.opts.image_save_dir
        os.makedirs(save_dir, exist_ok=True)
        # create feature saving dir
        # fea_save_dir = os.path.join(save_dir, "feature")
        # os.makedirs(fea_save_dir, exist_ok=True)
        # create iamge saving dir
        img_save_dir = os.path.join(save_dir, "image")
        os.makedirs(img_save_dir, exist_ok=True)

        for each_idx in range(x.shape[0]):
            case_name = x_id[each_idx].split(".")[0]
            # unmark the following three lines if you want save latent prediction
            # z_0 = x_samples.data.cpu().numpy()[each_idx]
            # z_1 = z.data.cpu().numpy()[each_idx]
            # np.savez(os.path.join(fea_save_dir, f"{case_name}.npz"), z_0=z_0, z_1=z_1, label=c['c3'][each_idx].data.cpu().numpy()) 
            self.save_image(img_samples[each_idx], os.path.join(img_save_dir, f'{case_name}_gen.png'))
            self.save_image(x[each_idx], os.path.join(img_save_dir, f'{case_name}_gt.png'))
            self.save_image(x_rec[each_idx], os.path.join(img_save_dir, f'{case_name}_rec.png'))
        del z, c, x, x_samples, img_samples, x_rec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        print("All Done!")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def configure_optimizers(self):
        lr = self.opts.learning_rate
        params = list(self.model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)

        scheduler_config = {'warm_up_steps': [10000],
                            'cycle_lengths': [10000000000000],
                            'f_start': [0.1],
                            'f_max': [1.0],
                            'f_min': [0.5]}
        scheduler = LambdaLinearScheduler(**scheduler_config)
        print("Setting up LambdaLR scheduler...")
        scheduler = [
            {
                'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                'interval': 'step',
                'frequency': 1
            }]
        return [opt], scheduler

def main(opts):
    print(opts)
    model_cls = SCPFlowLDM if opts.predictor_type == "scpflow" else LDM
    model = model_cls(opts)
    if opts.command == "fit":
        train_loader = SeqFundusDatamodule(
            data_root=opts.data_root,
            batch_size=opts.batch_size,
            image_size=opts.image_size,
            num_workers=opts.num_workers,
            shuffle=True,
            drop_last=True,
            mode='train',
            structure_roi_size=opts.structure_roi_size,
            missing_visit_count=opts.missing_visit_count,
            missing_visit_strategy=opts.missing_visit_strategy,
            missing_seed=opts.missing_seed,
        )
        val_loader = SeqFundusDatamodule(
            data_root=opts.data_root,
            batch_size=opts.batch_size,
            image_size=opts.image_size,
            num_workers=opts.num_workers,
            shuffle=True,
            drop_last=True,
            mode='validation',
            structure_roi_size=opts.structure_roi_size,
            missing_visit_count=opts.missing_visit_count,
            missing_visit_strategy=opts.missing_visit_strategy,
            missing_seed=opts.missing_seed,
        )
        ckpt_callback = [
            # checkpoint metric differs for diffusion and scpflow
            ModelCheckpoint(
                monitor='val/loss' if opts.predictor_type == "scpflow" else 'psnr_mean',
                mode='min' if opts.predictor_type == "scpflow" else 'max',
                save_top_k=3,
                filename='best_model-flow-{epoch:03d}' if opts.predictor_type == "scpflow" else 'best_model-{psnr_mean:.2f}',
                save_weights_only=True
            ),
        ]

        trainer = pl.Trainer(**from_argparse_args(opts, callbacks=ckpt_callback))
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=[val_loader])
    else:
        test_loader = SeqFundusDatamodule(
            data_root=opts.data_root,
            batch_size=opts.batch_size,
            image_size=opts.image_size,
            num_workers=opts.num_workers,
            shuffle=False,
            drop_last=False,
            mode=opts.test_type,
            structure_roi_size=opts.structure_roi_size,
            missing_visit_count=opts.missing_visit_count,
            missing_visit_strategy=opts.missing_visit_strategy,
            missing_seed=opts.missing_seed,
        )

        model = model_cls(opts)
        path = opts.resume
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'],strict=False)
        trainer = pl.Trainer(**from_argparse_args(opts))

        if opts.predictor_type == "diffusion":
            # show sampling schedule
            if opts.diff_time == model.num_timesteps:
                print("! => Using DDPM sampling schedule")
            elif opts.diff_time in range(0, model.num_timesteps):
                print("! => Using DDIM sampling schedule")
            else:
                raise RuntimeError(f"Receive a wrong range of diff_time: {opts.diff_time}")
        else:
            print("! => Using SCP-Flow one-shot latent forecasting")

        # start testing
        trainer.test(model, test_loader)


if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    os.environ['TORCH_HOME'] = "./pre-trained"  # path to save torch pre-train ckpt
    print("Using", torch.cuda.device_count(), "GPUs!")
    initExperiment(opts)
    main(opts)
