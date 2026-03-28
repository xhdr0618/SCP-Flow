"""
VQGAN:
per-train & interface

dataset requirement:
Example of the sturcture of the pretrained image dataset (data_root)
Image naming pattern: *.extension

your/fundus_img/dataset_path/           # This is the base_dir
├── train/                              # Split directory for training data (LAG ACRIMA train)
│   ├── image001.jpg           # Image naming pattern: *.extension
│   ├── image102.png
│   └── ...
├── validation/                     # Split directory for validation data  (SIGF validation)
│   ├── validation001.jpg           # Image naming pattern: *.extension
│   ├── validation102.png
│   └── ...
└── test/                               # Split directory for testing data   (SIGF test)
    ├── test001.jpg
    ├── test102.png
    └── ...
    
"""
# -*- coding:utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import cv2
import tqdm

from datamodule.single_fundus_2D_datamodule import SingleFundusDatamodule
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
from ldm.modules.ema import LitEma
from utils.util import Logger
from metrics.calculate_metric import calculate_psnr, calculate_ssim
import os
import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Training VQGAN Model')
    parser.add_argument('--data_root', type=str, default='path_to_your_autoencoder_image_dataset')
    parser.add_argument('--result_root', type=str, default='path_to_your_result')
    parser.add_argument('--image_size', default=(256, 256))
    parser.add_argument("--command", default="fit")
    parser.add_argument("--max_epochs", default=300)
    parser.add_argument("--base_learning_rate", type=float, default=4.5e-6)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--devices', default=[1], type=list, help='List of GPU indices to use')
    parser.add_argument('--resume', default='', type=str, help='Path for checkpoint to load and resume')
    parser.add_argument("--ema", default=0, type=int, help='training with ema (1:true/0:false)')
    args = parser.parse_args()
    return args


class VQGAN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ddconfig = {
            'double_z': False,
            'z_channels': 4,
            'resolution': 512,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1, 2, 4, 4],
            'num_res_blocks': 2,
            'attn_resolutions': [],
            'dropout': 0.0
        }

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.embed_dim = ddconfig["z_channels"]
        n_embed = 16384
        self.quantize = VectorQuantizer(n_embed, self.embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], self.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, ddconfig["z_channels"], 1)

        lossconfig = dict(
            disc_conditional=False,
            disc_in_channels=3,
            disc_num_layers=2,
            disc_start=1,
            disc_weight=0.6,
            codebook_weight=1.0
        )
        self.loss = VQLPIPSWithDiscriminator(**lossconfig)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, x, return_pred_indices=False):
        quant, diff, (_, _, ind) = self.encode(x)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff


class VQGANTrainer:
    def __init__(self, model, args):
        self.args = args
        self.model = model
        self.logger = Logger(result_root=args.result_root)
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

        # Multi-GPU setup (DataParallel)
        # if torch.cuda.device_count() > 1 and len(args.devices) > 1:
        #     self.logger.print(f"Using {len(args.devices)} GPUs!")
        #     self.model = torch.nn.DataParallel(self.model, device_ids=args.devices)

        self.model.to(self.device)

        # Setup EMA
        self.use_ema = bool(args.ema)
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            self.logger.print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        # Optimizers
        lr_d = args.base_learning_rate
        lr_g = args.base_learning_rate
        self.opt_ae = torch.optim.Adam(
            list(self.model.encoder.parameters()) +
            list(self.model.decoder.parameters()) +
            list(self.model.quantize.parameters()) +
            list(self.model.quant_conv.parameters()) +
            list(self.model.post_quant_conv.parameters()),
            lr=lr_g, betas=(0.5, 0.9)
        )
        self.opt_disc = torch.optim.Adam(
            self.model.loss.discriminator.parameters(),
            lr=lr_d, betas=(0.5, 0.9)
        )

        # Training state
        self.start_epoch = 0
        self.epoch = self.start_epoch
        self.step = 0
        self.best_val_loss = float('inf')

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_psnr = []
        self.train_ssim = []
        self.val_psnr = []
        self.val_ssim = []

    def train(self, train_loader, val_loader):
        cudnn.benchmark = True

        # Load checkpoint if resuming
        if os.path.isfile(self.args.resume):
            self.load_checkpoint(self.args.resume)

        for epoch in range(self.start_epoch, self.args.max_epochs):
            self.epoch = epoch
            self.logger.print(f'Epoch: {epoch}')
            start_time = time.time()

            # Training loop
            self.model.train()
            for batch_idx, batch in enumerate(train_loader):
                x = batch['image'].to(self.device)

                # Forward pass
                xrec, qloss, ind = self.model(x, return_pred_indices=True)

                # Generator step
                self.opt_ae.zero_grad()
                aeloss, log_dict_ae = self.model.loss(
                    qloss, x, xrec,
                    optimizer_idx=0,
                    global_step=self.step,
                    last_layer=self.model.decoder.conv_out.weight,
                    split="train"
                )
                aeloss.backward()
                self.opt_ae.step()

                # Discriminator step
                self.opt_disc.zero_grad()
                discloss, log_dict_disc = self.model.loss(
                    qloss, x, xrec,
                    optimizer_idx=1,
                    global_step=self.step,
                    last_layer=self.model.decoder.conv_out.weight,
                    split="train"
                )
                discloss.backward()
                self.opt_disc.step()


                # Update EMA
                if self.use_ema:
                    self.model_ema(self.model)

                self.step += 1

                # Logging
                if self.step > 0 and self.step % 100 == 0:
                    self.logger.print(f"Step: {self.step}, AE Loss: {aeloss.item():.4f}, Disc Loss: {discloss.item():.4f}, total Loss {discloss.item():.4f} speeds: time: {round(time.time() - start_time, 2)}s/100 step")
                    self.train_losses.append([self.step, aeloss.item(), discloss.item()])
                    self.logger.draw_loss_fig(self.train_losses, self.val_losses)


            # Validation
                if self.step > 0 and self.step % 1000 == 0:
                    val_loss = self.validate(val_loader, save_img=True, loader_type="val")
                    _ = self.validate(train_loader, save_img=True, loader_type="train")
                    self.val_losses.append([self.step, val_loss])
                    self.logger.draw_loss_fig(self.train_losses, self.val_losses)

                    self.logger.draw_psnr_fig(self.train_psnr, type="train")
                    self.logger.draw_ssim_fig(self.train_ssim, type="train")
                    self.logger.draw_psnr_fig(self.val_psnr, type="val")
                    self.logger.draw_ssim_fig(self.val_ssim, type="val")

                    # Save checkpoint if best validation loss
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(epoch, is_best=True)

                # Regular checkpoint saving
                if self.step > 0 and self.step % 2000 == 0:
                    self.save_checkpoint(epoch)

            self.logger.print(f"Epoch {epoch} completed!")

    def validate(self, val_loader, save_img=False, loader_type="val"):
        self.model.eval()
        val_loss = 0
        max_batches = 3
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx < max_batches:  # consider three batches of dataloader
                    x = batch['image'].to(self.device)
                    xrec, qloss, ind = self.model(x, return_pred_indices=True)

                    aeloss, _ = self.model.loss(
                        qloss, x, xrec,
                        optimizer_idx=0,
                        global_step=self.step,
                        last_layer=self.model.decoder.conv_out.weight,
                        split="val"
                    )
                    val_loss += aeloss.item()
                    if save_img:
                        each_psnr = []
                        each_ssim = []
                        for idx in range(x.shape[0]):
                            self.logger.save_img_concate(gt=batch['image'][idx], gene=xrec[idx], epoch=self.epoch, steps=self.step, loader_type=loader_type,  name=f"{batch['image_id'][idx]}_gt_gene")
                            each_psnr.append(calculate_psnr(batch['image'][idx], xrec[idx]))
                            each_ssim.append(calculate_ssim(batch['image'][idx], xrec[idx]))
            if save_img and (loader_type == "val"):
                self.val_psnr.append([self.step, np.mean(each_psnr), np.std(each_psnr)])
                self.val_ssim.append([self.step, np.mean(each_ssim), np.std(each_ssim)])
            if save_img and (loader_type == "train"):
                self.train_psnr.append([self.step, np.mean(each_psnr), np.std(each_psnr)])
                self.train_ssim.append([self.step, np.mean(each_ssim), np.std(each_ssim)])
            self.logger.print(f"{loader_type} - PSNR mean:{np.mean(each_psnr)} std:{np.std(each_psnr)} SSIM mean:{np.mean(each_ssim)} std:{np.std(each_ssim)} ")


        if loader_type == "val":
            val_loss /= min(len(val_loader),max_batches)
            self.logger.print(f"Validation Loss: {val_loss:.4f}")
        return val_loss

    def test(self, test_loader):
        """
        check all testing results
        """
        self.model.eval()
        save_dir = os.path.join(self.args.result_root, 'test_results')
        os.makedirs(save_dir, exist_ok=True)

        self.logger.print(f"Saving generated images at {save_dir}")

        with torch.no_grad():
            for batch_idx, batch in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
                x = batch['image'].to(self.device)
                xrec, _ = self.model(x, return_pred_indices=False)
                for each_idx in range(x.shape[0]):
                    img_name = f"img_{batch_idx}_{each_idx}.png"
                    self.save_image(xrec[each_idx], os.path.join(save_dir, img_name))
                    
        self.logger.print(f"All generated images saved to {save_dir}")



    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'state_dict': self.model.state_dict()
        }

        filename = os.path.join(self.args.result_root, f'checkpoint_epoch_{epoch}_step_{self.step}.ckpt')
        torch.save(checkpoint, filename)
        if is_best:
            best_filename = os.path.join(self.args.result_root, 'model_lowest_val_loss.ckpt')
            self.logger.print(f"save lowest val loss checkpoint: epoch-{epoch} steps-{self.step}")
            torch.save(checkpoint, best_filename)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.logger.print(f"Loaded checkpoint from path {checkpoint_path}")

    def save_test_images(self, reconstructed, image_ids):
        save_dir = os.path.join(self.args.result_root, 'test_results')
        os.makedirs(save_dir, exist_ok=True)

        for i in range(reconstructed.size(0)):
            img = reconstructed[i].cpu().float().numpy()
            img = (img * 255).transpose(1, 2, 0).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, f'{image_ids[i]}'), img)

    def save_image(self, img, path):
        img = (img * 255).permute(1, 2, 0)
        img[img > 255] = 255
        img[img < 0] = 0
        img = np.uint8(img.data.cpu().numpy())
        cv2.imwrite(path, img)


def main():
    args = parse_args()
    # initExperiment(args)
    if not os.path.exists(args.result_root):
        os.makedirs(args.result_root)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create model and trainer
    model = VQGAN()
    trainer = VQGANTrainer(model, args)


    if args.command == "fit":
        # Create data loaders
        train_loader = SingleFundusDatamodule(
            data_root=os.path.join(args.data_root, 'train'),
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=True
        )
        val_loader = SingleFundusDatamodule(
            data_root=os.path.join(args.data_root, 'validation'),
            batch_size=1,
            image_size=args.image_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False
        )

        trainer.logger.print(f'Train dataset length:{len(train_loader.dataset)}')
        trainer.logger.print(f'Validation dataset length: {len(val_loader.dataset)}')

        # Start training
        trainer.train(train_loader, val_loader)

    else:  # Test mode
        test_loader = SingleFundusDatamodule(
            data_root=os.path.join(args.data_root, 'test'),
            batch_size=args.batch_size,
            image_size=args.image_size,
            shuffle=False,
            drop_last=False
        )


        # Load checkpoint for testing
        checkpoint_path = args.resume
        trainer.load_checkpoint(checkpoint_path)
        trainer.logger.print(f'Test dataset length: {len(test_loader.dataset)}')
        trainer.test(test_loader)

if __name__ == '__main__':
    os.environ['TORCH_HOME'] = "./pre-trained"  # path to save torch pre-train ckpt
    main()