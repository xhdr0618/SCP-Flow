import cv2 as cv
import numpy as np
import torch

import os
import datetime
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, result_root):
        self.result_root = result_root
        if not os.path.exists(self.result_root):
            os.makedirs(self.result_root)

        self.log_file_name = f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def show_model(self, model):
        message = f"=> Using model: {str(model.__class__)}"
        self.print(message)

    def print(self, message):
        timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        log_message = f"{timestamp} {message}"
        print(log_message)
        with open(os.path.join(self.result_root, self.log_file_name), "a") as f:
            f.write(log_message + "\n")

    def save_img(self, img, epoch, steps, name):
        img = self._ch_img_to_cv(img)
        abs_path = os.path.join(self.result_root, f"epoch_{str(epoch)}_steps_{str(steps)}", name + ".png")
        cv.imwrite(abs_path, img)

    def save_img_concate(self, gene, gt, epoch, steps, loader_type, name):
        gene = self._ch_img_to_cv(gene)
        gt = self._ch_img_to_cv(gt)
        img = np.concatenate((gt,gene), axis=1)
        abs_path = os.path.join(self.result_root,  f"epoch_{str(epoch)}_steps_{str(steps)}", loader_type)
        if not os.path.exists(abs_path):
            os.makedirs(abs_path)
        cv.imwrite(os.path.join(abs_path, name + ".png"), img)

    @staticmethod
    def _ch_img_to_cv(img):
        """ change img to cv format """
        """ convert tensor to narray """
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
        """ convert channel """
        img = (img*255).transpose(1, 2, 0).astype(np.uint8)
        return img

    def draw_loss_fig(self, train_loss, val_loss):
        train_loss, val_loss = np.array(train_loss), np.array(val_loss)
        plt.clf()
        file_name = "train_val_loss_total.png"
        fig_path = os.path.join(self.result_root, file_name)
        plt.plot(train_loss[:, 0], train_loss[:, 1]+train_loss[:, 2], color='#1f77b4')
        if val_loss.shape[0] != 0:
            plt.plot(val_loss[:,0], val_loss[:, 1], color='#d62728')
        plt.xlabel('steps')
        plt.ylabel(f'train | val loss')
        plt.savefig(fig_path)

        plt.clf()
        file_name = "val_loss_total.png"
        fig_path = os.path.join(self.result_root, file_name)
        if val_loss.shape[0] != 0:
            plt.plot(val_loss[:, 0], val_loss[:, 1], color='#d62728')
        plt.xlabel('steps')
        plt.ylabel(f'val loss')
        plt.savefig(fig_path)

        plt.clf()
        file_name = "train_loss_detail.png"
        fig_path = os.path.join(self.result_root, file_name)
        plt.plot(train_loss[:, 0], train_loss[:, 1], color='#F4A460', label='AE Loss')
        plt.plot(train_loss[:, 0], train_loss[:, 2], color='#90EE90', label='Disc Loss')
        plt.plot(train_loss[:, 0], train_loss[:, 1] + train_loss[:, 2], color='#1f77b4', label="Total Loss")
        plt.legend()
        plt.xlabel('steps')
        plt.ylabel(f'train loss detail')
        plt.savefig(fig_path)

    def draw_psnr_fig(self, psnr, type="train"):
        psnr = np.array(psnr)
        steps, psnr_mean, psnr_std = psnr[:, 0], psnr[:, 1], psnr[:, 2]
        plt.clf()
        plt.plot(steps, psnr_mean, color='#006400')
        plt.fill_between(steps, psnr_mean - psnr_std, psnr_mean + psnr_std, color="#90EE90", alpha=0.2)
        plt.xlabel('steps')
        plt.ylabel(f'psnr(mean of 3 {type} batches)')
        plt.grid()
        file_name = f"{type}_psnr.png"
        fig_path = os.path.join(self.result_root, file_name)
        plt.savefig(fig_path)

    def draw_ssim_fig(self, ssim, type="train"):
        ssim = np.array(ssim)
        steps, ssim_mean, ssim_std = ssim[:, 0], ssim[:, 1], ssim[:, 2]
        plt.clf()
        plt.plot(steps, ssim_mean, color='#800000')
        plt.fill_between(steps, ssim_mean - ssim_std, ssim_mean + ssim_std, color="#F08080", alpha=0.2)
        plt.xlabel('steps')
        plt.ylabel(f'ssim (mean of 3 {type} batches)')
        plt.grid()
        file_name = f"{type}_ssim.png"
        fig_path = os.path.join(self.result_root, file_name)
        plt.savefig(fig_path)


def draw_psnr_fig(psnr, saving_path):
    psnr = np.array(psnr)
    epoch, psnr_mean, psnr_std = psnr[:,0], psnr[:,1], psnr[:,2]
    plt.clf()
    plt.plot(epoch, psnr_mean, color='#006400')
    plt.fill_between(epoch, psnr_mean - psnr_std, psnr_mean + psnr_std, color="#90EE90", alpha=0.2)
    plt.xlabel('epoch')
    plt.ylabel('psnr')
    plt.grid()
    plt.savefig(saving_path)


def draw_ssim_fig(ssim, saving_path):
    ssim = np.array(ssim)
    epoch, ssim_mean, ssim_std = ssim[:,0], ssim[:,1], ssim[:,2]
    plt.clf()
    plt.plot(epoch, ssim_mean, color='#800000')
    plt.fill_between(epoch, ssim_mean - ssim_std, ssim_mean + ssim_std, color="#F08080", alpha=0.2)
    plt.xlabel('epoch')
    plt.ylabel('ssim')
    plt.grid()
    plt.savefig(saving_path)

def draw_loss_fig(loss, saving_path):
    loss = np.array(loss)
    steps, loss_mean = loss[:, 0], loss[:, 1]
    plt.clf()
    plt.plot(steps, loss_mean, color='#483D8B')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.grid()
    plt.savefig(saving_path)